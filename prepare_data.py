
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import vstRender.vstRender as vr
import multiprocessing

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('plugin_file', type=str, default=[], help='plugin to model')
parser.add_argument('--audio_file', default=[], type=str, help='audio file to create examples')
parser.add_argument('--outfile', default='./records.tfrecord', type=str, help='tfrecord file to create')
parser.add_argument('--audio_samples', default=5120, type=int, help='duration of audio in samples for each example')
parser.add_argument('--nexamples', default=100000, type=int, help='number of examples')
parser.add_argument('--use_pink_noise', default=False, type=bool, help='use pink noise instead of uniform noise')


process_batch = 500
pool_size = 4
vst = None


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def generate_pink_noise(sample_length, ncols=16):
    array = np.empty((sample_length, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(sample_length)

    # the total number of changes is nrows
    n = sample_length
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(sample_length, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return np.float32(total.values)


def generate_uniform_noise(batch_size, sample_length):
    return np.random.random([batch_size,sample_length]).astype(np.float32) - 0.5


def generate_from_audio(audio_file, samples_length):
    audio_binary = tf.read_file(audio_file)
    waveform = tf.reduce_mean(tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='wav', samples_per_second=44100, channel_count=2), axis=1)
    waveform = tf.contrib.signal.frame(waveform, samples_length, samples_length)
    return waveform


def init_vst(sr, bf, plugin_file):
    global vst
    vst = vr.vstRender(sr, bf)
    vst.loadPlugin(plugin_file)


def vst_process_samples(params):
    samples = params[0]
    pidx = params[1]
    pval = params[2]

    parg = tuple([(int(i), float(j)) for i, j in zip(pidx, pval)])
    vst.setParams(parg)
    vst.renderAudio(samples)
    return samples


# args = parser.parse_args('/usr/lib/vst/ZamEQ2-vst.so --outfile /home/pepeu/workspace/Dataset/dpm.tfrecord'.split())
args = parser.parse_args('/usr/lib/vst/ZamEQ2-vst.so --audio_file ./music.wav --outfile /home/pepeu/workspace/Dataset/dpm_music.tfrecord'.split())
args.outfile = args.outfile.replace('.tfrecord', '_%d.tfrecord' % args.audio_samples)

pool = multiprocessing.Pool(initializer=init_vst, initargs=(44100,512, args.plugin_file), processes=pool_size)
vst = vr.vstRender(44100, 512)
vst.loadPlugin(args.plugin_file)

nparams = vst.getPluginParameterSize()
params_description = vst.getPluginParametersDescription()
params_description = [[int(i.split(':')[0]), i.split(':')[1].replace(' ', ''), float(i.split(':')[2]), int(i.split(':')[3])]
                      for i in params_description.split('\n')[:-1]]

writer = tf.python_io.TFRecordWriter(args.outfile)
sess = tf.Session()

if args.audio_file:
    print ('Using audio file to create examples...')
    args.nexamples = args.nexamples // process_batch
    try:
        for ex in range(args.nexamples):
            st = time.time()

            input_samples = sess.run(generate_from_audio(args.audio_file, args.audio_samples))
            process_batch = input_samples.shape[0] // 4
            vst_samples = input_samples.copy()
            pval = np.random.random([nparams])
            pidx = np.arange(nparams)

            params = []
            for b in range(process_batch):
                params.append([vst_samples[b, :], pidx, pval])

            vst_samples = np.array(pool.map(vst_process_samples, params)).astype(np.float32)

            features = {}
            for b in range(process_batch):
                # features['audio_length'] = int64_feature(args.audio_samples)
                features['vst_samples'] = bytes_feature(vst_samples[b, :].tostring())
                features['input_samples'] = bytes_feature(input_samples[b, :].tostring())
                for p in enumerate(params_description):
                    features[p[1][1]] = floats_feature(pval[p[0]])

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())

            print('processed step ex: ' + str(ex) + ' from ' + str(args.nexamples) + ' in ' + str(time.time() - st))
    finally:
        writer.close()

else:
    print('Generating examples from noise...')
    try:
        for ex in range(args.nexamples // process_batch):
            st = time.time()

            input_samples = generate_uniform_noise(process_batch, args.audio_samples)
            vst_samples = input_samples.copy()
            pval = np.random.random([process_batch, nparams])
            pidx = np.arange(nparams)

            params = []
            for b in range(process_batch):
                params.append([vst_samples[b,:], pidx, pval[b,:]])

            vst_samples = np.array(pool.map(vst_process_samples, params)).astype(np.float32)

            features = {}
            for b in range(process_batch):
                # features['audio_length'] = int64_feature(args.audio_samples)
                features['vst_samples'] = bytes_feature(vst_samples[b, :].tostring())
                features['input_samples'] = bytes_feature(input_samples[b, :].tostring())
                for p in enumerate(params_description):
                    features[p[1][1]] = floats_feature(pval[b,p[0]])

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())

            print('processed step ex: ' + str(ex) + ' from ' + str(args.nexamples // process_batch) + ' in ' + str(time.time() - st))

    finally:
        writer.close()
