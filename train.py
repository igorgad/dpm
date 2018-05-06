

import tensorflow as tf
import numpy as np
import vstRender.vstRender as vr
import matplotlib.pyplot as plt
import time

import models.dnn as md

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('record_file', type=str, default='./records.tfrecord', help='tf record file to use')
parser.add_argument('--plugin_file', type=str, default=[], help='plugin to extract features. Must match the plugin used to generate the tfrecord file')
parser.add_argument('--logdir', default='./logs', type=str, help='directory to log')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_steps', default=500000, type=int, help='number of training steps')

parser.add_argument('--fft_length', default=510, type=int, help='fft length')
parser.add_argument('--fft_step', default=512, type=int, help='fft step')
parser.add_argument('--fft_frame_length', default=1024, type=int, help='fft frame length in samples')


def parse_features_and_decode(tf_example, features):
    parsed_features = tf.parse_single_example(tf_example, features)
    parsed_features['samples'] = tf.decode_raw(parsed_features['samples'], tf.float32)
    parsed_features['l_samples'] = tf.decode_raw(parsed_features['l_samples'], tf.float32)
    return parsed_features


def prepare_examples(parsed_features):
    return parsed_features, tf.reshape(parsed_features['l_samples'], [-1])


def input_fn(dataset_file, dataset_features, batch_size):
    tfdataset = tf.data.TFRecordDataset(dataset_file)
    tfdataset = tfdataset.map(lambda feat: parse_features_and_decode(feat, dataset_features), num_parallel_calls=4)
    # tfdataset = tfdataset.map(lambda feat: compute_stft_feature(feat, args.fft_frame_length, args.fft_step, args.fft_length), num_parallel_calls=4)
    tfdataset = tfdataset.map(prepare_examples, num_parallel_calls=4).batch(batch_size)
    return tfdataset.make_one_shot_iterator().get_next()


def audio_input_fn(audio_file, audio_samples, dataset_features, batch_size):
    audio_binary = tf.read_file(audio_file)
    waveform = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='mp3', samples_per_second=44100, channel_count=1)
    waveform = tf.contrib.signal.frame(waveform, audio_samples, audio_samples // 2)
    tdata = tf.data.Dataset.from_tensor_slices(waveform)



args = parser.parse_args('/home/pepeu/workspace/Dataset/dpm_10240.tfrecord --plugin_file /usr/lib/vst/ZamEQ2-vst.so'.split())
args.audio_samples = int(args.record_file.split('/')[-1].split('.')[0].split('_')[-1])
vst_render = vr.vstRender(44100, 1024)
vst_render.loadPlugin(args.plugin_file)

nparams = vst_render.getPluginParameterSize()
params_description = vst_render.getPluginParametersDescription()
params_description = [[int(i.split(':')[0]), i.split(':')[1].replace(' ',''), float(i.split(':')[2]), int(i.split(':')[3])] for i in params_description.split('\n')[:-1]]

stft_shape = [args.audio_samples // args.fft_step - 1, args.fft_length // 2 + 1]
stft_size = int(np.prod(stft_shape))

dataset_features = {'samples': tf.FixedLenFeature([], tf.string), 'l_samples': tf.FixedLenFeature([], tf.string)}
feature_columns = [tf.feature_column.numeric_column(key='stfts/mag', shape=stft_shape)]
for p in range(nparams):
    dataset_features[params_description[p][1]] = tf.FixedLenFeature([], tf.float32)
    feature_columns.append(tf.feature_column.numeric_column(key=params_description[p][1], shape=(1,)))

# model = tf.estimator.DNNRegressor(feature_columns=feature_columns,
#                                   hidden_units=[stft_size, stft_size],
#                                   label_dimension=stft_size,
#                                   model_dir=args.logdir)

model = tf.estimator.Estimator(
    model_fn=md.model_func,
    model_dir=args.logdir,
    params={'feature_columns': feature_columns,
            'hidden_units': [stft_size, stft_size],
            'frame_length': args.fft_frame_length,
            'frame_step': args.fft_step,
            'fft_length': args.fft_length,
            'fft_nblocks': stft_shape[0],
            'label_size': stft_size})

model.train(input_fn=lambda: input_fn(args.record_file, dataset_features, args.batch_size), steps=args.train_steps)
pred = model.predict(input_fn=lambda: input_fn(args.record_file, dataset_features, args.batch_size))


# if __name__ == '__main__':
#     tf.logging.set_verbosity(tf.logging.INFO)
#     tf.app.run(main)