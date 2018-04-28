

import tensorflow as tf
import numpy as np
import pandas as pd
import vstRender.vstRender as vr
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('plugin_file', type=str, default=[], help='plugin to model')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')


def compute_log_spectogram(signals, log_offset=1e-6):
    stfts = tf.contrib.signal.stft(signals, frame_length=1024, frame_step=512, fft_length=1024)
    mag = tf.abs(stfts)
    return tf.log(mag + log_offset)


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


def generate_uniform_noise(sample_length):
    return np.random.rand(sample_length).astype(np.float32)


args = parser.parse_args('/usr/lib/vst/ZamTube-vst.so'.split())
vst_render = vr.vstRender(44100, 1024)
vst_render.loadPlugin(args.plugin_file)

nparams = vst_render.getPluginParameterSize()
param_description = vst_render.getPluginParametersDescription()


noise = generate_uniform_noise(1024)
noise1 = noise.copy()
noise2 = noise.copy()

vst_render.setParams(((0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5), (5, 0.0), (6, 0.5))) # OUTPUT = 0
r = vst_render.renderAudio(noise1)

vst_render.setParams(((0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5), (5, 1.0), (6, 0.5))) # OUTPUT = 1
r = vst_render.renderAudio(noise2)



# if __name__ == '__main__':
#     tf.logging.set_verbosity(tf.logging.INFO)
#     tf.app.run(main)