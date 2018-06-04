#! /usr/bin/env python3
# Author: Scott Hawley
# Source: Igor Gadelha  https://github.com/igorgad/dpm/blob/master/prepare_data.py

# Used simple echo plugin from Faust examples, https://github.com/grame-cncm/faust/blob/master-dev/examples/delayEcho/echo.dsp
# Pasted source into Faust Online Compiler https://faust.grame.fr/onlinecompiler/
# and Generated exec file for Linux, 'vst-64bits' architecture

# Main dependencies:
#   vstRender by Igor Gadelha https://github.com/igorgad/dpm/tree/master/vstRender
#   librosa by Brian McFee et al ('pip install librosa')


import numpy as np
import vstRender.vstRender as vr
import librosa
import argparse
import os

class Plugin():
    def __init__(self, plugin_file,sr=44100):
        # make sure the plugin file exists
        assert os.path.isfile(plugin_file), "Error, plugin file "+plugin_file+" not found"
        self.vst = vr.vstRender(sr, 512)
        self.vst.loadPlugin(plugin_file)

        # get vst parameters ('knob' names, etc)
        self.nparams = self.vst.getPluginParameterSize()
        self.params_description = self.vst.getPluginParametersDescription()
        self.params_description = [[int(i.split(':')[0]), i.split(':')[1].replace(' ', ''), float(i.split(':')[2]), int(i.split(':')[3])]
                              for i in self.params_description.split('\n')[:-1]]

    def run(self, samples, params):
        """where the actual audio processing call happens"""
        pidx = params[0]    # list of parameter ('knob') indices
        pval = params[1]    # list of settings for each parameter

        # match param indices with param values
        parg = tuple([(int(i), float(j)) for i, j in zip(pidx, pval)])

        self.vst.setParams(parg)       # set plugin parameters
        audio = samples.copy()         # renderAudio operates 'in place', so make a copy
        self.vst.renderAudio(audio)
        return audio.astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin_file', type=str, default='./faust_delay.so', help='plugin to model')
    parser.add_argument('--audio_file', default='./music.wav', type=str, help='audio file to read from')
    parser.add_argument('--outfile', default='./vst_out.wav', type=str, help='output file to create')
    parser.add_argument('--sr', default=44100, type=int, help='sample rate in Hertz')

    args = parser.parse_args()

    print("Initializing plugin file",args.plugin_file)
    plugin = Plugin(args.plugin_file)     # load and initialize the plugin
    print("   Plugin: nparams =",plugin.nparams)
    print("   Plugin: params_description =",plugin.params_description)

    # make sure the audio file exists
    assert os.path.isfile(args.audio_file), "Error, audio file "+args.audio_file+" not found"

    print ('Reading audio from',args.audio_file)
    audio_in, sr = librosa.core.load(args.audio_file, args.sr, mono=True)

    # set parameter values (randomly)
    pidx = np.arange(plugin.nparams)           # indices for parameters
    pval = np.random.random([plugin.nparams])  # values for those parameters
    params = [pidx, pval]

    audio_out = plugin.run(audio_in, params)   # apply the plugin to input audio

    print("Writing output to",args.outfile)
    librosa.output.write_wav(args.outfile, audio_out, args.sr)
