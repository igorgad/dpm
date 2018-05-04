
#include "vstRender.h"

vstRender::vstRender(int sr, int bs) {
	sampleRate = sr;
    bufferSize = bs;
    plugin = nullptr;
}

vstRender::~vstRender() {
    if (plugin != nullptr) {
        plugin->releaseResources();
        delete plugin;
    }
}

bool vstRender::loadPlugin (char *path) {
    juce::OwnedArray<PluginDescription> pluginDescriptions;
    juce::KnownPluginList pluginList;
    juce::AudioPluginFormatManager pluginFormatManager;
    String errorMessage;

    pluginFormatManager.addDefaultFormats();

    for (int i = pluginFormatManager.getNumFormats(); --i >= 0;) {
        pluginList.scanAndAddFile (String (path), true, pluginDescriptions, *pluginFormatManager.getFormat(i));
    }

    // If there is a problem here first check the preprocessor definitions
    // in the projucer are sensible - is it set up to scan for plugin's?
    jassert (pluginDescriptions.size() > 0);
    if (plugin != nullptr) delete plugin;

    plugin = pluginFormatManager.createPluginInstance (*pluginDescriptions[0], sampleRate, bufferSize, errorMessage);

    if (plugin != nullptr) {
        plugin->prepareToPlay (sampleRate, bufferSize);
        plugin->setNonRealtime (true);

        fillAvailablePluginParameters (pluginParameters);
        return true;
    }

    std::cout << "RenderEngine::loadPlugin error: " << errorMessage.toStdString() << std::endl;
    return false;
}

const float* vstRender::renderAudio (float* buffer, int sampleLength) {
    juce::MidiBuffer midi;    

    for (const auto& parameter : pluginParameters)
        plugin->setParameter (parameter.first, parameter.second);

    plugin->prepareToPlay (sampleRate, bufferSize);
    int numBuffers = std::ceil(sampleLength / bufferSize);
    for (int b = 0; b < numBuffers; b++) {
        juce::AudioSampleBuffer outputBuffer(&buffer, plugin->getTotalNumOutputChannels(), b * bufferSize, bufferSize);        
        plugin->processBlock (outputBuffer, midi);    
    }

    juce::AudioSampleBuffer outputBuffer(&buffer, plugin->getTotalNumOutputChannels(), 0, sampleLength);
    return outputBuffer.getReadPointer(0);
}

const size_t vstRender::getPluginParameterSize() {
    return pluginParameters.size();
}

const char* vstRender::getPluginParametersDescription() {
    String parameterListString ("");

    if (plugin != nullptr) {
        std::ostringstream ss;

        for (const auto& pair : pluginParameters) {
            ss << std::setw (3) << std::setfill (' ') << pair.first;

            const String name = plugin->getParameterName (pair.first);
            const String val (plugin->getParameterDefaultValue (pair.first));
            const String steps (plugin->getParameterNumSteps (pair.first));
            const String index (ss.str());

            parameterListString = parameterListString + index + ":" + name + ":" + val + ':' + steps + "\n";

            ss.str ("");
            ss.clear();
        }
    }
    else {
        std::cout << "Please load the plugin first!" << std::endl;
    }

    return (const char*) parameterListString.toUTF8();
}

void vstRender::setParams (const PluginParams params) {
    const size_t currentParameterSize = pluginParameters.size();
    const size_t newPatchParameterSize = params.size();

    if (currentParameterSize == newPatchParameterSize) {
        pluginParameters = params;
    }
    else {
        std::cout << "RenderEngine::setPatch error: Incorrect params size!" <<
        "\n- Current size:  " << currentParameterSize <<
        "\n- Supplied size: " << newPatchParameterSize << std::endl;
    }
}

const PluginParams vstRender::getParams() {
    return pluginParameters;
}

void vstRender::fillAvailablePluginParameters (PluginParams& params) {
    params.clear();
    params.reserve (plugin->getNumParameters());

    int usedParameterAmount = 0;
    for (int i = 0; i < plugin->getNumParameters(); ++i) {
        // Ensure the parameter is not unused.
        if (plugin->getParameterName(i) != "Param") {
            ++usedParameterAmount;
            params.push_back (std::make_pair (i, plugin->getParameterDefaultValue(i)));
        }
    }
    params.shrink_to_fit();
}
