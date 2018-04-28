
#ifndef VSTRENDER_H
#define VSTRENDER_H

#include <array>
#include <iomanip>
#include <sstream>
#include <string>
#include "../JuceLibraryCode/JuceHeader.h"

typedef std::vector<std::pair<int, float>>  PluginParams;

class vstRender {
public:
    vstRender (int sr, int bs);
    ~vstRender();

    bool loadPlugin (char *path);
    const float* renderAudio (float *buffer, int sampleLength);

    const size_t getPluginParameterSize();
    const char* getPluginParametersDescription();
    void setParams (const PluginParams params);
    const PluginParams getParams();

private:
    void fillAvailablePluginParameters (PluginParams& params);

    double                      sampleRate;
    int                         bufferSize;

    juce::AudioPluginInstance*  plugin;
    PluginParams                pluginParameters;
};


#endif  // VSTRENDER_H