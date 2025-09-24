#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

// Na start używamy GenericAudioProcessorEditor (auto-GUI).
// Jeśli chcesz custom UI, podmień createEditor() w procesorze na tę klasę i rozwiń poniżej.

class MultibandSCAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit MultibandSCAudioProcessorEditor (MultibandSCAudioProcessor& p) : juce::AudioProcessorEditor (&p)
    {
        setSize (520, 360);
        addAndMakeVisible (generic.reset (new juce::GenericAudioProcessorEditor (p)));
    }
    void resized() override
    {
        if (generic) generic->setBounds (getLocalBounds());
    }
private:
    std::unique_ptr<juce::GenericAudioProcessorEditor> generic;
};
