#pragma once
#include <JuceHeader.h>

class MultibandSCAudioProcessor : public juce::AudioProcessor
{
public:
    using APVTS = juce::AudioProcessorValueTreeState;

    MultibandSCAudioProcessor();
    ~MultibandSCAudioProcessor() override = default;

    // ==== AudioProcessor ====
    const juce::String getName() const override { return "MultibandSC"; }
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;

    // Programy/presety – minimalnie
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    // State
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Parametry
    APVTS& getVTS() { return parameters; }
    static APVTS::ParameterLayout createLayout();

private:
    // ====== DSP ======
    struct OnePoleEnv
    {
        void prepare (double sr, float attMs, float relMs)
        {
            sampleRate = sr; setTimes (attMs, relMs); z = 0.0f;
        }
        void setTimes (float attMs, float relMs)
        {
            // prosty detektor obwiedni (peak) z różnymi stałymi
            attack = std::exp (-1.0 / (0.001 * attMs * sampleRate + 1e-6));
            release = std::exp (-1.0 / (0.001 * relMs * sampleRate + 1e-6));
        }
        inline float process (float x)
        {
            x = std::fabs (x);
            const float a = (x > z ? attack : release);
            z = (1.0f - a) * x + a * z;
            return z;
        }
        double sampleRate = 44100.0;
        float attack = 0.9f, release = 0.99f, z = 0.0f;
    };

    struct Band
    {
        // crossovers (Linkwitz-Riley 24 dB/okt = 2x Butterworth 12 dB)
        juce::dsp::LinkwitzRileyFilter<float> low, high; // użyte zależnie od pasma
        // obwiednia sidechain
        OnePoleEnv envL, envR;
        // param cache
        float thresh = -24.0f;  // dB
        float ratio  = 4.0f;    // :1
        float attackMs  = 10.0f;
        float releaseMs = 100.0f;
        float makeupDb  = 0.0f;

        void reset()
        {
            low.reset(); high.reset();
            envL.z = envR.z = 0.0f;
        }
    };

    // 3 pasma
    Band bands[3];

    // Filtry zwrotnicy na główny tor (na każdy kanał osobno)
    juce::dsp::ProcessorDuplicator<juce::dsp::LinkwitzRileyFilter<float>, juce::dsp::LinkwitzRileyFilter<float>> xLow;
    juce::dsp::ProcessorDuplicator<juce::dsp::LinkwitzRileyFilter<float>, juce::dsp::LinkwitzRileyFilter<float>> xHigh;

    // Bufory robocze
    juce::AudioBuffer<float> bandBuf[3];
    juce::AudioBuffer<float> scBandBuf[3];

    // Parametry i stan
    APVTS parameters;
    double sr = 44100.0;
    int maxBlock = 512;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MultibandSCAudioProcessor)
};
