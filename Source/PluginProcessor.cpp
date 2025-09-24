#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace IDs {
    // crossover
    static constexpr auto xLow  = "xLowHz";
    static constexpr auto xHigh = "xHighHz";
    // global
    static constexpr auto bypass = "globalBypass";
    static constexpr auto scMode = "sidechainMode"; // 0=self, 1=external

    // per-band params
    static constexpr auto th1 = "thLo";  static constexpr auto ra1 = "raLo";
    static constexpr auto at1 = "atLo";  static constexpr auto re1 = "reLo";  static constexpr auto mk1 = "mkLo";
    static constexpr auto th2 = "thMi";  static constexpr auto ra2 = "raMi";
    static constexpr auto at2 = "atMi";  static constexpr auto re2 = "reMi";  static constexpr auto mk2 = "mkMi";
    static constexpr auto th3 = "thHi";  static constexpr auto ra3 = "raHi";
    static constexpr auto at3 = "atHi";  static constexpr auto re3 = "reHi";  static constexpr auto mk3 = "mkHi";
}

MultibandSCAudioProcessor::MultibandSCAudioProcessor()
: AudioProcessor (BusesProperties()
                    .withInput  ("Input",     juce::AudioChannelSet::stereo(), true)
                    .withOutput ("Output",    juce::AudioChannelSet::stereo(), true)
                    .withInput  ("Sidechain", juce::AudioChannelSet::stereo(), true) // aux bus
                 ),
  parameters (*this, nullptr, "PARAMS", createLayout())
{
}

bool MultibandSCAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    auto in  = layouts.getChannelSet (true, 0);
    auto out = layouts.getChannelSet (false, 0);
    if (in.isDisabled() || out.isDisabled()) return false;
    if (in != out) return false;
    // sidechain może być mono/stereo (tu oczekujemy stereo)
    return (in == juce::AudioChannelSet::mono() || in == juce::AudioChannelSet::stereo());
}

void MultibandSCAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    sr = sampleRate; maxBlock = samplesPerBlock;

    juce::dsp::ProcessSpec spec { sr, (juce::uint32) samplesPerBlock, (juce::uint32) getTotalNumOutputChannels() };

    // Ustawienia LR cross
    xLow.reset();  xHigh.reset();
    xLow.state->type  = juce::dsp::LinkwitzRileyFilterType::lowpass;
    xHigh.state->type = juce::dsp::LinkwitzRileyFilterType::highpass;
    xLow.prepare (spec); xHigh.prepare (spec);

    for (auto& b : bands) b.reset();

    // Bufory pomocnicze
    for (int i=0; i<3; ++i) {
        bandBuf[i].setSize (getTotalNumOutputChannels(), samplesPerBlock);
        scBandBuf[i].setSize (getTotalNumInputChannels(), samplesPerBlock);
    }

    // Env init
    auto fetch = [&] (int idx, const char* atID, const char* reID)
    {
        auto a = parameters.getRawParameterValue (atID)->load();
        auto r = parameters.getRawParameterValue (reID)->load();
        bands[idx].envL.prepare (sr, a, r);
        bands[idx].envR.prepare (sr, a, r);
    };
    fetch (0, IDs::at1, IDs::re1);
    fetch (1, IDs::at2, IDs::re2);
    fetch (2, IDs::at3, IDs::re3);
}

static inline float dbToLin (float db) { return juce::Decibels::decibelsToGain (db); }

void MultibandSCAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ignoreUnused (midi);
    juce::ScopedNoDenormals noDenormals;

    const bool isBypassed = parameters.getRawParameterValue (IDs::bypass)->load() > 0.5f;
    if (isBypassed) return;

    const float xLowHz  = parameters.getRawParameterValue (IDs::xLow)->load();
    const float xHighHz = parameters.getRawParameterValue (IDs::xHigh)->load();

    // zaktualizuj LR cutoffy (JUCE automatycznie zrobi smoothing)
    *xLow.state  = juce::dsp::LinkwitzRileyFilter<float>::Coefficients::makeLowPass  (sr, xLowHz);
    *xHigh.state = juce::dsp::LinkwitzRileyFilter<float>::Coefficients::makeHighPass (sr, xHighHz);

    // Wejście sidechain
    const bool externalSC = parameters.getRawParameterValue (IDs::scMode)->load() > 0.5f;
    auto* scBus = getBusBuffer (buffer, true, 1); // aux input index 1
    const bool scConnected = externalSC && (getBus(true, 1) != nullptr) && (getBus(true, 1)->isEnabled()) && (scBus.getNumChannels() > 0);

    // Skopiuj main do pasm (band splitting LR24: low, mid, high)
    const int numCh = buffer.getNumChannels();
    const int n = buffer.getNumSamples();

    // robocze kopie
    for (int i=0; i<3; ++i) {
        bandBuf[i].setSize (numCh, n, false, false, true);
        bandBuf[i].clear();
        scBandBuf[i].setSize (numCh, n, false, false, true);
        scBandBuf[i].clear();
    }

    // ---- Split MAIN na 3 pasma ----
    // Low = LP(xLow), High = HP(xHigh), Mid = reszta
    // Przetwarzamy per kanał
    for (int ch=0; ch<numCh; ++ch)
    {
        auto* src = buffer.getReadPointer (ch);
        auto* low = bandBuf[0].getWritePointer (ch);
        auto* mid = bandBuf[1].getWritePointer (ch);
        auto* hig = bandBuf[2].getWritePointer (ch);

        // Kopie robocze
        juce::HeapBlock<float> tmp1(n), tmp2(n);
        std::memcpy (tmp1.getData(), src, sizeof(float)*n);
        std::memcpy (tmp2.getData(), src, sizeof(float)*n);

        // Low
        juce::dsp::AudioBlock<float> bl (&tmp1[0], 1, (size_t) n);
        xLow.process (juce::dsp::ProcessContextReplacing<float> (bl));
        std::memcpy (low, tmp1.getData(), sizeof(float)*n);

        // High
        juce::dsp::AudioBlock<float> bh (&tmp2[0], 1, (size_t) n);
        xHigh.process (juce::dsp::ProcessContextReplacing<float> (bh));
        std::memcpy (hig, tmp2.getData(), sizeof(float)*n);

        // Mid = src - low - high
        for (int i=0; i<n; ++i) mid[i] = src[i] - low[i] - hig[i];
    }

    // ---- Przygotuj sygnał SIDECHAIN dla detektora (per pasmo) ----
    // Jeśli brak zewnętrznego SC, używamy main (self-key)
    for (int ch=0; ch<numCh; ++ch)
    {
        const float* scIn = nullptr;
        if (scConnected) scIn = scBus.getReadPointer (juce::jmin (ch, scBus.getNumChannels()-1));
        else             scIn = buffer.getReadPointer (ch);

        auto* scl = scBandBuf[0].getWritePointer (ch);
        auto* scm = scBandBuf[1].getWritePointer (ch);
        auto* sch = scBandBuf[2].getWritePointer (ch);

        // Użyj tych samych filtrów LR co dla main (kopie współczynników)
        juce::HeapBlock<float> t1(n), t2(n);
        std::memcpy (t1.getData(), scIn, sizeof(float)*n);
        std::memcpy (t2.getData(), scIn, sizeof(float)*n);

        // Low
        juce::dsp::AudioBlock<float> bl (&t1[0], 1, (size_t) n);
        auto lowCopy = xLow;  // kopia stanu
        lowCopy.process (juce::dsp::ProcessContextReplacing<float> (bl));
        std::memcpy (scl, t1.getData(), sizeof(float)*n);

        // High
        juce::dsp::AudioBlock<float> bh (&t2[0], 1, (size_t) n);
        auto highCopy = xHigh;
        highCopy.process (juce::dsp::ProcessContextReplacing<float> (bh));
        std::memcpy (sch, t2.getData(), sizeof(float)*n);

        // Mid
        for (int i=0; i<n; ++i) scm[i] = scIn[i] - scl[i] - sch[i];
    }

    // Zaktualizuj cache parametrów pasm (th, ratio, at, re, mk) oraz env times
    auto updBand = [&] (Band& b, const char* th, const char* ra, const char* at, const char* re, const char* mk)
    {
        b.thresh = parameters.getRawParameterValue (th)->load();
        b.ratio  = juce::jmax (1.001f, parameters.getRawParameterValue (ra)->load());
        b.attackMs  = parameters.getRawParameterValue (at)->load();
        b.releaseMs = parameters.getRawParameterValue (re)->load();
        b.makeupDb  = parameters.getRawParameterValue (mk)->load();
        b.envL.setTimes (b.attackMs, b.releaseMs);
        b.envR.setTimes (b.attackMs, b.releaseMs);
    };
    updBand (bands[0], IDs::th1, IDs::ra1, IDs::at1, IDs::re1, IDs::mk1);
    updBand (bands[1], IDs::th2, IDs::ra2, IDs::at2, IDs::re2, IDs::mk2);
    updBand (bands[2], IDs::th3, IDs::ra3, IDs::at3, IDs::re3, IDs::mk3);

    // ---- Kompresja per pasmo, z detekcją z odpow. pasma sidechain ----
    for (int bandIdx=0; bandIdx<3; ++bandIdx)
    {
        auto& b = bands[bandIdx];
        const float makeupLin = dbToLin (b.makeupDb);

        for (int ch=0; ch<numCh; ++ch)
        {
            auto* x = bandBuf[bandIdx].getWritePointer (ch);
            const auto* sc = scBandBuf[bandIdx].getReadPointer (juce::jmin(ch, scBandBuf[bandIdx].getNumChannels()-1));

            for (int i=0; i<n; ++i)
            {
                // detektor szczytowy (z at/re), następnie w dB
                float env = (ch == 0 ? b.envL.process (sc[i]) : b.envR.process (sc[i]));
                float envDb = juce::Decibels::gainToDecibels (env + 1e-8f);

                float over = envDb - b.thresh; // dB ponad próg
                float grDb = (over > 0.f) ? (over - over / b.ratio) : 0.f; // redukcja dB
                float grLin = dbToLin (-grDb);

                x[i] *= (grLin * makeupLin);
            }
        }
    }

    // Zsumuj pasma z powrotem do buffer
    for (int ch=0; ch<numCh; ++ch)
    {
        auto* dst = buffer.getWritePointer (ch);
        const float* l = bandBuf[0].getReadPointer (ch);
        const float* m = bandBuf[1].getReadPointer (ch);
        const float* h = bandBuf[2].getReadPointer (ch);
        for (int i=0; i<n; ++i)
            dst[i] = l[i] + m[i] + h[i];
    }
}

juce::AudioProcessorEditor* MultibandSCAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor (*this); // proste GUI automatyczne
}

void MultibandSCAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::MemoryOutputStream mos (destData, false);
    parameters.state.writeToStream (mos);
}
void MultibandSCAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto tree = juce::ValueTree::readFromData (data, sizeInBytes))
        parameters.replaceState (tree);
}

juce::AudioProcessorValueTreeState::ParameterLayout MultibandSCAudioProcessor::createLayout()
{
    using namespace juce;
    std::vector<std::unique_ptr<RangedAudioParameter>> p;

    auto floatParam = [] (const String& id, const String& nm, float min, float max, float def,
                          float skew = 1.0f, const String& su = {}) {
        NormalisableRange<float> r (min, max, 0.0f, skew);
        return std::make_unique<AudioParameterFloat> (ParameterID { id, 1 }, nm, r, def, su);
    };

    p.push_back (std::make_unique<AudioParameterBool> (ParameterID{IDs::bypass,1}, "Bypass", false));
    p.push_back (std::make_unique<AudioParameterChoice> (ParameterID{IDs::scMode,1}, "Sidechain",
                    StringArray { "Self", "External" }, 1)); // domyślnie External

    // crossovery
    p.push_back (floatParam (IDs::xLow,  "Xover Low (Hz)",  60.0f,  400.0f, 120.0f, 0.5f));
    p.push_back (floatParam (IDs::xHigh, "Xover High (Hz)", 1000.f, 8000.f, 3000.f, 0.5f));

    // Low band
    p.push_back (floatParam (IDs::th1, "Low Thresh (dB)", -60.f, 0.f, -24.f));
    p.push_back (floatParam (IDs::ra1, "Low Ratio", 1.0f, 20.f, 4.f));
    p.push_back (floatParam (IDs::at1, "Low Attack (ms)",  0.1f, 200.f, 10.f, 0.5f));
    p.push_back (floatParam (IDs::re1, "Low Release (ms)", 5.f,  1000.f, 120.f, 0.5f));
    p.push_back (floatParam (IDs::mk1, "Low Makeup (dB)", -24.f, 24.f, 0.f));

    // Mid band
    p.push_back (floatParam (IDs::th2, "Mid Thresh (dB)", -60.f, 0.f, -24.f));
    p.push_back (floatParam (IDs::ra2, "Mid Ratio", 1.0f, 20.f, 4.f));
    p.push_back (floatParam (IDs::at2, "Mid Attack (ms)",  0.1f, 200.f, 10.f, 0.5f));
    p.push_back (floatParam (IDs::re2, "Mid Release (ms)", 5.f,  1000.f, 120.f, 0.5f));
    p.push_back (floatParam (IDs::mk2, "Mid Makeup (dB)", -24.f, 24.f, 0.f));

    // High band
    p.push_back (floatParam (IDs::th3, "High Thresh (dB)", -60.f, 0.f, -24.f));
    p.push_back (floatParam (IDs::ra3, "High Ratio", 1.0f, 20.f, 4.f));
    p.push_back (floatParam (IDs::at3, "High Attack (ms)",  0.1f, 200.f, 6.f, 0.5f));
    p.push_back (floatParam (IDs::re3, "High Release (ms)", 5.f,  1000.f, 80.f, 0.5f));
    p.push_back (floatParam (IDs::mk3, "High Makeup (dB)", -24.f, 24.f, 0.f));

    return { p.begin(), p.end() };
}
