cmake_minimum_required(VERSION 3.15)
project(MultibandSC VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(JUCE)

juce_add_plugin(MultibandSC
    COMPANY_NAME "YourName"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE Juce
    PLUGIN_CODE Mscc
    FORMATS VST3 AU
    PRODUCT_NAME "Multiband Sidechain Compressor")

target_sources(MultibandSC PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h)

target_link_libraries(MultibandSC PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp)

juce_generate_juce_header(MultibandSC)
