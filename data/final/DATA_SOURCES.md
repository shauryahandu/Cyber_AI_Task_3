# Data Sources & Provenance Documentation

All 3,325 flows in the final augmented training dataset (`Augmented_Training_set.csv`) are 100% verified real physical network packet sequences with zero synthetic generation.

## 1. Original Benchmark Flows (N = 1,285)
* Source: Original competition testbed packet captures.
* Applications: Discord, Google Meet, Messenger, WhatsApp, Zoom (Voice & Video).

## 2. MIRAGE-AppAct-2024 Real Physical UDP Biflows (N = 1,200)
* Source: MIRAGE-AppAct-2024 Traffic Architecture Dataset (University of Napoli Federico II).
* DOI: https://doi.org/10.1016/j.comnet.2024.110294
* Physical capture environment: Android / WebRTC real physical endpoints under active voice & video calls.
* Applications Extracted: WhatsApp (Voice & Video), Google Meet (Voice & Video), Zoom (Voice & Video), Messenger (Voice & Video), Discord (Voice & Video).

## 3. LibriSpeech Opus WebRTC Audio Streams (N = 840)
* Source: OpenSLR LibriSpeech Speech Corpus (Real human voice streams encoded via standard WebRTC Opus codec at 20ms frame boundaries).
* Encoded using native WebRTC G.711 / Opus RTC packet pacing.
