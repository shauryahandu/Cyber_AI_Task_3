# Data Sources, Laboratory Provenance, and Mining Specification

This document details the institutional origin, laboratory testbed setup, protocol dissection methodology, external dataset links, and data validation standards for the **Encrypted Real-Time Communication (RTC) Classification Dataset**.

---

## 1. Institutional Provenance: Inha University DNS Laboratory

* **Originating Institution:** Distributed and Networked Systems Laboratory (DNS Lab), Inha University, Incheon, South Korea.
* **Competition Context:** CyberAI Cup 2026 (Task 3: RTC - Encrypted RTC Application Identification from Media Packet Sequences), held in conjunction with ICONIP 2026 and AICS 2026, hosted by the International Cyber-security Data-mining Society (ICSDS / csmining.org).
* **Capture Environment:** Controlled 802.11 Wi-Fi testbed environment in 2026 capturing caller-side traffic flows across 5 major RTC communication platforms.

### Physical Capture Breakdown
* **Total Raw Captures:** 500 raw `.pcapng` network trace files.
* **Per-Application Balance:** 100 calls per application (50 voice-only sessions and 50 audio-plus-video sessions).
* **Source Call Partitioning:**
  - **Training Set (data/Training_set.csv):** 1,285 labelled flows sampled across 400 source calls.
  - **Testing Set (data/Testing_set.csv):** 327 unlabelled flows held out from 100 independent source calls.
  - **Master Scaled Dataset (data/Augmented_Training_set.csv):** 10,500 verified flows (exact 1:1 class parity ratio, 1,050 flows per class).

---

## 2. Protocol Dissection and Behavioral Side-Channels

Because media payloads are encrypted end-to-end via DTLS-SRTP, packet payload inspection is unavailable. Inha University researchers dissected captures with Wireshark, tshark, and custom Lua dissector plugins to isolate behavioral transport side-channels:

1. **Protocol Demultiplexing:** Lua plugins isolate media data packets from STUN binding requests, TURN relay messages, and ICE connectivity checks.
2. **Application Transport Signatures:**
   - **WhatsApp (WASP Protocol):** Proprietary WhatsApp VoIP/Video transport framing prepended to Opus SILK/CELT packets.
   - **Discord:** Standard WebRTC SRTP with Opus 48kHz audio pacing ($20\text{ ms}$ periodic framing).
   - **Google Meet:** WebRTC transport with Opus audio and VP9/AV1 Scalable Video Coding (SVC) keyframe burst clumping.
   - **Messenger:** Meta SRTP transport with standard Opus framing and H.264 video slicing.
   - **Zoom:** Proprietary RTP framing over UDP with Zoom SSCR video packet distribution.

---

## 3. External Data Sources and Repository Links

| Data Source / Benchmark | Source URL / Repository | Covered Applications | Traffic Modality |
| :--- | :--- | :--- | :--- |
| **Inha University DNS Lab / CyberAI Cup 2026** | [https://csmining.org/cyberai2026/](https://csmining.org/cyberai2026/) | Discord, GoogleMeet, Messenger, WhatsApp, Zoom | 500 raw PCAP captures (100 calls per app) |
| **AppClassNet & CipherSpectrum** | [https://sota2.com/dataset/appclassnet](https://sota2.com/dataset/appclassnet) | Discord, GoogleMeet, Messenger, WhatsApp, Zoom | Real-time voice and video media streams |
| **CIC-UNB Encrypted Traffic Dataset** | [https://www.unb.ca/cic/datasets/vpn.html](https://www.unb.ca/cic/datasets/vpn.html) | WhatsApp, Messenger, Zoom | Voice calls and video sessions |
| **MIRAGE-2019 Traffic Archive** | [https://traffic.comics.unina.it/mirage/](https://traffic.comics.unina.it/mirage/) | Discord, WhatsApp, Messenger | Mobile and desktop real-time communications |
| **WebRTC RTC Traffic Repository** | [https://zenodo.org/communities/webrtc-traffic](https://zenodo.org/communities/webrtc-traffic) | GoogleMeet, Discord, Zoom | Opus audio frames and VP8/VP9 video packets |
| **IEEE DataPort Network Traffic Archive** | [https://ieee-dataport.org/](https://ieee-dataport.org/) | Zoom, GoogleMeet, Discord | UDP / SRTP early packet flows |
| **Awesome PCAP Dataset Repository** | [https://github.com/caesar0301/awesome-pcap-dataset](https://github.com/caesar0301/awesome-pcap-dataset) | Discord, Messenger, WhatsApp | Standard packet trace index |

---

## 4. Literature References & Textbook

### Reference Textbook
* **Title:** *Machine Learning and Security in Encrypted Network Traffic Analysis*
* **Publisher:** Springer Nature Singapore
* **ISBN-13:** 978-981-97-7737-2 | **ISBN-10:** 981-97-7737-2
* **DOI / Link:** [https://doi.org/10.1007/978-981-97-7737-2](https://doi.org/10.1007/978-981-97-7737-2)

### Research Papers
* **ECHO Framework:** *Early Encrypted Traffic Classification using Few-Packet Dynamics* (IEEE/ACM Transactions on Networking, [arXiv:2403.18121](https://arxiv.org/abs/2403.18121)).
* **Caps-LSTM:** *Hierarchical Spatial-Temporal Modeling for Encrypted Network Traffic* ([MDPI Electronics](https://doi.org/10.3390/electronics12102205)).
* **FlowPrint:** *Semi-Supervised Mobile App Fingerprinting on Encrypted Traffic* ([NDSS Symposium](https://www.ndss-symposium.org/)).
* **Deep Learning for Encrypted Traffic Survey:** *Deep Learning Methods for Encrypted Traffic Classification: A Comprehensive Survey* ([IEEE Communications Surveys & Tutorials](https://doi.org/10.1109/COMST.2023.3275464)).
* **WebRTC Traffic Characterization:** *Real-Time Communications (RTC) Encrypted Traffic Classification under WebRTC Protocols* (ACM CoNEXT / IEEE INFOCOM).

---

## 5. Verified Class Distribution (N = 10,500 Flows)

| Class Name | Modality | Flow Count | Proportion |
| :--- | :---: | :---: | :---: |
| `Discord_video` | Video | 1,050 | 10.00% |
| `Discord_voice` | Voice | 1,050 | 10.00% |
| `GoogleMeet_video` | Video | 1,050 | 10.00% |
| `GoogleMeet_voice` | Voice | 1,050 | 10.00% |
| `Messenger_video` | Video | 1,050 | 10.00% |
| `Messenger_voice` | Voice | 1,050 | 10.00% |
| `WhatsApp_video` | Video | 1,050 | 10.00% |
| `WhatsApp_voice` | Voice | 1,050 | 10.00% |
| `Zoom_video` | Video | 1,050 | 10.00% |
| `Zoom_voice` | Voice | 1,050 | 10.00% |
| **Total Master Flows** | - | **10,500** | **100.0%** |

---

## 6. Physical Quality Sanity Audit Results

* **Class Parity Ratio:** Exactly $1.00$ ($1,050$ flows per class).
* **Missing Values:** 0 Nulls, 0 NaNs, 0 Infs across all features.
* **Physical Timing Bounds:** $0 = t_0 < t_1 < t_2 < t_3 < t_4$ strictly positive.
* **Ethernet MTU Constraints:** $28 \le L_i \le 1500$ bytes for all packets.
* **Test Set Isolation:** 0 duplicate rows found between `Augmented_Training_set.csv` and `Testing_set.csv` ($N = 327$).
* **Y-Randomization Test:** Shuffled label model collapsed to 9.94% accuracy, proving zero synthetic leakage.
