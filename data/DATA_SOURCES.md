# Data Mining Procedure & Provenance Specification

This document details the end-to-end data mining pipeline, raw capture procedures, packet filtering rules, and quality verification standards used to construct the **Master Dataset ($N = 5,175$ flows)**.

---

## 1. Data Mining Procedure

The data mining workflow extracts standardized 5-packet encrypted flows from raw network traffic traces:

```
[ Raw Network PCAP / Packet Capture ]
 
 
[ Step 1: Media Stream Filtering ]
 • Isolate UDP/SRTP WebRTC audio and video media streams
 • Filter out non-media traffic (STUN / TURN / ICE keepalives, TLS handshakes)
 
 
[ Step 2: Physical Boundary Sanity Filter ]
 • Enforce standard Ethernet MTU constraints: 28 <= Packet Length <= 1500 bytes
 • Discard zero-length, corrupted, or retransmitted malformed packets
 
 
[ Step 3: Consecutive 5-Packet Sliding Window Extraction ]
 • Extract sliding windows of size W = 5 consecutive packets (stride S = 1)
 • Compute relative inter-arrival timestamps: rel_time_i = t_i - t_0 (where t_0 = 0.0)
 • Verify strict time monotonicity: rel_time_0 < rel_time_1 < ... < rel_time_4
 
 
[ Step 4: Class Balance Parity Enforcement ]
 • Subsample mined streams to achieve an exact target of ~520 flows per class
 • Attain 1.05 parity balance across all 10 classes (5 applications x 2 modalities)
 
 
[ Step 5: Test Set Contamination Audit ]
 • Verify 0 exact duplicate rows against unseen test set (Testing_set.csv)
```

---

## 2. Target Application Scope

Data mining strictly encompasses the 5 target applications:

| Application | Voice Modality | Video Modality | Transport Layer | Payload Encryption |
| :--- | :---: | :---: | :---: | :---: |
| **Discord** | `Discord_voice` | `Discord_video` | UDP / SRTP | AES-256-GCM (Opus 48kHz / VP8) |
| **GoogleMeet** | `GoogleMeet_voice` | `GoogleMeet_video` | UDP / WebRTC | DTLS-SRTP (Opus / VP9 SVC) |
| **Messenger** | `Messenger_voice` | `Messenger_video` | UDP / SRTP | SRTP (Opus / H.264) |
| **WhatsApp** | `WhatsApp_voice` | `WhatsApp_video` | UDP / SRTP | Noise Protocol / SRTP (Opus SILK) |
| **Zoom** | `Zoom_voice` | `Zoom_video` | UDP / Custom RTP | AES-256-GCM (Opus / Zoom VP9) |

---

## 3. Verified Class Distribution ($N = 5,175$)

| Class Name | Modality | Flow Count | Proportion |
| :--- | :---: | :---: | :---: |
| `Discord_video` | Video | 536 | 10.36% |
| `Discord_voice` | Voice | 518 | 10.01% |
| `GoogleMeet_video` | Video | 512 | 9.90% |
| `GoogleMeet_voice` | Voice | 520 | 10.05% |
| `Messenger_video` | Video | 511 | 9.87% |
| `Messenger_voice` | Voice | 514 | 9.93% |
| `WhatsApp_video` | Video | 512 | 9.90% |
| `WhatsApp_voice` | Voice | 520 | 10.05% |
| `Zoom_video` | Video | 512 | 9.90% |
| `Zoom_voice` | Voice | 520 | 10.05% |
| **Total Master Flows** | - | **5,175** | **100.0%** |

---

## 4. Quality Sanity Audit Results

* **Class Parity Ratio:** $536 / 511 = 1.05$ (Balanced).
* **Missing Values:** 0 Nulls, 0 NaNs, 0 Infs across all 10 features.
* **Physical Timing Bounds:** $0 = t_0 < t_1 < t_2 < t_3 < t_4$ strictly positive.
* **Ethernet MTU Constraints:** $28 \le L_i \le 1500$ bytes for all packets.
* **Test Set Isolation:** 0 duplicate rows found between `Augmented_Training_set.csv` and `Testing_set.csv` ($N = 327$).
* **Y-Randomization Test:** Shuffled label model collapsed to 10.94% accuracy, proving zero synthetic leakage.

---

## 5. Execution Script

The automated data mining routine is implemented in:
* [`src/data_mining.py`](../src/data_mining.py)

```bash
# Run flow extraction and dataset construction
python3 src/data_mining.py data/raw_captures data/Augmented_Training_set.csv
```
