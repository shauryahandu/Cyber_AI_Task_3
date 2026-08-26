import os
import sys
import glob
import wave
import numpy as np
import pandas as pd

def wav_to_packet_flow(wav_path, label, frame_duration_ms=20, window_size=5, stride=1):
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    bytes_per_sample = n_channels * sampwidth
    samples_per_packet = int(framerate * (frame_duration_ms / 1000.0))
    bytes_per_packet = samples_per_packet * bytes_per_sample

    total_packets = len(audio_bytes) // bytes_per_packet
    if total_packets < window_size:
        return pd.DataFrame()

    packet_lengths = []
    timestamps = []

    np.random.seed(42)
    current_time = 0.0

    for i in range(total_packets):
        raw_chunk = audio_bytes[i * bytes_per_packet:(i + 1) * bytes_per_packet]
        chunk_arr = np.frombuffer(raw_chunk, dtype=np.int16)
        rms = np.sqrt(np.mean(chunk_arr.astype(np.float32)**2) + 1e-9)

        if rms < 100:
            compressed_len = int(np.random.choice([70, 78, 85, 92]))
        else:
            base_len = int(np.clip(rms / 120.0 * 80 + 120, 110, 320))
            jitter_len = int(np.random.normal(0, 8))
            compressed_len = int(np.clip(base_len + jitter_len, 95, 350))

        packet_lengths.append(compressed_len)
        timestamps.append(current_time)

        jitter_time = np.random.normal(0, 0.0008)
        delta_t = max(0.005, (frame_duration_ms / 1000.0) + jitter_time)
        current_time += delta_t

    records = []
    for start in range(0, len(packet_lengths) - window_size + 1, stride):
        win_lens = packet_lengths[start:start + window_size]
        win_times = timestamps[start:start + window_size]
        t0 = win_times[0]
        rel_times = [float(t - t0) for t in win_times]

        row = {
            "packet_length_0": int(win_lens[0]),
            "packet_length_1": int(win_lens[1]),
            "packet_length_2": int(win_lens[2]),
            "packet_length_3": int(win_lens[3]),
            "packet_length_4": int(win_lens[4]),
            "relative_time_0": float(rel_times[0]),
            "relative_time_1": float(rel_times[1]),
            "relative_time_2": float(rel_times[2]),
            "relative_time_3": float(rel_times[3]),
            "relative_time_4": float(rel_times[4]),
            "label": label
        }
        records.append(row)

    return pd.DataFrame(records)

def convert_audio_directory(input_dir, output_csv="data/audio_extracted_flows.csv"):
    files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    all_dfs = []

    for f in files:
        fname = os.path.basename(f).lower()
        if "discord" in fname:
            lbl = "Discord_voice"
        elif "meet" in fname or "google" in fname:
            lbl = "GoogleMeet_voice"
        elif "messenger" in fname:
            lbl = "Messenger_voice"
        elif "whatsapp" in fname:
            lbl = "WhatsApp_voice"
        elif "zoom" in fname:
            lbl = "Zoom_voice"
        else:
            lbl = "Discord_voice"

        df_flow = wav_to_packet_flow(f, label=lbl)
        if not df_flow.empty:
            all_dfs.append(df_flow)

    if all_dfs:
        out_df = pd.concat(all_dfs, ignore_index=True)
        out_df.to_csv(output_csv, index=False)
        print(f"Extracted {len(out_df)} flows from audio files into {output_csv}")
        return out_df
    else:
        print(f"No audio files processed from {input_dir}")
        return pd.DataFrame()

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/audio_raw"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/audio_extracted_flows.csv"
    if os.path.exists(in_path):
        convert_audio_directory(in_path, out_path)
    else:
        print(f"Input path {in_path} does not exist.")
