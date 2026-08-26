import os
import glob
import pandas as pd
import numpy as np
from scapy.all import rdpcap, UDP, IP, IPv6
RAW_DIR = "data/external_raw"
OUT_CSV = "data/Augmented_Training_set.csv"
VALID_CLASSES = [
    'Discord_video', 'Discord_voice',
    'GoogleMeet_video', 'GoogleMeet_voice',
    'Messenger_video', 'Messenger_voice',
    'WhatsApp_video', 'WhatsApp_voice',
    'Zoom_video', 'Zoom_voice'
]
def map_filename_to_class(filename):
    lower = filename.lower()
    for c in VALID_CLASSES:
        app, mod = c.split('_')
        if app.lower() in lower and mod.lower() in lower:
            return c
    return None
def parse_pcap(filepath, class_label, stride=2):
    print(f"  Parsing PCAP: {filepath} -> Target Class: {class_label}")
    packets = rdpcap(filepath)
    timestamps = []
    payload_lengths = []
    for pkt in packets:
        if UDP in pkt:
            payload_len = len(pkt[UDP].payload)
            if payload_len > 0:
                timestamps.append(float(pkt.time))
                payload_lengths.append(payload_len)
    if len(timestamps) < 5:
        print(f"  ⚠️ Warning: Only {len(timestamps)} UDP packets found in {filepath}. Need >= 5.")
        return pd.DataFrame()
    timestamps = np.array(timestamps)
    payload_lengths = np.array(payload_lengths)
    rows = []
    for i in range(0, len(timestamps) - 5, stride):
        t_win = timestamps[i:i+5] - timestamps[i]
        l_win = payload_lengths[i:i+5]
        row = {
            'relative_time_0': 0.0,
            'packet_length_0': int(l_win[0]),
            'relative_time_1': float(t_win[1]),
            'packet_length_1': int(l_win[1]),
            'relative_time_2': float(t_win[2]),
            'packet_length_2': int(l_win[2]),
            'relative_time_3': float(t_win[3]),
            'packet_length_3': int(l_win[3]),
            'relative_time_4': float(t_win[4]),
            'packet_length_4': int(l_win[4]),
            'label': class_label
        }
        rows.append(row)
    df_res = pd.DataFrame(rows)
    print(f"  ✅ Extracted {len(df_res)} 5-packet flows from {os.path.basename(filepath)}")
    return df_res
def parse_raw_csv(filepath, class_label, stride=2):
    print(f"  Parsing CSV: {filepath} -> Target Class: {class_label}")
    df_raw = pd.read_csv(filepath)
    if 'packet_length_0' in df_raw.columns and 'relative_time_1' in df_raw.columns:
        df_raw['label'] = class_label
        return df_raw
    time_col = [c for c in df_raw.columns if 'time' in c.lower() or 'timestamp' in c.lower()][0]
    len_col = [c for c in df_raw.columns if 'len' in c.lower() or 'length' in c.lower() or 'byte' in c.lower()][0]
    timestamps = df_raw[time_col].values
    payload_lengths = df_raw[len_col].values
    rows = []
    for i in range(0, len(timestamps) - 5, stride):
        t_win = timestamps[i:i+5] - timestamps[i]
        l_win = payload_lengths[i:i+5]
        row = {
            'relative_time_0': 0.0,
            'packet_length_0': int(l_win[0]),
            'relative_time_1': float(t_win[1]),
            'packet_length_1': int(l_win[1]),
            'relative_time_2': float(t_win[2]),
            'packet_length_2': int(l_win[2]),
            'relative_time_3': float(t_win[3]),
            'packet_length_3': int(l_win[3]),
            'relative_time_4': float(t_win[4]),
            'packet_length_4': int(l_win[4]),
            'label': class_label
        }
        rows.append(row)
    df_res = pd.DataFrame(rows)
    print(f"  ✅ Extracted {len(df_res)} 5-packet flows from {os.path.basename(filepath)}")
    return df_res
def main():
    print("="*70)
    print("🔍 Scanning data/external_raw/ for New Network Capture Files...")
    print("="*70)
    files = glob.glob(os.path.join(RAW_DIR, "*"))
    if not files:
        print(f"⚠️ No files found in {RAW_DIR}!")
        print(f"Please drop your .pcap, .pcapng, or .csv files into: {os.path.abspath(RAW_DIR)}")
        return
    new_dfs = []
    for f in sorted(files):
        c_label = map_filename_to_class(os.path.basename(f))
        if not c_label:
            print(f"⚠️ Skipping '{os.path.basename(f)}': Unable to infer class name. Name file like 'meet_voice.pcap' or 'zoom_video.csv'.")
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.pcap', '.pcapng']:
            df_parsed = parse_pcap(f, c_label)
            if not df_parsed.empty:
                new_dfs.append(df_parsed)
        elif ext in ['.csv', '.tsv', '.txt']:
            df_parsed = parse_raw_csv(f, c_label)
            if not df_parsed.empty:
                new_dfs.append(df_parsed)
    if not new_dfs:
        print("No valid packet flows extracted.")
        return
    df_new_all = pd.concat(new_dfs, ignore_index=True)
    df_orig = pd.read_csv("data/Training_set.csv")
    df_augmented = pd.concat([df_orig, df_new_all], ignore_index=True)
    df_augmented.to_csv(OUT_CSV, index=False)
    print("\n" + "="*70)
    print(f"🎉 SUCCESS! Merged {len(df_new_all)} external flows with {len(df_orig)} base flows.")
    print(f"📁 Augmented Dataset saved to: {OUT_CSV} (Total: {len(df_augmented)} rows)")
    print("="*70)
    print("\nUpdated Class Distribution:")
    print(df_augmented['label'].value_counts())
if __name__ == '__main__':
    main()
