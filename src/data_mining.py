import os
import sys
import glob
import pandas as pd
import numpy as np

def extract_5packet_flows(pcap_csv_file, label, window_size=5, stride=1):
 df_raw = pd.read_csv(pcap_csv_file)
 if "Length" in df_raw.columns and "Time" in df_raw.columns:
 lengths = df_raw["Length"].values
 timestamps = df_raw["Time"].values
 elif "packet_length" in df_raw.columns and "timestamp" in df_raw.columns:
 lengths = df_raw["packet_length"].values
 timestamps = df_raw["timestamp"].values
 else:
 cols = df_raw.columns
 lengths = df_raw[cols[0]].values
 timestamps = df_raw[cols[1]].values

 valid_mask = (lengths >= 28) & (lengths <= 1500)
 lengths = lengths[valid_mask]
 timestamps = timestamps[valid_mask]

 n_packets = len(lengths)
 records = []

 for start in range(0, n_packets - window_size + 1, stride):
 win_lens = lengths[start:start + window_size]
 win_times = timestamps[start:start + window_size]
 t0 = win_times[0]
 rel_times = win_times - t0

 if not np.all(np.diff(rel_times) > 0):
 continue

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

def build_master_dataset(input_dir, output_path="data/Augmented_Training_set.csv", target_per_class=520):
 all_dfs = []
 classes = [
 "Discord_video", "Discord_voice",
 "GoogleMeet_video", "GoogleMeet_voice",
 "Messenger_video", "Messenger_voice",
 "WhatsApp_video", "WhatsApp_voice",
 "Zoom_video", "Zoom_voice"
 ]

 for c in classes:
 pattern = os.path.join(input_dir, f"{c}*.csv")
 files = glob.glob(pattern)
 class_records = []
 for f in files:
 df_sub = extract_5packet_flows(f, label=c)
 class_records.append(df_sub)

 if len(class_records) > 0:
 df_class = pd.concat(class_records, ignore_index=True)
 if len(df_class) > target_per_class:
 df_class = df_class.sample(n=target_per_class, random_state=42)
 all_dfs.append(df_class)

 if len(all_dfs) > 0:
 master_df = pd.concat(all_dfs, ignore_index=True)
 master_df = master_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
 master_df.to_csv(output_path, index=False)
 print(f"Master dataset created with {len(master_df)} flows saved to {output_path}")
 return master_df
 else:
 print(f"Existing dataset verified at {output_path}")
 return pd.read_csv(output_path)

if __name__ == "__main__":
 in_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw_captures"
 out_file = sys.argv[2] if len(sys.argv) > 2 else "data/Augmented_Training_set.csv"
 if os.path.exists(in_dir):
 build_master_dataset(in_dir, out_file)
 else:
 print(f"Directory {in_dir} not found. Existing dataset in {out_file} contains {len(pd.read_csv(out_file))} flows.")
