import os
import glob
import json
import numpy as np
import pandas as pd
import soundfile as sf
mirage_dir = "./data/MIRAGE-AppAct-2024"
app_targets = {
    'Discord': {'audiocall': 'Discord_voice', 'videocall': 'Discord_video'},
    'Messenger': {'audiocall': 'Messenger_voice', 'videocall': 'Messenger_video'}
}
extracted_rows = []
for app, activity_map in app_targets.items():
    app_folder = os.path.join(mirage_dir, app)
    if not os.path.exists(app_folder):
        continue
    json_files = glob.glob(os.path.join(app_folder, "*.json"))
    app_counts = {v: 0 for v in activity_map.values()}
    max_per_class = 200
    for jf in json_files:
        if all(c >= max_per_class for c in app_counts.values()):
            break
        try:
            with open(jf) as fp:
                flow_data = json.load(fp)
            for k, flow in flow_data.items():
                prot = k.split(',')[-1]
                if prot != '17':
                    continue
                meta = flow.get('flow_metadata', {})
                act = str(meta.get('BF_activity', ''))
                target_class = None
                for act_key, cls_name in activity_map.items():
                    if act_key in act and app_counts[cls_name] < max_per_class:
                        target_class = cls_name
                        break
                if not target_class:
                    continue
                pdata = flow.get('packet_data', {})
                ts = pdata.get('timestamp', [])
                lens = pdata.get('L4_payload_bytes', [])
                if len(ts) < 5 or len(lens) < 5:
                    continue
                ts = np.array(ts)
                lens = np.array(lens)
                valid_mask = (lens > 0)
                ts = ts[valid_mask]
                lens = lens[valid_mask]
                if len(ts) < 5:
                    continue
                for i in range(0, len(ts) - 5, 3):
                    if app_counts[target_class] >= max_per_class:
                        break
                    t_win = ts[i:i+5] - ts[i]
                    l_win = lens[i:i+5]
                    if np.all(np.diff(t_win) >= 0) and np.all(l_win <= 1500) and np.all(l_win >= 20):
                        extracted_rows.append({
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
                            'label': target_class
                        })
                        app_counts[target_class] += 1
        except Exception:
            pass
flac_files = glob.glob("./data/dev-clean/LibriSpeech/dev-clean/*/*/*.flac")
rng = np.random.RandomState(42)
voice_configs = {
    'GoogleMeet_voice': {'target_count': 260, 'base_len': 110, 'len_std': 6, 'jitter_scale': 0.0003},
    'Zoom_voice': {'target_count': 220, 'base_len': 105, 'len_std': 8, 'jitter_scale': 0.0004},
    'WhatsApp_voice': {'target_count': 220, 'base_len': 120, 'len_std': 45, 'jitter_scale': 0.0006},
}
for cls_name, cfg in voice_configs.items():
    needed = cfg['target_count']
    created = 0
    for flac_p in flac_files:
        if created >= needed:
            break
        try:
            audio, sr = sf.read(flac_p)
            frame_len = int(sr * 0.020)
            n_frames = len(audio) // frame_len
            if n_frames < 5:
                continue
            for start_f in range(0, n_frames - 5, 2):
                if created >= needed:
                    break
                frame_energies = []
                for f_idx in range(start_f, start_f + 5):
                    frame = audio[f_idx * frame_len : (f_idx + 1) * frame_len]
                    rms = np.sqrt(np.mean(frame**2) + 1e-9)
                    frame_energies.append(rms)
                norm_energy = np.array(frame_energies) / (np.max(frame_energies) + 1e-6)
                l_win = cfg['base_len'] + (norm_energy * cfg['len_std']).astype(int) + rng.randint(-3, 4, size=5)
                l_win = np.clip(l_win, 40, 350)
                dt_base = 0.020
                jitters = rng.normal(0, cfg['jitter_scale'], size=4)
                t_win = [0.0]
                curr_t = 0.0
                for j in range(4):
                    curr_t += max(0.0001, dt_base + jitters[j])
                    t_win.append(curr_t)
                extracted_rows.append({
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
                    'label': cls_name
                })
                created += 1
        except Exception:
            pass
video_configs = {
    'WhatsApp_video': {'target_count': 220, 'base_len': 1000, 'len_std': 12, 'sub_ms': True},
    'Zoom_video': {'target_count': 130, 'base_len': 1050, 'len_std': 180, 'sub_ms': True},
    'GoogleMeet_video': {'target_count': 190, 'base_len': 1180, 'len_std': 150, 'sub_ms': True}
}
for cls_name, cfg in video_configs.items():
    for _ in range(cfg['target_count']):
        t_offsets = np.sort(rng.uniform(0.00005, 0.0006, size=4))
        l_win = np.clip(rng.normal(cfg['base_len'], cfg['len_std'], size=5).astype(int), 100, 1450)
        if rng.rand() < 0.4:
            l_win[rng.choice(5, size=2, replace=False)] = rng.randint(900, 1300, size=2)
        extracted_rows.append({
            'relative_time_0': 0.0,
            'packet_length_0': int(l_win[0]),
            'relative_time_1': float(t_offsets[0]),
            'packet_length_1': int(l_win[1]),
            'relative_time_2': float(t_offsets[1]),
            'packet_length_2': int(l_win[2]),
            'relative_time_3': float(t_offsets[2]),
            'packet_length_3': int(l_win[3]),
            'relative_time_4': float(t_offsets[3]),
            'packet_length_4': int(l_win[4]),
            'label': cls_name
        })
df_base = pd.read_csv("data/Training_set.csv")
df_new = pd.DataFrame(extracted_rows)
df_augmented = pd.concat([df_base, df_new], ignore_index=True)
out_csv = "data/Augmented_Training_set.csv"
out_xlsx = "data/Augmented_Training_set.xlsx"
df_augmented.to_csv(out_csv, index=False)
df_augmented.to_excel(out_xlsx, index=False)
summary = df_augmented['label'].value_counts().reset_index()
summary.columns = ['Traffic_Class', 'Sample_Count']
summary['Percentage'] = (summary['Sample_Count'] / len(df_augmented) * 100).round(2).astype(str) + '%'
