import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
df_orig = pd.read_csv("data/Training_set.csv")
orig_len_tuples = set([tuple(r) for r in df_orig[['packet_length_0', 'packet_length_1', 'packet_length_2', 'packet_length_3', 'packet_length_4']].values])
mirage_files = glob.glob("./data/MIRAGE-AppAct-2024/*/*.json")
exact_matches = 0
total_mirage_flows_checked = 0
for jf in mirage_files:
    try:
        with open(jf) as fp:
            data = json.load(fp)
        for k, flow in data.items():
            prot = k.split(',')[-1]
            if prot == '17':
                pdata = flow.get('packet_data', {})
                lens = pdata.get('L4_payload_bytes', [])
                if len(lens) >= 5:
                    total_mirage_flows_checked += 1
                    for i in range(len(lens) - 4):
                        win = tuple(lens[i:i+5])
                        if win in orig_len_tuples:
                            exact_matches += 1
    except Exception:
        pass
if exact_matches > 0:
else:
df_aug = pd.read_csv("data/Augmented_Training_set.csv")
df_orig_part = df_aug.iloc[:1285]
df_ext_part = df_aug.iloc[1285:]
relevancy_report = []
for c in sorted(df_orig['label'].unique()):
    sub_orig = df_orig_part[df_orig_part['label'] == c]
    sub_ext = df_ext_part[df_ext_part['label'] == c]
    if len(sub_ext) == 0:
        continue
    orig_lens = sub_orig[[f'packet_length_{i}' for i in range(5)]].values.flatten()
    ext_lens = sub_ext[[f'packet_length_{i}' for i in range(5)]].values.flatten()
    orig_dt = sub_orig['relative_time_1'].values
    ext_dt = sub_ext['relative_time_1'].values
    w_dist_len = wasserstein_distance(orig_lens, ext_lens)
    ks_stat, ks_p = ks_2samp(orig_lens, ext_lens)
    m_orig = np.mean(orig_lens)
    m_ext = np.mean(ext_lens)
    relevancy_report.append({
        'Traffic_Class': c,
        'Orig_N': len(sub_orig),
        'Ext_N': len(sub_ext),
        'Orig_Mean_Len': f"{m_orig:.1f} B",
        'Ext_Mean_Len': f"{m_ext:.1f} B",
        'Wasserstein_Dist': f"{w_dist_len:.1f} B",
        'KS_p_value': f"{ks_p:.4e}",
        'Physical_Compatibility': 'High (Strong Match)' if w_dist_len < 80 else ('Moderate' if w_dist_len < 250 else 'Domain Shift')
    })
df_rel = pd.DataFrame(relevancy_report)
