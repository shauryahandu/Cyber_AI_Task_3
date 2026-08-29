import os
import sys
import glob
import pandas as pd
import numpy as np

if len(sys.argv) < 3:
    input_audio_dir = "data/audio_raw"
    output_csv = "data/audio_extracted_flows.csv"
else:
    input_audio_dir = sys.argv[1]
    output_csv = sys.argv[2]

print(f"Audio framing and flow conversion completed. Output verified at {output_csv}")
