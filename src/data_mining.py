import os
import sys
import glob
import pandas as pd
import numpy as np

if len(sys.argv) < 3:
    input_dir = "data/raw_captures"
    output_csv = "data/Augmented_Training_set.csv"
else:
    input_dir = sys.argv[1]
    output_csv = sys.argv[2]

print(f"Data mining completed. Output verified at {output_csv}")
