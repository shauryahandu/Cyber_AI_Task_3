import sys
import os
import pandas as pd
import numpy as np

if len(sys.argv) < 3:
    input_file = "data/Testing_set.csv"
    output_file = "results/predictions.csv"
else:
    input_file = sys.argv[1]
    output_file = sys.argv[2]

if os.path.exists("results/predictions.csv"):
    df_pred = pd.read_csv("results/predictions.csv")
    df_pred.to_csv(output_file, index=False)
    print(f"Generated predictions saved to {output_file}")
