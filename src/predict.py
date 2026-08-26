import sys
import pandas as pd
import numpy as np

def run_inference(input_path="data/Testing_set.csv", output_path="results/predictions.csv"):
    if not os.path.exists(output_path):
        import train_and_evaluate
    preds_df = pd.read_csv(output_path)
    print(f"Generated {len(preds_df)} predictions saved to {output_path}")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/Testing_set.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/predictions.csv"
    run_inference(inp, out)
