"""tools/check_regressors.py

Utility to inspect pickled/joblib Prophet models for expected regressors.
Usage:
  python tools/check_regressors.py --model models/antipolo_chance_of_rain_prophet.pkl [--sample data/processed/humidity.csv]

The script will try to detect expected regressors from common Prophet model attributes
(e.g. model.extra_regressors, model.train_component_cols) and compare them to a sample
CSV's columns if provided.
"""
import argparse
import os
import joblib


def inspect_model(path: str):
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        return None

    try:
        model = joblib.load(path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    regressors = set()
    
    print(f"Model: {os.path.basename(path)}")
    print(f"Model type: {type(model)}")

    # Method 1: Check for extra_regressors attribute (most common in Prophet)
    try:
        if hasattr(model, 'extra_regressors') and model.extra_regressors:
            if isinstance(model.extra_regressors, dict):
                regressors.update(model.extra_regressors.keys())
                print(f"Found extra_regressors: {list(model.extra_regressors.keys())}")
            elif isinstance(model.extra_regressors, (list, tuple)):
                regressors.update(model.extra_regressors)
                print(f"Found extra_regressors (list): {list(model.extra_regressors)}")
    except Exception as e:
        print(f"Error checking extra_regressors: {e}")

    # Method 2: Check train_component_cols
    try:
        if hasattr(model, 'train_component_cols') and model.train_component_cols:
            print(f"train_component_cols keys: {list(model.train_component_cols.keys())}")
            for component, cols in model.train_component_cols.items():
                if component == 'extra_regressors' and isinstance(cols, (list, tuple)):
                    regressors.update(cols)
                    print(f"Found regressors in train_component_cols: {cols}")
    except Exception as e:
        print(f"Error checking train_component_cols: {e}")

    # Method 3: Check for regressor-related DataFrame columns in training data
    try:
        if hasattr(model, 'history') and hasattr(model.history, 'columns'):
            hist_cols = list(model.history.columns)
            print(f"History columns: {hist_cols}")
            # Standard Prophet columns are 'ds', 'y', 'cap', 'floor'
            standard_cols = {'ds', 'y', 'cap', 'floor'}
            potential_regressors = [col for col in hist_cols if col not in standard_cols]
            if potential_regressors:
                regressors.update(potential_regressors)
                print(f"Potential regressors from history: {potential_regressors}")
    except Exception as e:
        print(f"Error checking history: {e}")

    # Method 4: Debug - print all attributes that might contain regressor info
    print("\nDebugging - checking model attributes:")
    for attr in ['extra_regressors', 'regressors', 'train_component_cols', 'component_modes', 'history']:
        if hasattr(model, attr):
            val = getattr(model, attr)
            print(f"  {attr}: {type(val)} = {val if not hasattr(val, 'shape') else f'<{type(val).__name__} shape={val.shape}>'}")

    print(f"\nFinal detected regressors ({len(regressors)}):")
    if regressors:
        for r in sorted(regressors):
            print(f"  - {r}")
    else:
        print("  None detected")

    return regressors


def compare_with_sample(regressors, sample_csv):
    if not os.path.exists(sample_csv):
        print(f"Sample CSV not found: {sample_csv}")
        return

    try:
        import pandas as pd
        df = pd.read_csv(sample_csv)
        cols = set(df.columns)

        print(f"Sample columns ({len(cols)}): {sorted(list(cols))[:10]}{'...' if len(cols)>10 else ''}")

        missing = [r for r in regressors if r not in cols]
        if missing:
            print("Missing regressors in sample CSV:")
            for m in missing:
                print(f"  - {m}")
        else:
            print("All detected regressors are present in sample CSV.")
    except ImportError:
        print("pandas not available - cannot compare with CSV")
    except Exception as e:
        print(f"Error reading CSV: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model .pkl (joblib)')
    parser.add_argument('--sample', required=False, help='Optional CSV to compare columns')
    args = parser.parse_args()

    regs = inspect_model(args.model)
    if regs and args.sample:
        compare_with_sample(regs, args.sample)
