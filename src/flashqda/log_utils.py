# log_utils.py — utility functions for managing processing logs

import pandas as pd
import os


def update_log(log_path, message):
    with open(log_path, "a") as f:
        f.write(f"{message}\n")


def get_start_ids(processed_file):
    if not os.path.exists(processed_file):
        return set()
    df = pd.read_csv(processed_file)
    return set(df.get("id", []))
