# log_utils.py — utility functions for managing processing logs

import pandas as pd
import os

def update_log(log_path, message, level=None):
    """
    Append a message to the log file (and echo to stdout).
    Accepts an optional 'level' (e.g., 'info', 'error') for consistent logging.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Format message with optional level prefix
    if level:
        line = f"[{level.upper()}] {message}"
    else:
        line = message

    # Write to file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    # Also print to console for quick debugging
    # print(line)


def get_start_ids(processed_file):
    """
    Read a CSV of previously processed items and return a set of IDs.
    Returns an empty set if the file doesn't exist or has no 'id' column.
    """
    if not os.path.exists(processed_file):
        return set()
    df = pd.read_csv(processed_file)
    return set(df.get("id", []))
