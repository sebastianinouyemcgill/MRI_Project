import os
import json
from utils.config import cfg
from datetime import datetime

def parse_date(d):
    return datetime.strptime(d, "%Y-%m-%d")

def create_sliding_windows(label_json, seq_len):
    """
    Returns:
        X: list of dicts with:
            {
                "pid": str,
                "dates": [t1, t2, ..., tT],
                "start_idx": int   # index into sorted transitions
            }
        y: list of labels
    """

    X = []
    y = []

    for pid, transitions in label_json.items():
        # sort
        sorted_items = sorted(
            transitions.items(),
            key=lambda x: parse_date(x[1]["date_ti"])
        )

        # continuity check
        valid = True
        for i in range(len(sorted_items) - 1):
            if sorted_items[i][1]["date_tf"] != sorted_items[i + 1][1]["date_ti"]:
                print(f"[SKIP] {pid} has non-contiguous dates")
                valid = False
                break
        if not valid:
            continue

        # build date timeline
        dates = [item[1]["date_ti"] for item in sorted_items]
        dates.append(sorted_items[-1][1]["date_tf"])

        if len(dates) < seq_len + 1:
            continue

        # sliding windows
        for i in range(len(dates) - seq_len):
            input_dates = dates[i:i + seq_len]

            # label = transition from last input → next
            transition = sorted_items[i + seq_len - 1][1]
            label = transition["label"]

            X.append({
                "pid": pid,
                "dates": input_dates,
                "start_idx": i 
            })
            y.append(label)

    return X, y

if __name__ == "__main__":
    DATA_ROOT = cfg.DATA_ROOT
    LABEL_JSON = cfg.JSON_ROOT
    SEQ_LEN = cfg.SEQ_LEN

    with open(LABEL_JSON, "r") as f:
        labels = json.load(f)

    X, y = create_sliding_windows(labels, SEQ_LEN)

    print(f"Total sequences: {len(X)}")