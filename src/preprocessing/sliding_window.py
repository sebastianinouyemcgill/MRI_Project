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
        # ---- robust sort ----
        sorted_items = sorted(
            transitions.items(),
            key=lambda x: parse_date(x[1]["date_ti"])
        )

        # ---- continuity check (hard fail = skip patient) ----
        valid = True
        for i in range(len(sorted_items) - 1):
            if sorted_items[i][1]["date_tf"] != sorted_items[i + 1][1]["date_ti"]:
                print(f"[SKIP] {pid} has non-contiguous dates")
                valid = False
                break
        if not valid:
            continue

        # ---- build date timeline ----
        dates = [item[1]["date_ti"] for item in sorted_items]
        dates.append(sorted_items[-1][1]["date_tf"])

        if len(dates) < seq_len + 1:
            continue

        # ---- sliding windows ----
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
    import os
    import json

    DATA_ROOT = cfg.DATA_ROOT
    LABEL_JSON = cfg.JSON_ROOT
    SEQ_LEN = cfg.SEQ_LEN

    with open(LABEL_JSON, "r") as f:
        labels = json.load(f)

    X, y = create_sliding_windows(labels, SEQ_LEN)

    print(f"Total sequences: {len(X)}")

    issues = {
        "missing_folder": 0,
        "bad_days_len": 0,
        "date_order": 0,
    }

    for i in range(min(50, len(X))):  # sample first 50
        meta = X[i]
        pid = meta["pid"]
        dates = meta["dates"]
        start_idx = meta["start_idx"]

        # ---- check ordering ----
        parsed = [parse_date(d) for d in dates]
        if parsed != sorted(parsed):
            print(f"[ORDER ERROR] {pid} {dates}")
            issues["date_order"] += 1

        # ---- check folders ----
        for d in dates:
            folder = os.path.join(DATA_ROOT, pid, d)
            if not os.path.exists(folder):
                print(f"[MISSING] {pid} {d}")
                issues["missing_folder"] += 1

        # ---- check days alignment ----
        sorted_items = sorted(
            labels[pid].items(),
            key=lambda x: parse_date(x[1]["date_ti"])
        )

        days = [0.0] + [
            float(sorted_items[j][1]["days_elapsed"])
            for j in range(start_idx, start_idx + len(dates) - 1)
        ]

        if len(days) != len(dates):
            print(f"[DAYS LEN ERROR] {pid}")
            issues["bad_days_len"] += 1

        # ---- debug print one clean sample ----
        if i == 0:
            print("\n=== SAMPLE ===")
            print("PID:", pid)
            print("Dates:", dates)
            print("Start idx:", start_idx)
            print("Days:", days)
            print("Label:", y[i])

    print("\n===== SUMMARY =====")
    for k, v in issues.items():
        print(f"{k}: {v}")