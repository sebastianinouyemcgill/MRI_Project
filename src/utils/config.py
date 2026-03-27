# config.py
import os

# Paths
DATA_ROOT = "../../data/processed"       # for preprocessed tensors
SPLIT_ROOT = "../../splits"             # for train/test/val txts
JSON_ROOT = "../../data/json"           # for volumes.json / labels.json

# Sliding window / sequences
SEQ_LEN = 2                              # number of timepoints per sequence

# Labels
GROWTH_THRESHOLD = 0.2

# Randomness
RANDOM_SEED = 42

# CNN/LSTM
BATCH_SIZE = 4
FEATURE_DIM = 256  # placeholder for CNN output size

# Misc
MODALITY = "POST"  # only using POST for now