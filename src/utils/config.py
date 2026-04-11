# config.py
import os

# Paths
DATA_ROOT =  "../../data/processed"  # "/Volumes/SSD 2/Projects/MRI Project/Processed Data"
SPLIT_ROOT = "../../splits"             # for train/test/val txts
JSON_ROOT = "../../data/json"           # for volumes.json / labels.json
CHECKPOINT_ROOT = "../../checkpoint/minitest2"   # for saving model checkpoints

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

NUM_EPOCHS = 10
LR = 1e-4