# config.py
import os

LOCAL_ROOT    = '/content/MRI_Project'

# Paths
DATA_ROOT =  os.path.join(LOCAL_ROOT, 'data/processed') # "../../data/processed"  # "/Volumes/SSD 2/Projects/MRI Project/Processed Data"
SPLIT_ROOT = os.path.join(LOCAL_ROOT, 'splits') # "../../splits/mini"             # for train/test/val txts
JSON_ROOT = os.path.join(LOCAL_ROOT, 'data/json') # "../../data/json/mini"           # for volumes.json / labels.json
CHECKPOINT_ROOT = os.path.join(LOCAL_ROOT, 'checkpoints') # "../../checkpoint/minitest"   # for saving model checkpoints

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