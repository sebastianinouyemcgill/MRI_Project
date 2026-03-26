# DATA
IMG_SIZE = (128, 128, 128)   # D, H, W
IN_CHANNELS = 1

SEQ_LEN = 2                  # [t1, t2] → predict t3
PRED_HORIZON = 1             # predicting next timepoint

# MODEL
FEATURE_DIM = 128            # CNN output size
HIDDEN_DIM = 64              # LSTM hidden size
NUM_LSTM_LAYERS = 1
DROPOUT = 0.1

# TRAINING
BATCH_SIZE = 4               # adjust based on GPU
EPOCHS = 50
LEARNING_RATE = 1e-4

# LOSS / TASK
TASK_TYPE = "classification"   # or "regression"
NUM_CLASSES = 1                # binary classification (logit output)

# DEVICE
DEVICE = "cuda"  # or "cpu"

# PATHS
DATA_DIR = "data/"
SPLIT_DIR = "splits/"

# DEBUG / TESTING
OVERFIT_BATCH = False
PRINT_SHAPES = False