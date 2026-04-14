import os

class Config:
    def __init__(self):
        self.LOCAL_ROOT   = "/content/drive/MyDrive/MRI_Project"
        self.DATA_ROOT = os.path.join(self.LOCAL_ROOT, "data/processed/Processed Post")
        self.SPLIT_ROOT = os.path.join(self.LOCAL_ROOT, "splits")
        self.JSON_ROOT = os.path.join(self.LOCAL_ROOT, "data/json")
        self.COLAB_ROOT = "/content/MRI_Data/Processed Post"
        self.CHECKPOINT_ROOT = "/content/local_checkpoints"

        self.SEQ_LEN = 3
        self.GROWTH_THRESHOLD = 0.2
        self.RANDOM_SEED = 42
        self.BATCH_SIZE = 4
        self.FEATURE_DIM = 256
        self.MODALITY = "POST"
        self.NUM_EPOCHS = 20
        self.LR = 1e-4
        self.POS_WEIGHT = 2470 / 1347 # update based on SEQ_LEN

cfg = Config()

# DATA_ROOT =  os.path.join(LOCAL_ROOT, 'data/processed') # "../../data/processed"  # "/Volumes/SSD 2/Projects/MRI Project/Processed Data"
# SPLIT_ROOT = os.path.join(LOCAL_ROOT, 'splits') # "../../splits/mini"             # for train/test/val txts
# JSON_ROOT = os.path.join(LOCAL_ROOT, 'data/json') # "../../data/json/mini"           # for volumes.json / labels.json
# CHECKPOINT_ROOT = os.path.join(LOCAL_ROOT, 'checkpoints') # "../../checkpoint/minitest"   # for saving model checkpoints