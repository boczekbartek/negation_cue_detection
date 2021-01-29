# -------------------- DATA ------------------
# Paths to datasets
DEV_DATA = "data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt"
TRAIN_DATA = "data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt"
TEST_CIRCLE_DATA = "data/SEM-2012-SharedTask-CD-SCO-test-circle.txt"
TEST_CARDBOARD_DATA = "data/SEM-2012-SharedTask-CD-SCO-test-cardboard.txt"

# -------------------- FEATURES ------------------
# Paths to files with generated features
TRAIN_FEATURES = "data/SEM-2012-SharedTask-CD-SCO-training-simple.v2-features.tsv"
DEV_FEATURES = "data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2-features.tsv"
TEST_CIRCLE_FEATURES = "data/SEM-2012-SharedTask-CD-SCO-test-circle-features.tsv"
TEST_CARDBOARD_FEATURES = "data/SEM-2012-SharedTask-CD-SCO-test-cardboard-features.tsv"
# Mapping
TAG2IDX = {"B-NEG": 0, "O": 1, "I-NEG": 2}

# -------------------- PRE-TRAINED ------------------
# name of pretrained model from HuggingFace Transformer library
PRETRAINED_MODEL = "bert-base-uncased"

# -------------------- CHECKPOINTS ------------------
# path to baseline model checkpoint
BSL_MODEL_CKPT = "neg_cue_detection_model_baseline"
# path to baseline+lexicals model checkpoint
LEX_MODEL_CKPT = "neg_cue_detection_model_lex"

# -------------------- TRAIN-PARAMS ------------------
EPOCHS = 25
BATCH_SIZE = 32
SEED = 777

# -------------------- Logging ------------------
LOG_LEVEL = "INFO"

# -------------------- OTHERS ------------------
REPORTS_DIR = "reports"

