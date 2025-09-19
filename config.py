# ==== Data Paths ====
RAW_DATA_PATH = r"D:/MBTI project/data/mbti_1.csv"
CLEAN_DATA_PATH = r"D:/MBTI project/data/mbti_clean.csv"
FINAL_DATA_PATH = r"D:/MBTI project/data/mbti_1_augmentednclean.csv"
EMB_CACHE_PATH = r"D:/MBTI project/src/embeddings_multiclass.npy"

# ==== DATA AUGMENTATION ====
# Label augment 1 time
AUGMENT_1_TIME = ['ISFJ', 'ENFJ', 'ISTJ', 'ENTJ', 'ISFP', 'ISTP'] # 6 nhãn trung bình
# Labels augment 4 times
AUGMENT_4_TIME = ['ESTJ', 'ESFJ', 'ESFP', 'ESTP']  # 4 nhãn ít nhất

# ==== Model Configuration ====
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
RANDOM_STATE = 42

# Logistic Regression evaluation
MAX_ITER = 5000
C_VALUES = [0.01, 0.1, 1, 10]

# ==== OUTPUT DIRECTORIES ====
OUTPUT_DIR_BINARY = "binary_model"
OUTPUT_DIR_MULTICLASS = "multiclass_model"

# ==== NOTE ====
# - Other users only need to change the paths or parameters here.
# - Do not modify the training/evaluation files directly.
