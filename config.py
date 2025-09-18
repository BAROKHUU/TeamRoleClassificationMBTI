# ==== ĐƯỜNG DẪN DỮ LIỆU ====
RAW_DATA_PATH = r"D:/MBTI project/data/mbti_1.csv"
CLEAN_DATA_PATH = r"D:/MBTI project/data/mbti_clean.csv"
FINAL_DATA_PATH = r"D:/MBTI project/data/mbti_1_augmentednclean.csv"
EMB_CACHE_PATH = r"D:/MBTI project/src/embeddings_multiclass.npy"

# ==== DATA AUGMENTATION ====
# Nhãn augment 1 lần
AUGMENT_1_TIME = ['ISFJ', 'ENFJ', 'ISTJ', 'ENTJ', 'ISFP', 'ISTP'] # 6 nhãn trung bình
# Nhãn augment 4 lần
AUGMENT_4_TIME = ['ESTJ', 'ESFJ', 'ESFP', 'ESTP']  # 4 nhãn ít nhất

# ==== THÔNG SỐ MÔ HÌNH ====
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
RANDOM_STATE = 42

# Logistic Regression evaluation
MAX_ITER = 5000
C_VALUES = [0.01, 0.1, 1, 10]

# ==== OUTPUT DIRECTORIES ====
OUTPUT_DIR_BINARY = "binary_model"
OUTPUT_DIR_MULTICLASS = "multiclass_model"

# ==== LƯU Ý ====
# - Các user khác chỉ cần thay đổi đường dẫn hoặc tham số ở đây.
# - Không nên sửa trực tiếp trong các file training/evaluation.
