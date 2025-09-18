# file: train_mbti_multiclass.py
import os, json, random
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)
import xgboost as xgb
import joblib
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import config
# --- Config ---
DATA_PATH = config.FINAL_DATA_PATH
MODEL_NAME = config.MODEL_NAME
OUTPUT_DIR = config.OUTPUT_DIR_MULTICLASS
EMB_CACHE = config.EMB_CACHE_PATH
BATCH_SIZE = config.BATCH_SIZE
RANDOM_STATE = config.RANDOM_STATE

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Seed reproducibility ---
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# --- Load data ---
df = pd.read_csv(DATA_PATH).dropna(subset=["type", "posts"])
X_texts = df["posts"].astype(str).tolist()
y_types = df["type"].astype(str).str.upper().tolist()

# Encode MBTI to numeric labels
mbti_types = sorted(df["type"].unique())  # 16 types
label2id = {t: i for i, t in enumerate(mbti_types)}
id2label = {i: t for t, i in label2id.items()}
y = np.array([label2id[t] for t in y_types])

# --- Embedding ---
def encode_posts(posts, embedder):
    parts = posts.split("|||")
    emb = embedder.encode(parts, convert_to_numpy=True, batch_size=BATCH_SIZE)
    return emb.mean(axis=0)

embedder = SentenceTransformer(MODEL_NAME, device=device)

if not os.path.exists(EMB_CACHE):
    print("Encoding embeddings...")
    X_emb = np.array([encode_posts(p, embedder) for p in tqdm(X_texts, desc="Encoding posts")])
    np.save(EMB_CACHE, X_emb)
else:
    print("Loading cached embeddings...")
    X_emb = np.load(EMB_CACHE)

# --- Train/Val/Test Split (70/15/15) ---
X_temp, X_test, y_temp, y_test = train_test_split(
    X_emb, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_temp
)  # 0.1765 * 85% ≈ 15%

print(f"Dataset split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# --- Model: XGBoost multiclass ---
clf = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(mbti_types),
    n_estimators=1000,
    learning_rate=0.001,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric=["mlogloss", "merror"],
    tree_method="gpu_hist" if device == "cuda" else "hist",
    predictor="gpu_predictor" if device == "cuda" else "cpu_predictor",
    gpu_id=0 if device == "cuda" else -1,
    use_label_encoder=False,
    n_jobs=-1,
    verbosity=1
)

print("Training XGBoost...")
clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)], 
    verbose=20
)

# --- Evaluate on validation ---
y_val_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average="macro")

print("\n🔎 Validation Results")
print("Accuracy:", val_acc)
print("Macro-F1:", val_f1)
print(classification_report(y_val, y_val_pred, target_names=[id2label[i] for i in range(len(mbti_types))]))

# --- Evaluate on test ---
y_test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average="macro")

print("\n🧪 Test Results")
print("Accuracy:", test_acc)
print("Macro-F1:", test_f1)
print(classification_report(y_test, y_test_pred, target_names=[id2label[i] for i in range(len(mbti_types))]))

# --- Confusion Matrix (Validation) ---
cm = confusion_matrix(y_val, y_val_pred, labels=list(range(len(mbti_types))))
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[id2label[i] for i in range(len(mbti_types))],
            yticklabels=[id2label[i] for i in range(len(mbti_types))])
plt.title("Confusion Matrix - Validation Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_val.png"))
plt.close()

# --- Training Curves ---
results = clf.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# Log Loss
plt.figure()
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train LogLoss')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Val LogLoss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title('XGBoost Training vs Validation Log Loss')
plt.savefig(os.path.join(OUTPUT_DIR, "training_curve_logloss.png"))
plt.close()

# Accuracy (1 - merror)
plt.figure()
plt.plot(x_axis, 1 - np.array(results['validation_0']['merror']), label='Train Acc')
plt.plot(x_axis, 1 - np.array(results['validation_1']['merror']), label='Val Acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('XGBoost Training vs Validation Accuracy')
plt.savefig(os.path.join(OUTPUT_DIR, "training_curve_accuracy.png"))
plt.close()

# --- F1 per Class (Validation) ---
prec, rec, f1, _ = precision_recall_fscore_support(
    y_val, y_val_pred, labels=list(range(len(mbti_types)))
)
plt.figure(figsize=(12,6))
plt.bar([id2label[i] for i in range(len(mbti_types))], f1)
plt.title("F1-score per MBTI Type (Validation Set)")
plt.xticks(rotation=45)
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_per_class.png"))
plt.close()

# --- Save model + label map + evals_result ---
model_path = os.path.join(OUTPUT_DIR, "xgb_multiclass.joblib")
joblib.dump(clf, model_path)
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "evals_result.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Model and plots saved to {OUTPUT_DIR}")
