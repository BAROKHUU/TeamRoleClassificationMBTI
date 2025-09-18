# file: evaluate_mbti_primary.py
import os, json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config
# --- Config ---
DATA_PATH = config.FINAL_DATA_PATH
MODEL_NAME = config.MODEL_NAME
RANDOM_STATE = config.RANDOM_STATE
OUTPUT_DIR = config.OUTPUT_DIR_BINARY
EMB_CACHE = config.EMB_CACHE_PATH
BATCH_SIZE = config.BATCH_SIZE
MAX_ITER = config.MAX_ITER
C_VALUES = config.C_VALUES

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Load data ---
df = pd.read_csv(DATA_PATH).dropna(subset=["type", "posts"])
X_texts = df["posts"].astype(str).tolist()
y_types = df["type"].astype(str).tolist()  # full 16-class labels

# --- Embedding with mean pooling ---
def encode_posts(posts, embedder):
    parts = posts.split("|||")
    emb = embedder.encode(parts, convert_to_numpy=True, batch_size=BATCH_SIZE)
    return emb.mean(axis=0)

embedder = SentenceTransformer(MODEL_NAME, device=device)

if not os.path.exists(EMB_CACHE):
    print("Encoding embeddings (this may take a while)...")
    X_emb = np.array([encode_posts(p, embedder) for p in tqdm(X_texts, desc="Encoding posts")])
    np.save(EMB_CACHE, X_emb)
else:
    print("Loading cached embeddings...")
    X_emb = np.load(EMB_CACHE)

# --- Train/val/test split: 70/15/15 ---
X_temp, X_test, y_temp, y_test = train_test_split(
    X_emb, y_types, test_size=0.15, random_state=RANDOM_STATE, stratify=y_types
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_temp
)  # 0.1765 * 85% ≈ 15%

# --- Grid search for best C ---
best_c, best_f1 = None, -1
print("\nPerforming Grid Search for best C value...")
for c in C_VALUES:
    clf_val = LogisticRegression(
        max_iter=MAX_ITER,
        solver="saga",
        random_state=RANDOM_STATE,
        class_weight="balanced",
        C=c,
        n_jobs=-1,
        tol=1e-3,
        verbose=0
    )
    clf_val.fit(X_train, y_train)
    preds_val = clf_val.predict(X_val)
    f1_val = f1_score(y_val, preds_val, average="macro")
    print(f"  C={c} -> Val Macro-F1: {f1_val:.4f}")
    if f1_val > best_f1:
        best_f1 = f1_val
        best_c = c

print(f"\nBest C: {best_c} (Val Macro-F1={best_f1:.4f})")

# --- Retrain on train+val with best C ---
print("\nRetraining model on combined train+val data with best C...")
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])
clf = LogisticRegression(
    max_iter=MAX_ITER,
    solver="saga",
    random_state=RANDOM_STATE,
    class_weight="balanced",
    C=best_c,
    n_jobs=-1,
    tol=1e-3,
    verbose=0
)
clf.fit(X_trainval, y_trainval)

# --- Evaluate on test ---
print("\nEvaluating final model on the test set...")
preds_test = clf.predict(X_test)
acc = accuracy_score(y_test, preds_test)
macro_f1 = f1_score(y_test, preds_test, average="macro")
print(f"\nTest Accuracy: {acc:.4f}, Test Macro-F1: {macro_f1:.4f}")

report = classification_report(y_test, preds_test, output_dict=True)
cm = confusion_matrix(y_test, preds_test, labels=clf.classes_)

print("\nClassification Report:\n", classification_report(y_test, preds_test))
# print("Confusion Matrix:\n", cm) # This can be very large, so we'll rely on the saved image

# --- Save model ---
model_path = os.path.join(OUTPUT_DIR, "primary_mbti_clf.joblib")
joblib.dump(clf, model_path)

# --- Save confusion matrix heatmap ---
plt.figure(figsize=(12, 10)) # Increased size for better readability
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=clf.classes_, yticklabels=clf.classes_)
# --- THIS IS THE CORRECTED LINE ---
plt.title("Confusion Matrix - Test Set", fontsize=16) 
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "primary_confusion_matrix.png"))
plt.close()

# --- Save summary ---
results = {
    "best_c": best_c,
    "val_macro_f1": best_f1,
    "test_accuracy": acc,
    "test_macro_f1": macro_f1,
    "report": report,
    "confusion_matrix": cm.tolist(),
    "model_path": model_path
}
with open(os.path.join(OUTPUT_DIR, "primary_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Evaluation done. Results saved to '{OUTPUT_DIR}' directory.")