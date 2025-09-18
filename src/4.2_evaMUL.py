# file: evaluate_mbti_multiclass.py
import os
import pandas as pd
import torch
import joblib
import json
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config
# --- Config ---
DATA_PATH = config.FINAL_DATA_PATH
MODEL_DIR = config.OUTPUT_DIR_MULTICLASS
MODEL_NAME = config.MODEL_NAME
RANDOM_STATE = config.RANDOM_STATE

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Load data ---
df = pd.read_csv(DATA_PATH).dropna(subset=["type", "posts"])
X_texts = df["posts"].astype(str).tolist()
y_types = df["type"].astype(str).str.upper().tolist()

# --- Split train/val/test: 70/15/15 ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X_texts, y_types, test_size=0.30, random_state=RANDOM_STATE, stratify=y_types
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Dataset split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# --- Load trained model + label map ---
model_path = os.path.join(MODEL_DIR, "xgb_multiclass.joblib")
clf = joblib.load(model_path)

with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
    label_map = json.load(f)
label2id = label_map["label2id"]
id2label = {int(i): t for i, t in label_map["id2label"].items()}

# --- Embedding test set ---
print("\nEmbedding test set...")
embedder = SentenceTransformer(MODEL_NAME, device=device)
X_test_emb = embedder.encode(X_test, convert_to_numpy=True, show_progress_bar=True)

# --- Predict ---
print("\nPredicting on test set...")
y_test_pred_ids = clf.predict(X_test_emb)
y_test_pred = [id2label[i] for i in y_test_pred_ids]

# --- Evaluate only on test set ---
print("\n🔎 Evaluation on Test Set (15%)")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, labels=list(label2id.keys())))

# --- NEW: Generate and save confusion matrix ---
print("\n📊 Generating and saving confusion matrix...")

# 1. Get the class labels in the correct order for the plot
class_labels = list(label2id.keys())

# 2. Calculate the confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=class_labels)

# 3. Create a DataFrame for better labeling with seaborn
cm_df = pd.DataFrame(cm,
                     index=class_labels,
                     columns=class_labels)

# 4. Create the plot
plt.figure(figsize=(14, 11))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Test Set', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout() # Adjust layout to prevent labels from being cut off

# 5. Save the plot to a file
output_path = os.path.join(MODEL_DIR, "confusion_matrix_test.png")
plt.savefig(output_path)

print(f"\n✅ Confusion matrix saved successfully to: {output_path}")
