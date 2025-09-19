# file: train_mbti_binary_xgb.py
import os, json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb
import joblib
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import config
# --- Config ---
DATA_PATH = config.FINAL_DATA_PATH
MODEL_NAME = config.MODEL_NAME
OUTPUT_DIR = config.OUTPUT_DIR_BINARY
EMB_CACHE = config.EMB_CACHE_PATH
BATCH_SIZE = config.BATCH_SIZE
RANDOM_STATE = config.RANDOM_STATE

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Load data ---
df = pd.read_csv(DATA_PATH).dropna(subset=["type", "posts"])
X_texts = df["posts"].astype(str).tolist()
y_types = df["type"].astype(str).str.upper().tolist()

# --- Embedding ---
# This function will not change compared to script multiclass
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

# --- Create 4 Binary Labels ---
tasks = {
    "EI": [t[0] for t in y_types],
    "SN": [t[1] for t in y_types],
    "TF": [t[2] for t in y_types],
    "JP": [t[3] for t in y_types],
}

# --- Training loop for 4 Binary models ---
for task_name, labels in tasks.items():
    print(f"\n{'='*20} TRAINING FOR: {task_name} {'='*20}")
    
    task_output_dir = os.path.join(OUTPUT_DIR, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    # --- Encoding Labels (Ex: 'E' -> 0, 'I' -> 1) ---
    le = LabelEncoder()
    y = le.fit_transform(labels)
    # Tạo mapping để tham chiếu sau này
    label2id = {label: int(idx) for label, idx in zip(le.classes_, le.transform(le.classes_))}
    id2label = {int(idx): label for label, idx in label2id.items()}
    
    # --- Train/Val/Test Split (70/15/15) - Same Logic in script multiclass ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_emb, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"Dataset split for {task_name}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # --- Model: XGBoost binary ---
    # Using the same parameters for multiclass for comparision
    clf = xgb.XGBClassifier(
        objective="binary:logistic",  # change objective for Binary problem
        eval_metric="logloss",
        n_estimators=1000,
        learning_rate=0.001,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        tree_method="hist",
        device="cuda" if device == "cuda" else "cpu",
        n_jobs=-1,
        verbosity=1,
        # Early Stopping to avoid Overfitting and find best iteration
    )

    print(f"Training XGBoost for {task_name}...")
    # Include X_train, y_train into eval_set to draw train loss
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=20)

    # --- Evaluate on validation ---
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    print(f"\n🔎 Validation Results for {task_name}")
    print("Accuracy:", val_acc)
    print("Macro-F1:", val_f1)
    print(classification_report(y_val, y_val_pred, target_names=label2id.keys()))

    # --- Confusion Matrix (Validation Set) ---
    cm = confusion_matrix(y_val, y_val_pred, labels=list(id2label.keys()))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(label2id.keys()),
                yticklabels=list(label2id.keys()))
    plt.title(f"Confusion Matrix - {task_name} (Validation)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(task_output_dir, "confusion_matrix_val.png"))
    plt.close()

    # --- Training Curve (Log Loss) ---
    results = clf.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    plt.figure()
    plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
    plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title(f'XGBoost Training vs Validation Loss ({task_name})')
    plt.savefig(os.path.join(task_output_dir, "training_curve.png"))
    plt.close()

    # --- F1 per Class (Validation Set) ---
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, labels=list(id2label.keys()))
    plt.figure(figsize=(8, 5))
    plt.bar(label2id.keys(), f1)
    plt.title(f"F1-score per Class - {task_name} (Validation)")
    plt.ylabel("F1-score")
    plt.tight_layout()
    plt.savefig(os.path.join(task_output_dir, "f1_per_class.png"))
    plt.close()

    # --- Save model + label map ---
    model_path = os.path.join(task_output_dir, f"xgb_{task_name}.joblib")
    joblib.dump(clf, model_path)
    with open(os.path.join(task_output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print(f"\n✅ Model and plots for {task_name} saved to {task_output_dir}")

print("\n🎉 All binary classification models have been trained successfully!")