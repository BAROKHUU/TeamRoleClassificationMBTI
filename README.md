# MBTI Personality Classification (NLP + XGBoost)

This project develops a system for classifying **MBTI personality types** based on textual posts, using:
- **Data preprocessing** (cleaning + augmentation)
- **Sentence Transformers** for embedding generation
- **XGBoost** for training and evaluation  (both binary & multiclass)
```
MBTI_project/
│
├── data/                 # Contains raw, cleaned, and augmented datasets
│   ├── mbti_1.csv
│   ├── mbti_clean.csv
│   └── mbti_1_augmentednclean.csv
│
├── sbert_all-MiniLM-L6-v2/ # Stores the pre-trained Sentence Transformer model
│
├── src/                  # Main source code
│   ├── 1_view_data.py
│   ├── 2.1_data_clean.py
│   ├── 2.2_data_augmentation.py
│   ├── 3.1_training_binary.py
│   ├── 3.2_training_XGBoost.py
│   ├── 4.1_evaBIN.py
│   ├── 4.2_evaMUL.py
│   ├── binary_model/       # Stores trained binary classification models (.joblib) and reports (.json)
│   └── multiclass_model/   # Stores trained multi-class model and evaluation results
│
├── config.py             # Configuration variables (paths, parameters)
├── demo.ipynb            # Jupyter Notebook for the Gradio-based interactive demo
├── requirements.txt      # List of Python dependencies
└── README.md             # Project documentation
```

Pretrained Model --> .joblib \n

Summary --> .json


## ⚙️ Environment Setup

### 1. Clone repo + Setup
```bash
git clone https://github.com/BAROKHUU/TeamRoleClassificationMBTI.git
cd MBTI_project
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```
## 🚀 How to Run the Code
1. Prepare the dataset
Place the original file mbti_1.csv inside the data/ directory.
This file will be processed into multiple versions during execution.


2. Data Filtering
```bash
python src/2.1_data_clean.py
```
Input: data/mbti_1.csv
Output: data/mbti_clean.csv


3. Data Augmentation
```bash
python src/2.2_data_augmentation.py
```
Input: data/mbti_clean.csv
Output: data/mbti_1_augmentednclean.csv


4. Training

4.1. Binary classification
```bash
python src/3.1_training_binary.py
```
Output: Model XGBoost Binary inside the binary_model/ directory.

4.2. Multiclass classification 
```bash
Run: python src/3.2_training_XGBoost.py
```
Output: Model XGBoost Multiclass inside the multiclass_model/ directory.


5. Evaluation

5.1. Logistic Regression baseline (binary)
```bash
python src/4.1_evaBIN.py
```
Output: binary_model/primary_mbti_clf.joblib + confusion matrix + report JSON

5.2. XGBoost multiclass evaluation
```bash
python src/4.2_evaXGB.py
```
Output: multiclass_model/confusion_matrix_test.png + classification report


6. Demo Inference
```bash
jupyter notebook demo.ipynb # http://127.0.0.1:7860
```
Open Jupyter notebook and run demo UI through Gradio.