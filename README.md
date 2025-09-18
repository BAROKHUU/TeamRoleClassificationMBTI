# MBTI Personality Classification (NLP + XGBoost)

Dự án này xây dựng một hệ thống phân loại **MBTI personality types** dựa trên văn bản (posts), sử dụng:
- **Data preprocessing** (clean + augmentation)
- **Sentence Transformers** để sinh embedding
- **XGBoost** cho huấn luyện và đánh giá (binary & multiclass)

MBTI_project/
│── data/ # Nơi chứa dataset gốc & file đã xử lý
│ ├── mbti_1.csv
│ ├── mbti_clean.csv
│ ├── mbti_1_augmentednclean.csv
│ ├── training_pipeline.png
│ ├── Inference_pipeline.png
│ ├── eda_label_distribution(clean+balanced).png
│ └── eda_label_distribution(unprocessed).png
│
│── sbert_all-MiniLM-L6-v2 # Lưu model
│
│── src/ # Code chính
│ ├── 1_view_data.py # EDA: thống kê, visualization
│ ├── 2.1_data_clean.py # Làm sạch dữ liệu (remove link,…)
│ ├── 2.2_data_augmentation.py # Data augmentation với WordNet
│ ├── 3.1_training_binary.py # Huấn luyện 4 mô hình nhị phân (EI, SN, TF, JP)
│ ├── 3.2_training_XGBoost.py # Huấn luyện mô hình đa lớp (16 MBTI types)
│ ├── 4.1_evaBIN.py # Đánh giá mô hình Logistic Regression (binary)
│ ├── 4.2_evaMUL.py # Đánh giá mô hình XGBoost (multiclass)
│ ├── GPU_check.py # Kiểm tra đã sử dụng GPU chưa
│ ├── embeddings_multiclass.npy # Lưu text embeddings
│ ├── binary_model/ # Kết quả huấn luyện Binary
│ └── multiclass_model/ # Kết quả huấn luyện Multiclass
│
│── demo.ipynb # demo
│── requirements.txt # Danh sách thư viện
│── README.md # Tài liệu này, hướng dẫn sử dụng
├── config.py # Lưu biến cấu hình


Pretrained Model --> .joblib
Summary --> .json


## ⚙️ Cài đặt môi trường

### 1. Clone repo + Setup
```bash
git clone https://github.com/BAROKHUU/TeamRoleClassificationMBTI.git
cd MBTI_project
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

### Hướng dẫn chạy code
1. Chuẩn bị dữ liệu
Đặt file dữ liệu gốc mbti_1.csv vào thư mục data/.
File này sẽ được xử lý thành nhiều version khác nhau trong quá trình chạy.


2. Data Filtering
Run: python src/2.1_data_clean.py
Input: data/mbti_1.csv
Output: data/mbti_clean.csv


3. Data Augmentation
Run: python src/2.2_data_augmentation.py
Input: data/mbti_clean.csv
Output: data/mbti_1_augmentednclean.csv


4. Training
4.1. Binary classification
Run: python src/3.1_training_binary.py
Output: mô hình XGBoost nhị phân trong thư mục binary_model/

4.2. Multiclass classification 
Run: python src/3.2_training_XGBoost.py
Output: mô hình XGBoost đa lớp trong thư mục multiclass_model/


5. Evaluation
5.1. Logistic Regression baseline (binary)
Run: python src/4.1_evaBIN.py
Output: binary_model/primary_mbti_clf.joblib + confusion matrix + report JSON

5.2. XGBoost multiclass evaluation
Run: python src/4.2_evaXGB.py
Output: multiclass_model/confusion_matrix_test.png + classification report


6. Demo Inference
Mở notebook và chạy các cell để mở UI qua Gradio 