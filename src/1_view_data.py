import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
# Load dataset
df = pd.read_csv(config.FINAL_DATA_PATH)

# 1. Phân phối nhãn
plt.figure(figsize=(10,6))
sns.countplot(y="type", data=df, order=df['type'].value_counts().index)
plt.title("Phân phối MBTI types")
plt.savefig("eda_label_distribution(clean+balanced).png")
plt.close()

# 2. Độ dài văn bản
df['text_len'] = df['posts'].apply(len)
plt.figure(figsize=(8,6))
plt.hist(df['text_len'], bins=50, color="skyblue")
plt.title("Phân phối độ dài văn bản")
plt.xlabel("Số ký tự")
plt.ylabel("Số mẫu")
plt.savefig("eda_text_length(clean+balanced).png")
plt.close()
