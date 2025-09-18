import pandas as pd
import re
import config
# Đọc file gốc
df = pd.read_csv(config.RAW_DATA_PATH)

# Hàm xoá link trong văn bản
def remove_links(text):
    if pd.isna(text):
        return text
    # Regex xoá link http, https, www, và tên miền phổ biến
    return re.sub(r"http\S+|www\.\S+|\S+\.(com|org|net|io|co|us|uk)\S*", "", text)

# Áp dụng cho cột posts
df["posts"] = df["posts"].apply(remove_links)

# Lưu file sạch
df.to_csv(config.CLEAN_DATA_PATH, index=False)

