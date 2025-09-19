import pandas as pd
import re
import config
# Read the original file
df = pd.read_csv(config.RAW_DATA_PATH)

# Delete links from file function
def remove_links(text):
    if pd.isna(text):
        return text
    # Regex xoá link http, https, www, và tên miền phổ biến
    return re.sub(r"http\S+|www\.\S+|\S+\.(com|org|net|io|co|us|uk)\S*", "", text)

# Apply the function to the 'posts' column
df["posts"] = df["posts"].apply(remove_links)

# Save cleaned file
df.to_csv(config.CLEAN_DATA_PATH, index=False)

