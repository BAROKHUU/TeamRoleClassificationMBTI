import pandas as pd
from nltk.corpus import wordnet
import random
import nltk

nltk.download('punkt_tab')
nltk.download('wordnet')
import config
# ---- 1. Load dữ liệu ----
df = pd.read_csv(config.CLEAN_DATA_PATH)

# ---- 2. Định nghĩa hàm augment ----
def synonym_replacement(sentence, n=1):
    """Thay thế ngẫu nhiên n từ bằng từ đồng nghĩa"""
    words = nltk.word_tokenize(sentence)
    new_words = words.copy()
    random_word_list = list(set([w for w in words if len(wordnet.synsets(w)) > 0]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if w == random_word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def rewrite_post(post, num_augments):
    """Tạo ra num_augments câu mới từ post"""
    sentences = nltk.sent_tokenize(post)
    new_posts = []
    for _ in range(num_augments):
        new_sentences = []
        for sent in sentences:
            new_sentences.append(synonym_replacement(sent, n=1))
        new_posts.append(' '.join(new_sentences))
    return new_posts  # trả list câu augment

# ---- 3. Xác định số lần augment cho từng nhãn ----

augment_1_time = config.AUGMENT_1_TIME
augment_4_times = config.AUGMENT_4_TIME
augmented_rows = []

# ---- 4. Loop qua dữ liệu ----
for idx, row in df.iterrows():
    t = row['type']
    post = row['posts']

    # Tạo số lần augment theo nhãn
    if t in augment_4_times:
        num_augments = 4
    elif t in augment_1_time:
        num_augments = 1
    else:
        num_augments = 0

    # Nếu cần augment
    if num_augments > 0:
        new_posts = rewrite_post(post, num_augments)
        for np_ in new_posts:
            augmented_rows.append({'type': t, 'posts': np_})

# ---- 5. Tạo DataFrame mới gồm dữ liệu cũ + augment ----
aug_df = pd.DataFrame(augmented_rows)
final_df = pd.concat([df, aug_df], ignore_index=True)

# ---- 6. Xuất ra file mới ----
final_df.to_csv(config.FINAL_DATA_PATH, index=False)

print("Số lượng dữ liệu sau augment:")
print(final_df['type'].value_counts())
