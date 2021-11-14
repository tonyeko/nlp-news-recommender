import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def remove_non_ascii(sentence):
    return "".join(i for i in sentence if ord(i) < 128)


def make_lower_case(text):
    return text.lower()


def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    texts = [w for w in text if w.isalpha()]
    texts = " ".join(texts)
    return texts


def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text


def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def extract_best_indices(m, top_k):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    top_k (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0)
    else:
        cos_sim = m
    index = np.argsort(cos_sim)[::-1]  # from highest idx to smallest score
    mask = np.logical_or(cos_sim[index] != 0, np.ones(
        len(cos_sim)))  # eliminate 0 cosine distance
    best_index = index[mask][:top_k]
    return best_index


def print_recomendation(news_m, user_read_idx, cosine_m, top_k=10):
    # get similarity values with other articles
    top_k_indices = extract_best_indices(cosine_m[user_read_idx], top_k)

    print(f"Article Read: {news_m[user_read_idx][:50]}...")
    print(" ---------------------------------------------------------- ")
    for i in range(len(top_k_indices)):
        print(
            f"Recomendation {i+1}: (IDX: {top_k_indices[i]}), score: {cosine_m[top_k_indices[i]][user_read_idx]} | {news_m[top_k_indices[i]][:50]}...")
        print()
