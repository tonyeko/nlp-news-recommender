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


def extract_best_indices(cos_sim, top_k):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    top_k (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    index = np.argsort(cos_sim)[::-1]  # from highest idx to smallest score
    mask = np.logical_or(cos_sim[index] != 0, np.ones(
        len(cos_sim)))  # eliminate 0 cosine distance
    best_index = index[mask]
    return best_index if top_k != -1 else best_index[:top_k]


def get_recommendation(news_m, cosine_m, top_k=-1, verbose=False):
    if len(cosine_m.shape) > 1:
        cosine_m = np.mean(cosine_m, axis=0)
    top_k_indices = extract_best_indices(cosine_m, top_k)
    result = []
    for i in range(len(top_k_indices)):
        best_idx = top_k_indices[i]
        result.append(
            {"idx": best_idx, "score": cosine_m[best_idx], "text": news_m[best_idx]})
        if verbose:
            print(
                f"Recomendation {i+1}: (IDX: {best_idx}), score: {cosine_m[best_idx]} | {news_m[best_idx][:50]}...")
            print()
    return result
