import numpy as np
from gensim.models.word2vec import Word2Vec


def print_recomendation_w2v(news_m, cosine_m, top_k=10, verbose=False):
    top_k_indices = np.argsort(cosine_m)[::-1][:top_k]
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


def get_similarity_matrix(user_input, dataset, model):
    user_input = user_input.split()
    in_vocab_list = []
    for w in user_input:
        if w in model.wv.key_to_index.keys():
            in_vocab_list.append(w)
    # Retrieve the similarity between two words as a distance
    if len(in_vocab_list) > 0:
        sim_mat = np.zeros(len(dataset))
        for i, data_sentence in enumerate(dataset):
            if data_sentence:
                sim_sentence = model.wv.n_similarity(
                    in_vocab_list, data_sentence)
            else:
                sim_sentence = 0
            sim_mat[i] = np.array(sim_sentence)
    return sim_mat


def word2vec_recommendation(news_arr, user_input, top_k=10):
    word2vec_model = Word2Vec(min_count=0, workers=8)
    word2vec_model.build_vocab(news_arr)
    cosine_similarities = get_similarity_matrix(
        user_input, news_arr, word2vec_model)
    return print_recomendation_w2v(news_arr, cosine_similarities)
