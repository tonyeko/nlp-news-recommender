import os
from doc_similarity.model_transformer import BertModel
from doc_similarity.model_tfidf import tfidf_recommendation
import pandas as pd

project_path = os.getcwd()


def recommend(user_input, news_keywords_arr):
    model = BertModel()
    model.load(f'{project_path}/../../saved_models/bert_embed_mat.npy')
    return model.recommend(news_keywords_arr, user_input)
