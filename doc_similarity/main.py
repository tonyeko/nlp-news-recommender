import os
from doc_similarity.model import BertModel

project_path = os.getcwd()

def recommend(news_keywords_arr):
  bert_model = BertModel()
  bert_model.load(f'{project_path}/saved_models/bert_embed_mat.npy')
  bert_model.recommend(news_keywords_arr, 3)