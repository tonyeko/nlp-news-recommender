import os
from doc_similarity.model_transformer_embed import BertModel
from doc_similarity.model_tfidf import tfidf_recommendation

project_path = os.getcwd()

def recommend(news_keywords_arr):
  # bert_model = BertModel()
  # bert_model.load(f'{project_path}/saved_models/bert_embed_mat.npy')
  # bert_model.recommend(news_keywords_arr, 3)
  tfidf_recommendation(news_keywords_arr)