import os
from doc_similarity.model_transformer import BertModel
from doc_similarity.model_tfidf import tfidf_recommendation

project_path = os.getcwd()


def recommend(user_input, news_keywords_arr, predicted_topic):
    return tfidf_recommendation(news_keywords_arr, user_input)
