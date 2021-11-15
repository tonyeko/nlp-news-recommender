# some_file.py
import os
import sys
import pandas as pd
# insert at 1, 0 is the script path (or '' in REPL)
project_path = os.getcwd()

sys.path.insert(0, project_path + '/../../')
sys.path.insert(0, project_path + '/../../topic_classification')
sys.path.insert(0, project_path + '/../../keyword_extraction')


from topic_classification.main import main_classification
from keyword_extraction.utils import get_text_keywords
from doc_similarity.main import recommend

data = pd.read_csv(f'{project_path}/../../datasets/bbc-text-keywords.csv')

def mainProgram(berita):
    hasil = []
    hasil.append(berita)
    topic = main_classification(hasil)
    return topic

def extractKeywords(berita, num_of_keywords=0):
    keywords = get_text_keywords(berita, num_of_keywords)
    return keywords

def documentSimilarity(user_input, predicted_topic):
    return recommend(user_input, data['keywords'].values, predicted_topic)

