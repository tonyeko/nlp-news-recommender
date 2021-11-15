# some_file.py
import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '../../topic_classification')
from main import main_classification
sys.path.insert(0, '../../keyword_extraction')
from utils import get_text_keywords
# sys.path.insert(0, '../../doc_similarity')
# from main import recommend

def mainProgram(berita):
    hasil = []
    hasil.append(berita)
    topic = main_classification(hasil)
    return topic

def extractKeywords(berita, num_of_keywords=0):
    keywords = get_text_keywords(berita, num_of_keywords)
    return keywords

