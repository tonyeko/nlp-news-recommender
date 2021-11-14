from classification import classify_topic
import pandas as pd

# Input berupa list of news
# Ouput berupa list of topic


def main_classification(news):
    list_topic = classify_topic(news)
    print(list_topic)
    return list_topic
