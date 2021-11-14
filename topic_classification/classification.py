
import pandas as pd
import keras
import numpy as np
import keras.utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from model_keras import model_keras
from model_tfidf import tfidf
from model_word2vec import word2vec
from model_bert import bert
import nltk


def classify_topic(news):
    df = pd.read_csv("../datasets/preprocess-bbc-text.csv")

    # Split dataset
    for index, text in enumerate(df["text"]):
        df.at[index, "tokenized_text"] = str(nltk.word_tokenize(text))

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=False)

    x_train = df_train["tokenized_text"]
    y_train = df_train["category"]
    x_test = df_test["tokenized_text"]
    y_test = df_test["category"]

    # Tokenize data and create into matrix
    tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=1000, char_level=False)
    tokenizer.fit_on_texts(df_train["text"])
    x_train_matrix = tokenizer.texts_to_matrix(df_train["text"])
    x_test_matrix = tokenizer.texts_to_matrix(df_test["text"])

    # Encoding
    # Convert label strings to numbered index
    encoder = LabelEncoder().fit(y_train)
    y_train_enc = encoder.transform(y_train)
    y_test_enc = encoder.transform(y_test)

    # Converts the labels to a one-hot representation
    number_class = np.max(y_train_enc) + 1
    y_train_oh = keras.utils.to_categorical(y_train_enc, number_class)
    y_test_oh = keras.utils.to_categorical(y_test_enc, number_class)

    result = model_keras(df_test, x_train_matrix, x_test_matrix, y_train,
                         y_train_enc, y_train_oh,  tokenizer, news)

    # result = tfidf(df, x_train, x_test, y_train, y_test, tokenizer, news)

    # word2vec(df_train, df_test, y_train_oh)

    # result = bert(df, df_train, df_test)

    return result
