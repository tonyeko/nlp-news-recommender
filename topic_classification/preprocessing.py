import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.text import text_to_word_sequence
import nltk
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt')


def get_tag(kata):
    # Melakukan mapping post tag untuk first character
    tag_kata = nltk.pos_tag([kata])[0][1][0].upper()
    dictionary = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                  "V": wordnet.VERB, "R": wordnet.ADV}
    return dictionary.get(tag_kata, wordnet.NOUN)


def preprocess(text, stopwords=set(stopwords.words("english"))):
    # Melakukan preprocessing terhadap text data
    # stopwords = set(stopwords.words("english"))
    text = str(text)
    tokenized_text = text_to_word_sequence(
        text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")

    # Melakukan penghapusan stopword yang ada di corpus
    text_nonstopword = []
    for idx, word in enumerate(tokenized_text):
        if not tokenized_text[idx] in stopwords:
            text_nonstopword.append(tokenized_text[idx])
    text_nonstopword_join = " ".join(text_nonstopword)

    # menghapus number
    text_nonnumber = "".join(
        i for i in text_nonstopword_join if not i.isdigit())

    # melakukan stemming
    stem = PorterStemmer()
    input_stem = nltk.word_tokenize(text_nonnumber)
    text_stem = ' '.join([stem.stem(word) for word in input_stem])

    # melakukan lematisasi
    lem = WordNetLemmatizer()
    input_lem = nltk.word_tokenize(text_stem)
    text_lem = ' '.join([lem.lemmatize(word, get_tag(word))
                         for word in input_lem])
    return text_lem


def preprocessing_data(df):
    """## Check dataset"""
    df["text"] = df["text"].apply(preprocess)
    return df


df = pd.read_csv("../datasets/bbc-text.csv")
print(df)
df = preprocessing_data(df)
print(df)
df.to_csv("../datasets/preprocess-bbc-text.csv")
