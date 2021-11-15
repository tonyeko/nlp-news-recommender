import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from doc_similarity.utils import get_recommendation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')

MIN_WORDS = 4
MAX_WORDS = 200
STOPWORDS = set(stopwords.words('english'))

def tokenizer(sentence):
  return [w for w in word_tokenize(sentence)]


def tfidf_recommendation(news_arr, user_input, top_k=10):
    token_stop = tokenizer(' '.join(STOPWORDS))
    tf = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    tfidf_matrix = tf.fit_transform(news_arr)
    tokens = [str(tok) for tok in tokenizer(user_input)]
    user_matrix = tf.transform(tokens)
    cosine_similarities = cosine_similarity(user_matrix, tfidf_matrix)
    return get_recommendation(news_arr, cosine_similarities, top_k, True)
