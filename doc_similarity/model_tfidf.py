from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from doc_similarity.utils import print_recomendation

def tfidf_recommendation(news_arr):
  ## analyzer -- to select individual words# default
  ## max_df[0.0,1.0] - used to ignore words with frequency more than 0.8 these words can be useless words as these words may appear only once and may not have a significant meaning
  tf = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.8, ngram_range=(1,3))
  tfidf_matrix = tf.fit_transform(news_arr)
  cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
  print_recomendation(news_arr, 3, cosine_similarities)