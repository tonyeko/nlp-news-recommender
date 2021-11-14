import nltk
import numpy as np
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from doc_similarity.utils import print_recomendation

nltk.download('punkt')
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def transformer_recommendation(news_arr):
  model = SentenceTransformer(MODEL_NAME)
  documents = list(news_arr)

  vectors = []
  for i, document in enumerate(documents):

    sentences = sent_tokenize(document)
    embeddings_sentences = model.encode(sentences)
    embeddings = np.mean(np.array(embeddings_sentences), axis=0)

    vectors.append(embeddings)

    if i % 100 == 0:
      print("making vector at index:", i)

  cosine_similarities = cosine_similarity(vectors, vectors)

  print_recomendation(news_arr, 3, cosine_similarities)