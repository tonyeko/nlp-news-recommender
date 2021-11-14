"""## Word2Vec

"""

# using gensim data
import gensim
import gensim.downloader as gensim_api
import sklearn
import numpy as np
import tensorflow
from sklearn.metrics import classification_report
from tensorflow.keras import models, layers, preprocessing as kprocessing
import gensim


def word2vec(df_train, df_test, y_train_oh):
    gen = gensim_api.load("word2vec-google-news-300")
    # membuat list unigram
    unigram_corpus = []
    for sentence in df_train["text"]:
        uniword = sentence.split()
        unigram = [" ".join(uniword[i:i+1]) for i in range(0, len(uniword), 1)]
    unigram_corpus.append(unigram)

    # Mendeteksi unigram
    bigram = gensim.models.phrases.Phrases(
        unigram_corpus, min_count=5, threshold=10)
    bigram = gensim.models.phrases.Phraser(bigram)
    unigram_corpus = list(bigram[unigram_corpus])

    # Mendeteksi trigram
    trigram = gensim.models.phrases.Phrases(
        bigram[unigram_corpus], min_count=5, threshold=10)
    trigram = gensim.models.phrases.Phraser(trigram)
    unigram_corpus = list(trigram[unigram_corpus])

    # transorm menjadi vector
    gen = gensim.models.word2vec.Word2Vec(
        unigram_corpus, vector_size=100, window=5, min_count=1, sg=1)

    # Feature engineering
    # Melakukan preprocess korpus dari list unigram yang suda diproses menjadi word2vec menjadi list sequence

    tokenize = kprocessing.text.Tokenizer(
        lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenize.fit_on_texts(unigram_corpus)
    vocabulary = tokenize.word_index

    # membuat sekuense
    unigram_text_to_seq = tokenize.texts_to_sequences(unigram_corpus)

    # melakukan padding untuk sequnece
    x_train_w2v = kprocessing.sequence.pad_sequences(
        unigram_text_to_seq, maxlen=15, padding="post", truncating="post")

    # Test data tokenize
    unigram_corpus_test = []
    for sentence in df_test["text"]:
        uniword_test = sentence.split()
        unigram_test = [" ".join(uniword_test[i:i+1])
                        for i in range(0, len(uniword_test), 1)]
    unigram_corpus_test.append(unigram_test)

    unigram_corpus_test = list(bigram[unigram_corpus_test])
    unigram_corpus_test = list(trigram[unigram_corpus_test])

    unigram_text_to_seq_test = tokenize.texts_to_sequences(unigram_corpus_test)
    x_test_w2v = kprocessing.sequence.pad_sequences(
        unigram_text_to_seq_test, maxlen=15, padding="post", truncating="post")

    # membuat matriks embeding
    matriks_embedding = np.zeros((len(vocabulary)+1, 300))
    for kata, i in vocabulary.items():
        try:
            matriks_embedding[i] = gen[kata]
        except:
            pass

    # Membuat deeplearning modelnya

    # Attention layer : untuk mengcapture bobot dari tiap instance
    # 2 layer menggunakan bidirectional LSTM
    # 2 dense layer untuk memprefiksi probabilitas dari berita di tiap kategori

    def layer_attent(input, neuron):
        layer = layers.Permute((2, 1))(input)
        layer = layers.Dense(neuron, activation="softmax")(layer)
        layer = layers.Permute((2, 1), name="attention")(layer)
        layer = layers.multiply([input, layer])
        return layer

    # layer untuk input
    layer_input = layers.Input(shape=(15,))

    # layer untuk embedding
    layer_embedding = layers.Embedding(input_dim=matriks_embedding.shape[0],  output_dim=matriks_embedding.shape[1], weights=[
                                       matriks_embedding], input_length=15, trainable=False)(layer_input)

    # layer attention
    layer_attentional = layer_attent(layer_embedding, neuron=15)

    # Layer untuk bidirectional lstm
    layer_lstm = layers.Bidirectional(layers.LSTM(
        units=15, dropout=0.2, return_sequences=True))(layer_attentional)
    layer_lstm = layers.Bidirectional(
        layers.LSTM(units=15, dropout=0.2))(layer_lstm)

    # Layer Dense
    layer_dense = layers.Dense(64, activation='relu')(layer_lstm)
    layer_output = layers.Dense(5, activation='softmax')(layer_dense)

    # compile
    model_w2v = models.Model(layer_input, layer_output)
    model_w2v.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    dic_y_mapping = {label: n for n, label in
                     enumerate(np.unique(df_train['category']))}

    y_train_w2v = np.array([dic_y_mapping[y] for y in df_train['category']])

    trained = model_w2v.fit(x=x_train_w2v, y=y_train_oh, batch_size=256,
                            epochs=20, shuffle=True,
                            validation_split=0.3)

    # Test model

    predicted_prob = model_w2v.predict(x_test_w2v)
    predicted = [round(pred[0]) for pred in predicted_prob]

    y_testz = np.array([dic_y_mapping[y] for y in df_test['category']])

    print(classification_report(y_testz, predicted,
                                target_names=df_test['category'].unique()))
