
import pandas as pd
import keras
import numpy as np
import keras.utils
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model


def model_keras(df_test, x_train_matrix, x_test_matrix, y_train, y_train_enc, y_train_oh, tokenizer, teks_pipeline):
    # Build the model
    number_class = np.max(y_train_enc) + 1
    layers = keras.layers
    models = keras.models

    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(1000,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(number_class))
    model.add(layers.Activation('softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Train the model
    result = model.fit(x_train_matrix, y_train_oh, batch_size=30,
                       epochs=2, verbose=1, validation_split=0.1)

    # Melihat model yang dibantuk
    # plot_model(model, show_shapes=True, show_layer_names=True)

    # Evaluate score of keras model
    score = model.evaluate(x_train_matrix, y_train_oh,
                           batch_size=30, verbose=1)
    print('Test accuracy score:', score[1])
    print('Test loss score:', score[0])

    # Try to predict
    rows = []
    encoder = LabelEncoder().fit(y_train)
    for i in range(10):
        x_input = model.predict(np.array([x_test_matrix[i]]))
        y_predict = encoder.classes_[np.argmax(x_input)]
        rows.append([df_test["text"].iloc[i], y_predict,
                     df_test["category"].iloc[i]])

    # print table result of prediction
    tabel_result = pd.DataFrame(
        rows, columns=["Input", "Prediction", "Actual"])

    teks_pipeline_matrix = model.predict(
        tokenizer.texts_to_matrix(teks_pipeline))
    teks_pipeline_matrix.shape

    y_predict = []
    for i in range(len(teks_pipeline_matrix)):
        y_predict.append(encoder.classes_[np.argmax(teks_pipeline_matrix[i])])
    return y_predict


def train_model(news):
    df = pd.read_csv("../datasets/preprocess-bbc-text.csv")

    # Split dataset
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=False)
    y_train = df_train["category"]
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
    return result
