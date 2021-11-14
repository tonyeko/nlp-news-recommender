import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder


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
                       epochs=30, verbose=1, validation_split=0.1)

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
