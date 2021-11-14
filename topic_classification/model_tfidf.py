
"""## TFIDF dan Shalow Algortima"""

#import library
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import model_selection, svm, tree
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk


def tfidf(df, x_train, x_test, y_train, y_test, tokenizer, teks_pipeline):
    # # Menggunakan algoritma shalow machine untuk training model and predict

    # # Decision Tree
    # model_dtl = Pipeline([('tfidf', TfidfVectorizer()),
    #                       ('dt', DecisionTreeClassifier()), ])
    # y_predict_dtl = model_dtl.fit(x_train, y_train).predict(x_test)
    # print("-------------------------------------------------------------\nUsing Decision Tree Learning")
    # print(classification_report(y_predict_dtl, y_test,
    #                             target_names=df['category'].unique()))

    # # XGBoost
    # model_xgb = Pipeline([('tfidf', TfidfVectorizer()),
    #                       ('xgb', XGBClassifier(n_estimators=100)), ])
    # y_predict_xgb = model_xgb.fit(x_train, y_train).predict(x_test)
    # print("\n-------------------------------------------------------------\nUsing XGboost")
    # print(classification_report(y_predict_xgb, y_test,
    #                             target_names=df['category'].unique()))

    # # SVM
    # model_svm = Pipeline([('tfidf', TfidfVectorizer()),
    #                       ('svm', svm.SVC(kernel='linear', C=0.1, degree=3)), ])
    # y_predict_svm = model_svm.fit(x_train, y_train).predict(x_test)
    # print("\n-------------------------------------------------------------\nUsing SVM")
    # print(classification_report(y_predict_svm, y_test,
    #                             target_names=df['category'].unique()))

    # Multinomial Naive Bayes
    model_mnb = Pipeline(
        [('tfidf', TfidfVectorizer()), ('MNB', MultinomialNB()), ])
    y_predict_mnb = model_mnb.fit(x_train, y_train).predict(x_test)
    # print("\n-------------------------------------------------------------\nUsing Multinomial Naive Bayes")
    # print(classification_report(y_predict_mnb, y_test,
    #                             target_names=df['category'].unique()))

    # # Random Forrest Classifier
    # model_rfc = Pipeline([('tfidf', TfidfVectorizer()),
    #                       ('rfc', RandomForestClassifier(n_estimators=100))])
    # y_predict_rfc = model_rfc.fit(x_train, y_train).predict(x_test)
    # print("\n-------------------------------------------------------------\nUsing Random Forrest Classifier")
    # print(classification_report(y_predict_rfc, y_test,
    #                             target_names=df['category'].unique()))

    df2 = []
    for index, text in enumerate(teks_pipeline):
        df2.append(str(nltk.word_tokenize(text)))

    y_predict = model_mnb.predict(df2)
    return y_predict
