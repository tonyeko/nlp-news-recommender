import pandas as pd
import torch
import numpy as np
import transformers
import torch
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AdamW, BertConfig


def main_bert(news):
 # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model_distill = BertForSequenceClassification.from_pretrained(
            "../saved_models/topic_classification_model")

        tokenizer_bert = BertTokenizerFast.from_pretrained(
            "../saved_models/topic_classification_model")

    except:
        model_distill = BertForSequenceClassification.from_pretrained(
            "../../saved_models/topic_classification_model")

        tokenizer_bert = BertTokenizerFast.from_pretrained(
            "../../saved_models/topic_classification_model")

    x_test_bert_tokenize = tokenizer_bert.batch_encode_plus(
        news, max_length=100, padding=True, truncation=True, return_token_type_ids=False)

    # datatest
    x_test_bert_id = torch.tensor(x_test_bert_tokenize['input_ids'])
    x_test_bert_mask = torch.tensor(x_test_bert_tokenize['attention_mask'])

    # get predictions for test data
    with torch.no_grad():
        preds_bert_distill = model_distill(
            x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_bert_distill = preds_bert_distill.logits.detach().cpu().numpy()

    # model's performance
    preds_bert_distill = np.argmax(preds_bert_distill, axis=1)
    list_topic = ["business", "entertainment", "politics", "sport", "tech"]
    topic = list_topic[preds_bert_distill[0]]
    y_predict = []
    y_predict.append(topic)
    # y_predict = []
    # for i in range(len(preds_bert_distill)):
    #     y_predict.append(encoder.classes_[np.argmax(preds_bert_distill[i])])
    return y_predict
