#Import libraries
import nltk
import numpy as np
import pandas as pd
from nltk import tokenize
import torch
import tensorflow as tf
from transformers import BertConfig, AutoModelForTokenClassification, BertTokenizer

nltk.download('punkt')

def tokenizeSentences(tokenizer, sentences):
  tokenized_sentences = []
  for sentence in sentences:
    tokenized_sentences.append(tokenizer.tokenize(sentence))
  return tokenized_sentences

def fullyTokenizeSentences(tokenizer, sentences, tokenized_sentences):
  input_ids = []
  attention_masks = []
  max_len = 0
  for tokenized_sentence in tokenized_sentences:
    if(len(tokenized_sentence) > max_len):
      max_len = len(tokenized_sentence)

  for sentence in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sentence,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len+2,           # Pad & truncate all sentences.
                        padding='max_length',
                        truncation = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
    )

    input_ids.append(encoded_dict['input_ids'][0])
    attention_masks.append(encoded_dict['attention_mask'][0])
  
  return input_ids, attention_masks

def get_list_of_keywords(tokenized_sentences, model, input_ids, attention_masks):
  input_ids = torch.stack(input_ids)
  attention_masks = torch.stack(attention_masks)
  prediction = model(input_ids, attention_mask=attention_masks)[0].detach().numpy()
  print("=============================")
  print(prediction.shape)
  print("=============================")
  prediction = np.argmax(prediction, axis=2)
  keywords=[]
  for i in range(len(tokenized_sentences)):
    for j in range(len(tokenized_sentences[i])):
      if(prediction[i][j+1]==1): #+1 karena ada token CLS di awal sentence
        keywords.append(tokenized_sentences[i][j])
  return keywords

def get_text_keywords(text):

  config_class, model_class, tokenizer_class = (BertConfig, AutoModelForTokenClassification, BertTokenizer)
  model = model_class.from_pretrained("saved_model/")
  tokenizer = tokenizer_class.from_pretrained("saved_model/")

  sentences = tokenize.sent_tokenize(text)
  tokenized_sentences = tokenizeSentences(tokenizer, sentences)
  input_ids, attention_masks = fullyTokenizeSentences(tokenizer, sentences, tokenized_sentences)
  list_of_keywords = list(set(get_list_of_keywords(tokenized_sentences, model, input_ids, attention_masks)))
  return list_of_keywords
