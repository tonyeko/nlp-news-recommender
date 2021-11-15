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
  
  return input_ids, attention_masks, max_len

def flat_tokenized_sentences(tokenized_sentences, maxlen):
  flatten_tokenized_sentences = []
  for tokenized_sentence in tokenized_sentences:
    flatten_tokenized_sentences.append("[CLS]")
    for token in tokenized_sentence:
      flatten_tokenized_sentences.append(token)
    flatten_tokenized_sentences.append("[SEP]")
    num_of_padding = maxlen-len(tokenized_sentence)
    for i in range(num_of_padding):
      flatten_tokenized_sentences.append("padding")
  
  return flatten_tokenized_sentences

def get_list_of_keywords(tokenized_sentences, model, input_ids, attention_masks, maxlen, nKeywords):
  input_ids = torch.stack(input_ids)
  attention_masks = torch.stack(attention_masks)
  prediction = model(input_ids, attention_mask=attention_masks)
  prediction = prediction[0].detach().numpy()
  keywords=[]

  flatten_tokenized_sentences = flat_tokenized_sentences(tokenized_sentences, maxlen)

  if(nKeywords==0):
    prediction = np.argmax(prediction, axis=2)
    for i in range(len(tokenized_sentences)):
      for j in range(len(tokenized_sentences[i])):
        if(prediction[i][j+1]==1): #+1 karena ada token CLS di awal sentence
          keywords.append(tokenized_sentences[i][j])
  else:
    flatten_prediction = prediction.reshape(prediction.shape[0]*prediction.shape[1], prediction.shape[2])[:,1]
    indices = (-flatten_prediction).argsort()
    num_of_entries=0
    for idx in indices:
      keyword = flatten_tokenized_sentences[idx]
      if(keyword not in keywords):
        keywords.append(keyword)
        num_of_entries += 1
      if(num_of_entries==nKeywords):
        break

  return keywords

def get_text_keywords(text, num_of_keywords=0):

  config_class, model_class, tokenizer_class = (BertConfig, AutoModelForTokenClassification, BertTokenizer)
  
  try:
    model = model_class.from_pretrained("../saved_models/keyword extraction")
    tokenizer = tokenizer_class.from_pretrained("../saved_models/keyword extraction")
  except: 
    model = model_class.from_pretrained("../../saved_models/keyword extraction")
    tokenizer = tokenizer_class.from_pretrained("../../saved_models/keyword extraction")

  sentences = tokenize.sent_tokenize(text)
  tokenized_sentences = tokenizeSentences(tokenizer, sentences)
  #Check if tokenized_sentences > 500 words cut that tokens (embedding layer max tokens is 512)
  for j in range(len(tokenized_sentences)):
    if(len(tokenized_sentences[j])>500):
      tokenized_sentences[j] = tokenized_sentences[j][:500]
  input_ids, attention_masks, maxlen = fullyTokenizeSentences(tokenizer, sentences, tokenized_sentences)
  list_of_keywords = list(set(get_list_of_keywords(tokenized_sentences, model, input_ids, attention_masks, maxlen, num_of_keywords)))
  return list_of_keywords
