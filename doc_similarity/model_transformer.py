import os
import torch
import numpy as np
from tqdm import tqdm
from numpy import save, load
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from doc_similarity.utils import get_recommendation

project_path = os.getcwd()
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

class BertModel:
    def __init__(self, model_name=MODEL_NAME, batch_size=4):
        """ init model attributes """
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def embed(self, data):
        """ Create the embedded matrice from original sentences """
        nb_batchs = 1 if (len(data) < self.batch_size) else len(
            data) // self.batch_size
        batchs = np.array_split(data, nb_batchs)
        mean_pooled = []
        for batch in tqdm(batchs, total=len(batchs), desc='Training...'):
            mean_pooled.append(self.transform(batch))
        mean_pooled_tensor = torch.tensor(
            len(data), dtype=float)
        mean_pooled = torch.cat(mean_pooled, out=mean_pooled_tensor)
        self.embed_mat = mean_pooled

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def transform(self, data):
        data = list(data)
        token_dict = self.tokenizer(
            data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        with torch.no_grad():
            token_embed = self.model(**token_dict)
        attention_mask = token_dict['attention_mask']
        # average pooling of masked embeddings
        mean_pooled = self.mean_pooling(
            token_embed, attention_mask)
        return mean_pooled

    def save(self, file_path):
        save(file_path, self.embed_mat)

    def load(self, file_path):
        self.embed_mat = load(file_path)

    def recommend(self, news_arr, user_input, top_k=10):
        user_input = self.transform(user_input)
        cosine_similarities = cosine_similarity(user_input, self.embed_mat)
        return get_recommendation(news_arr, cosine_similarities, top_k)
