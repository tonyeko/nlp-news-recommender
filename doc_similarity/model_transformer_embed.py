import os
import torch
import numpy as np
from tqdm import tqdm
from numpy import save, load
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
from doc_similarity.utils import print_recomendation

project_path = os.getcwd()
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

class BertModel:
    def __init__(self, model_name=MODEL_NAME, batch_size=4):
        """ init model attributes """
        self.model_name = model_name
        self.device = "cpu"
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.pipeline = pipeline(
            'feature-extraction', model=self.model, tokenizer=self.tokenizer, device=-1)

    def embed(self, data):
        """ Create the embedded matrice from original sentences """
        nb_batchs = 1 if (len(data) < self.batch_size) else len(
            data) // self.batch_size
        batchs = np.array_split(data, nb_batchs)
        mean_pooled = []
        for batch in tqdm(batchs, total=len(batchs), desc='Training...'):
            mean_pooled.append(self.transform(batch))
        mean_pooled_tensor = torch.tensor(
            len(data), dtype=float).to(self.device)
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
        # send all values to device by calling v.to(device)
        token_dict = {k: v.to(self.device) for k, v in token_dict.items()}
        with torch.no_grad():
            token_embed = self.model(**token_dict)
        attention_mask = token_dict['attention_mask']
        # average pooling of masked embeddings
        mean_pooled = self.mean_pooling(
            token_embed, attention_mask)
        mean_pooled = mean_pooled.to(self.device)
        return mean_pooled

    def save(self, file_path):
        save(file_path, self.embed_mat)

    def load(self, file_path):
        self.embed_mat = load(file_path)

    def recommend(self, news_m, user_read_idx, top_k=10):
        cosine_similarities = cosine_similarity(self.embed_mat, self.embed_mat)
        print_recomendation(news_m, user_read_idx, cosine_similarities, top_k)
