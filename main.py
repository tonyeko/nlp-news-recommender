import os
import pandas as pd
from doc_similarity.main import recommend

project_path = os.getcwd()
data = pd.read_csv(f'{project_path}/datasets/bbc-text-keywords.csv')
recommend(data['keywords'].values)