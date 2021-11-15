import os
import pandas as pd
from doc_similarity.main import recommend

project_path = os.getcwd()
data = pd.read_csv(f'{project_path}/datasets/bbc-text-keywords.csv')
recommendation_result = recommend(data['keywords'].values[3], data['keywords'].values, "topic")[0]
print(recommendation_result["score"])
print(recommendation_result["text"])
