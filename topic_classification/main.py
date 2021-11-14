from model import train_model

# Input berupa list of news
# Ouput berupa list of topic
list_topic = train_model("../datasets/bbc-text-keyword")
print(list_topic)
