import transformers
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, BertConfig
from sklearn.model_selection import train_test_split

def load_bert(x_test_bert_id, x_test_bert_mask, model_class, tokenizer_class, output_dir, y_train, news):
    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(output_dir)
    tokenizer_bert = BertTokenizerFast.from_pretrained(output_dir)
    
    model_distill = BertForSequenceClassification.from_pretrained(
            model,
            output_attentions=False,
            output_hidden_states=False,
            num_labels=5
        )
    
    # get predictions for test data
    with torch.no_grad():
        preds_bert_distill = model_distill(x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_bert_distill = preds_bert_distill.logits.detach().cpu().numpy()

    # model's performance
    preds_bert_distill = np.argmax(preds_bert_distill, axis = 1)

    encoder = LabelEncoder().fit(y_train)
    y_predict = []
    for i in range(len(preds_bert_distill)):
        y_predict.append(encoder.classes_[np.argmax(preds_bert_distill[i])])

    y_predict