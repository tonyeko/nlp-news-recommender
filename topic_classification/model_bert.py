
"""## Fine-Tuning BERT

### Proses BERT
"""

# Import library
import pandas as pd
import numpy as np
import datetime
import time
import tensorflow as tf
import transformers
import torch
from transformers import AutoModel, BertTokenizerFast
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from sklearn.metrics import classification_report
torch.cuda.empty_cache()


def bert(df, df_train, df_test):
    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    # Split dataset

    df_train2, df_val2 = train_test_split(
        df_train, test_size=0.2, random_state=False)

    x_train2 = df_train2["text"]
    y_train2 = df_train2["category"]
    x_val2 = df_val2["text"]
    y_val2 = df_val2["category"]
    x_test2 = df_test["text"]
    y_test2 = df_test["category"]

    # Convert label strings to numbered index
    encoder = LabelEncoder().fit(y_train2)
    y_train2_enc = encoder.transform(y_train2)
    y_val2_enc = encoder.transform(y_val2)
    y_test2_enc = encoder.transform(y_test2)

    y_train2_enc

    # import bert model dan bert tokenizer
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')

    x_train_bert_tokenize = tokenizer_bert.batch_encode_plus(x_train2.tolist(
    ), max_length=100, pad_to_max_length=True, truncation=True, return_token_type_ids=False)
    x_val_bert_tokenize = tokenizer_bert.batch_encode_plus(x_val2.tolist(
    ), max_length=100, pad_to_max_length=True, truncation=True, return_token_type_ids=False)
    x_test_bert_tokenize = tokenizer_bert.batch_encode_plus(x_test2.tolist(
    ), max_length=100, pad_to_max_length=True, truncation=True, return_token_type_ids=False)

    x_train_bert_tokenize

    # Mengubah sekuens angka menjadi tensors

    # data training
    x_train_bert_id = torch.tensor(x_train_bert_tokenize['input_ids'])
    x_train_bert_mask = torch.tensor(x_train_bert_tokenize['attention_mask'])
    y_train_bert = torch.tensor(y_train2_enc.tolist())

    # datatest
    x_val_bert_id = torch.tensor(x_val_bert_tokenize['input_ids'])
    x_val_bert_mask = torch.tensor(x_val_bert_tokenize['attention_mask'])
    y_val_bert = torch.tensor(y_val2_enc.tolist())

    # datatest
    x_test_bert_id = torch.tensor(x_test_bert_tokenize['input_ids'])
    x_test_bert_mask = torch.tensor(x_test_bert_tokenize['attention_mask'])
    y_test_bert = torch.tensor(y_test2_enc.tolist())

    x_train_bert_id

    # Membungkus dengan tensor
    x_train_bert = TensorDataset(
        x_train_bert_id, x_train_bert_mask, y_train_bert)
    x_val_bert = TensorDataset(x_val_bert_id, x_val_bert_mask, y_val_bert)

    # Ambil random sampling untuk training
    sample_x = RandomSampler(x_train_bert)
    sample_x_val = RandomSampler(x_val_bert)

    # Melakukan dataLoader untuk training data
    x_train_loader = DataLoader(x_train_bert, sampler=sample_x, batch_size=32)
    x_val_loader = DataLoader(x_val_bert, sampler=sample_x_val, batch_size=32)

    x_train_loader

    def train(model_bert, optimizer_bert, train_dataloader, val_dataloader, epochs=4, print_update_every=40):
        loss_values = []
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer_bert,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        # Melakukan training tiap epochnya
        for epoch_i in range(0, epochs):
            print(
                '\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Melakuakan Training')
            model_bert.train()

            # melakukan kalkulasi terhadap waktu
            t0 = time.time()
            total_loss = 0

            # Untuk setiap batch dari training data
            for step, batch in enumerate(train_dataloader):

                # Melakukan update untuk setiap masukkan 40 batch
                if step % print_update_every == 0 and not step == 0:

                    # Mengkalkulasi waktu yang dibuang
                    elapsed = change_format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(train_dataloader), elapsed))

                # Setiap batch memiliki pytorch tensor [0]: input ids [1]: attention masks [2]: labels
                batch_input_ids = batch[0].to(device)
                batch_input_mask = batch[1].to(device)
                batch_labels = batch[2].to(device)

                # Menghapus kalkulasi gradien sebelumnya sebelum melakukan backward
                model_bert.zero_grad()

                # Melakukan forward passing
                outputs = model_bert(batch_input_ids,
                                     token_type_ids=None,
                                     attention_mask=batch_input_mask,
                                     labels=batch_labels)

                # Mengambil loss value dari model
                loss = outputs[0]

                # Mengakumulasi training loss untuk semua batch sehingga kita dapat mengkalkulasi rata-rata loss di akhir
                total_loss += loss.item()

                # Melakukan proses backward
                loss.backward()

                # Melakukan klipping menjadi bentuk gradien 1, mencegah exploading gradien
                torch.nn.utils.clip_grad_norm_(model_bert.parameters(), 1.0)

                # Melakukan update parameter dan step dengan komputer gradien
                optimizer_bert.step()

                # Mengupdate learning rate
                scheduler.step()

            # Mengkalkulasi rata-rata training loss dan memasukkannnya ke loss value
            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(
                change_format_time(time.time() - t0)))

            # Melakukan proses validasi
            print("")
            print("Melakukan Validation")

            t0 = time.time()

            # Melakukan evaluasi terhadap model
            model_bert.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluasi data dari tiap batch dalam satu epoch
            for batch in val_dataloader:

                # Memasukkan batch ke GPU
                batch = tuple(t.to(device) for t in batch)

                # mengunpack data loader [0] imput ids [1] attention mask [2] label
                batch_input_ids, batch_input_mask, batch_labels = batch

                with torch.no_grad():
                    # Melakukan forward passing dan predic logits
                    outputs = model_bert(batch_input_ids,
                                         token_type_ids=None,
                                         attention_mask=batch_input_mask)

                # Mengambil nilai logits dari output
                logits = outputs[0]

                # Mrmindahkan logits dan label ids ke cpu
                logits = logits.detach().cpu().numpy()
                label_ids = batch_labels.to('cpu').numpy()

                # Menghitung kalkulasi final, total kalkulasi akuras
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy

                # Menghitung nilai batch
                nb_eval_steps += 1

            # Kurasi akhir dari proses calidasi ini
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            print("  Validation took: {:}".format(
                change_format_time(time.time() - t0)))

    # Menghubah waktu dalam detik menjadi jam:menit:detik
    def change_format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    # Mengkalkulasi akurasi dari prediction vs label kita
    def flat_accuracy(preds, labels):
        prediction_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(prediction_flat == labels_flat) / len(labels_flat)

    # Melakukan print terhadap parameter dari model yang dimliki
    def parameter_model(model):
        params = list(model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(
            len(params)))

        # Parameter 0 hingga 4 untuk embeding layer
        print("\n-------------------------------------------------------------")
        print("Layer Embedding\n")
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        # Parameter 5 hingga 20 untuk first transformer
        print("\n-------------------------------------------------------------")
        print("Transformer Pertama\n")
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        # Parameter sisanya untuk output
        print("\n-------------------------------------------------------------")
        print("Layer Output\n")
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    """### BERT-Base Multilingual"""

    model_multilingual = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased',
        output_attentions=False,
        output_hidden_states=False,
        num_labels=5
    )

    if device.type == 'cuda':
        model_multilingual = model_multilingual.to(device)

    parameter_model(model_multilingual)

    optimizer_multilingual = AdamW(model_multilingual.parameters(), lr=1e-5)
    optimizer_multilingual

    train(model_multilingual, optimizer_multilingual,
          x_train_loader, x_val_loader)

    

    # get predictions for test data
    with torch.no_grad():
        preds_multilingual = model_multilingual(
            x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_multilingual = preds_multilingual.logits.detach().cpu().numpy()

    # model's performance
    preds_multilingual = np.argmax(preds_multilingual, axis=1)
    print(classification_report(y_test_bert, preds_multilingual,
                                target_names=df['category'].unique()))

    """**Do more training and increase epoch**"""

    train(model_multilingual, optimizer_multilingual,
          x_train_loader, x_val_loader, 7)

    

    # get predictions for test data
    with torch.no_grad():
        preds_multilingual = model_multilingual(x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_multilingual = preds_multilingual.logits.detach().cpu().numpy()

    # model's performance
    preds_multilingual = np.argmax(preds_multilingual, axis=1)
    print(classification_report(y_test_bert, preds_multilingual,
                                target_names=df['category'].unique()))

    """**Do more training once more and increase epoch**"""

    train(model_multilingual, optimizer_multilingual,
          x_train_loader, x_val_loader, 10)

    # get predictions for test data
    with torch.no_grad():
        preds_multilingual = model_multilingual(
            x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_multilingual = preds_multilingual.logits.detach().cpu().numpy()

    # model's performance
    preds_multilingual = np.argmax(preds_multilingual, axis=1)
    print(classification_report(y_test_bert, preds_multilingual,
                                target_names=df['category'].unique()))

    train(model_multilingual, optimizer_multilingual,
          x_train_loader, x_val_loader, 15)


    # get predictions for test data
    with torch.no_grad():
        preds_multilingual = model_multilingual(
            x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_multilingual = preds_multilingual.logits.detach().cpu().numpy()

    # model's performance
    preds_multilingual = np.argmax(preds_multilingual, axis=1)
    print(classification_report(y_test_bert, preds_multilingual,
                                target_names=df['category'].unique()))

    # Save the model
    model_multilingual.save_pretrained('saved_model/model_multilingual_bert')

    """### BERT-large uncased"""

    # distillBERT base uncased
    model_bert_large = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased",
        output_attentions=False,
        output_hidden_states=False,
        num_labels=5
    )

    if device.type == 'cuda':
        # push the model to GPU
        model_bert_large = model_bert_large.to(device)

    optimizer_bert_large = AdamW(model_bert_large.parameters(), lr=1e-5)
    optimizer_bert_large

    train(model_bert_large, optimizer_bert_large, x_train_loader, x_val_loader)

    # get predictions for test data
    with torch.no_grad():
        preds_bert_large = model_bert_large(
            x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_bert_large = preds_bert_large.logits.detach().cpu().numpy()

    # model's performance
    preds_bert_large = np.argmax(preds_bert_large, axis=1)
    print(classification_report(y_test_bert, preds_bert_large))

    """### DistilBERT"""

    # distillBERT base uncased
    model_distill = BertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        output_attentions=False,
        output_hidden_states=False,
        num_labels=5
    )

    if device.type == 'cuda':
        # push the model to GPU
        model_distill = model_distill.to(device)

    optimizer_distill = AdamW(model_distill.parameters(), lr=1e-5)
    optimizer_distill

    train(model_distill, optimizer_distill, x_train_loader, x_val_loader)

    from sklearn.metrics import classification_report

    # get predictions for test data
    with torch.no_grad():
        preds_bert_distill = model_distill(
            x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_bert_distill = preds_bert_distill.logits.detach().cpu().numpy()

    # model's performance
    preds_bert_distill = np.argmax(preds_bert_distill, axis=1)
    print(classification_report(y_test_bert, preds_bert_distill,
                                target_names=df['category'].unique()))

    """**Do more training and increase epoch**"""

    train(model_distill, optimizer_distill, x_train_loader, x_val_loader, 10)

    # get predictions for test data
    with torch.no_grad():
        preds_bert_distill = model_distill(
            x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_bert_distill = preds_bert_distill.logits.detach().cpu().numpy()

    # model's performance
    preds_bert_distill = np.argmax(preds_bert_distill, axis=1)
    print(classification_report(y_test_bert, preds_bert_distill,
                                target_names=df['category'].unique()))

    """**Do more training and increase epoch**"""

    train(model_distill, optimizer_distill, x_train_loader, x_val_loader, 50)


    # get predictions for test data
    with torch.no_grad():
        preds_bert_distill = model_distill(
            x_test_bert_id.to(device), x_test_bert_mask.to(device))
        preds_bert_distill = preds_bert_distill.logits.detach().cpu().numpy()

    # model's performance
    preds_bert_distill = np.argmax(preds_bert_distill, axis=1)
    print(classification_report(y_test_bert, preds_bert_distill,
                                target_names=df['category'].unique()))

    # Save the model
    model_distill.save_pretrained('saved_model/model_distill_bert')
