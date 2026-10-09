import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import csv
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statistics

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import transformers
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

data = pd.read_csv("korpusi.csv", encoding='unicode_escape')
data.head()

print("Number of tags: {}".format(len(data.Label.unique())))
frequencies = data.Label.value_counts()
# frequencies

tags = {}
for tag, count in zip(frequencies.index, frequencies):
    if tag != "O":
        if tag[2:5] not in tags.keys():
            tags[tag[2:5]] = count
        else:
            tags[tag[2:5]] += count
    continue

print(sorted(tags.items(), key=lambda x: x[1], reverse=True))

data = data.fillna(method='ffill')
data.head()


data['Sentence_ID'] = (data['Token'] == '.').cumsum()  # Assign a unique ID to each sentence

# Group tokens by sentence
data['sentence'] = data.groupby('Sentence_ID')['Token'].transform(lambda x: ' '.join(x))

# Group labels by sentence
data['word_labels'] = data.groupby('Sentence_ID')['Label'].transform(lambda x: ','.join(x))

# # Drop duplicate rows to keep only one row per sentence
# data = data.drop_duplicates(subset=['Sentence_ID'])[['Sentence_ID', 'sentence', 'word_labels']]

# # Reset index and save to a new CSV file
# data = data.reset_index(drop=True)
# data.to_csv('grouped_sentences.csv', index=False)

data.head()

label2id = {k: v for v, k in enumerate(data.Label.unique())}
id2label = {v: k for v, k in enumerate(data.Label.unique())}


data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
data.head()

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.data.sentence[index]
        word_labels = self.data.word_labels[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)

        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, "O") # add outside label for [CLS] token
        labels.insert(-1, "O") # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
          # truncate
          tokenized_sentence = tokenized_sentence[:maxlen]
          labels = labels[:maxlen]
        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
          labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [label2id[label] for label in labels]
        # the following line is deprecated
        #label_ids = [label if label != 0 else -100 for label in label_ids]

        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len







    train_size = 0.8
    train_dataset = data.sample(frac=train_size, random_state=200)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = dataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                       num_labels=len(id2label),
                                                       id2label=id2label,
                                                       label2id=label2id)
    model.to(device)

    ids = training_set[0]["ids"].unsqueeze(0)
    mask = training_set[0]["mask"].unsqueeze(0)
    targets = training_set[0]["targets"].unsqueeze(0)
    ids = ids.to(device)
    mask = mask.to(device)
    targets = targets.to(device)
    outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
    initial_loss = outputs[0]
    initial_loss

    tr_logits = outputs[1]
    tr_logits.shape

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    def train(epoch):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        model.train()

        for idx, batch in enumerate(training_loader):

            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

            # compute training accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_preds.extend(predictions)
            tr_labels.extend(targets)

            tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

        for epoch in range(EPOCHS):
            print(f"Training epoch: {epoch + 1}")
            train(epoch)

            def valid(model, testing_loader):
                # put model in evaluation mode
                model.eval()

                eval_loss, eval_accuracy = 0, 0
                nb_eval_examples, nb_eval_steps = 0, 0
                eval_preds, eval_labels = [], []

                with torch.no_grad():
                    for idx, batch in enumerate(testing_loader):

                        ids = batch['ids'].to(device, dtype=torch.long)
                        mask = batch['mask'].to(device, dtype=torch.long)
                        targets = batch['targets'].to(device, dtype=torch.long)

                        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
                        loss, eval_logits = outputs.loss, outputs.logits

                        eval_loss += loss.item()

                        nb_eval_steps += 1
                        nb_eval_examples += targets.size(0)

                        if idx % 100 == 0:
                            loss_step = eval_loss / nb_eval_steps
                            print(f"Validation loss per 100 evaluation steps: {loss_step}")

                        # compute evaluation accuracy
                        flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
                        active_logits = eval_logits.view(-1,
                                                         model.num_labels)  # shape (batch_size * seq_len, num_labels)
                        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)
                        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
                        active_accuracy = mask.view(-1) == 1  # active accuracy is also of shape (batch_size * seq_len,)
                        targets = torch.masked_select(flattened_targets, active_accuracy)
                        predictions = torch.masked_select(flattened_predictions, active_accuracy)

                        eval_labels.extend(targets)
                        eval_preds.extend(predictions)

                        tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
                        eval_accuracy += tmp_eval_accuracy

                # print(eval_labels)
                # print(eval_preds)

                labels = [id2label[id.item()] for id in eval_labels]
                predictions = [id2label[id.item()] for id in eval_preds]

                # print(labels)
                # print(predictions)

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_steps
                print(f"Validation Loss: {eval_loss}")
                print(f"Validation Accuracy: {eval_accuracy}")

                return labels, predictions

            labels, predictions = valid(model, testing_loader)

            from seqeval.metrics import classification_report

            print(classification_report([labels], [predictions]))