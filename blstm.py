from datetime import time

import numpy as np
import os, sys, argparse, random, itertools
from sklearn.metrics import f1_score, accuracy_score
from sklearn_crfsuite.metrics import flat_classification_report

from blstm_utils import *

## read file content
def file_records_to_list(file_path):
	'''read file line blocks and store them in a list that is returned'''
	with open(file_path, "r", encoding='utf-8') as inf:
		content = inf.read()								# read whole file content as string
	sent_lst = content.split('\n\n')						# split per sentence
	sent_lst = [*filter(None, sent_lst)]					# remove empty units
	return sent_lst

## convert NER lines to tuples
def lines_to_tuples(sent_lst):
	sent_tuple_lst, word_lst = [], []
	for s in sent_lst:
		try:
			line_lst = s.split('\n')			  			# get (word, tag) pairs of each line in a list
			line_lst = [*filter(None, line_lst)]			# remove empty units
			word_lst = []
			for l in line_lst:
				w = l.split('\t\t')[0]						# get each word
				t = l.split('\t\t')[1]						# get each tag
				word_lst.append((w, t))						# append tuple in list
			sent_tuple_lst.append(word_lst)					# append sentence in sentence list
		except:
			continue
	return sent_tuple_lst

## data files
data_file = "./korpusi.txt"
# train_file = "./data_final/train.txt"
# test_file = "./data_final/test.txt"
# dev_file = "./data_final/dev.txt"

## path of the saved model
MODEL_PATH = './model0.pth'

predefined_tags = ['O', 'B-PER', 'I-PER', 'B-PRO', 'I-PRO', 'B-ORG', 'I-ORG', 'B-EVENT', 'I-EVENT', 'B-SHESH', 'I-SHESH', 'B-RRUGE', 'I-RRUGE', 'B-VEND_0', 'I-VEND_0', 'B-VEND_1', 'I-VEND_1', 'B-DATE_0', 'I-DATE_0', 'B-DATE_1', 'I-DATE_1']

## read sentences from files to a lists
sent_lst_full = file_records_to_list(data_file)
# sent_lst_train = file_records_to_list(train_file)
# sent_lst_test = file_records_to_list(test_file)
# sent_lst_dev = file_records_to_list(dev_file)

## put all parts together
# sent_lst_full = sent_lst_train + sent_lst_test + sent_lst_dev

## convert sentence strings to list of tuples
sent_tuple_lst = lines_to_tuples(sent_lst_full)

## length of longest sentence
MAX_SENTENCE_LENGTH = max(len(x) for x in sent_tuple_lst)

## collect all words and tags
all_words = [t[0] for sent in sent_tuple_lst for t in sent]
all_tags = [t[1] for sent in sent_tuple_lst for t in sent]

## count unique words and tags
unique_words = list(set(all_words))
unique_tags = list(set(all_tags))

sent_tup_lst = []
for sent_lst in sent_tuple_lst:
    w_lst, t_lst = [], []
    for tup in sent_lst:
        w = tup[0] ; t = tup[1]
        w_lst.append(w) ; t_lst.append(t)
    sent_tup_lst.append((w_lst, t_lst))

##
training_data = [(item[0], item[1]) for item in sent_tup_lst]


word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {START_TAG: 0, STOP_TAG: 1, 'O': 2, 'B-PER': 3, 'I-PER': 4, 'B-PRO': 5, 'I-PRO': 6, 'B-ORG': 7, 'I-ORG': 8, 'B-EVENT': 9, 'I-EVENT': 10, 'B-SHESH': 11, 'I-SHESH': 12, 'B-RRUGE': 13, 'I-RRUGE': 14, 'B-VEND_0': 15, 'I-VEND_0': 16, 'B-VEND_1': 17, 'I-VEND_1': 18, 'B-DATE_0': 19, 'I-DATE_0': 20, 'B-DATE_1': 21, 'I-DATE_1': 22}
ix_to_tag = {idx: word for word, idx in tag_to_ix.items()}

## split train and test data
num_train = math.floor(len(training_data) * 0.9)
training_data, test_data = training_data[:num_train], training_data[num_train:]

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_WEIGHT, weight_decay=WEIGHT_DECAY)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    precheck_sent, precheck_tags = precheck_sent.to(device), precheck_tags.to(device)

losses = []

## train the model if it is not already saved
if not os.path.exists(MODEL_PATH):
    for epoch in range(5):
        for sentence, tags in training_data:
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            sentence_in, targets = sentence_in.to(device), targets.to(device)

            loss = model.neg_log_likelihood(sentence_in, targets)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        if (epoch+1) % 3 == 0:
            print("Epoch: {} Loss: {}".format(epoch+1, np.mean(losses)))

    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    print(f"Model saved!")

## if model exists load it
if os.path.exists(MODEL_PATH):
    start = time.time()
    model.load_state_dict(torch.load(MODEL_PATH))
    end = time.time()
    model.eval()
    print(f"Model loaded!")

## separate word lists from tag lists
test_words = [l[0] for l in test_data]
test_tags = [l[1] for l in test_data]

## flatten tagx
flat_tags = [r for block in test_tags for r in block]

## get ids for each tag
flat_tags_ix = [tag_to_ix[t] for t in flat_tags]

## to keep the predictions
flat_preds = []

with torch.no_grad():
    for l in test_words:
        precheck_sent = prepare_sequence(l, word_to_ix)
        precheck_sent = precheck_sent.to(device)
        # print(precheck_sent)
        pred =  model(precheck_sent)[1]
        flat_preds.extend(pred)

## evaluate by computing the accuracy score
accuracy = accuracy_score(flat_preds, flat_tags_ix)
f1_score = f1_score(flat_tags_ix, flat_preds, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 score: {f1_score:.4f}")
print(f"Koha e trajnimit te blstm eshte: {end - start:.5f} sekonda")

report = flat_classification_report(flat_tags, [ix_to_tag[ix] for ix in flat_preds])
# report = flat_classification_report(y_test, y_pred)
print(report)
