import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import f1_score
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras.preprocessing.text import text_to_word_sequence
import pickle


# lexon permbajtjen e file-it
def file_to_list(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        content = file.read()
    sentences_list = content.split('\n\n')
    sentences_list = list(filter(None, sentences_list))
    return sentences_list


# kthimi i fjalive ne tuples (fjale, etikete)
def lines_to_tuples(sentences_list):
    sentences_tuple_lst = []
    for s in sentences_list:
        try:
            line_list = s.split('\n')
            line_list = list(filter(None, line_list))
            words_list = []
            for line in line_list:
                word = line.split('\t\t')[0]
                tag = line.split('\t\t')[1]
                words_list.append((word, tag))
            sentences_tuple_lst.append(words_list)
        except:
            continue
    return sentences_tuple_lst



corpus = "korpusi.txt"


#analiza e argumenteve ne rreshtin e komandave
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classifier', choices=['blstm'], required=True)
args = parser.parse_args()

sentences_list = file_to_list(corpus)
sentences_tuple_list = lines_to_tuples(sentences_list)


#mbledhja e fjaleve
all_words = []
for sentence in sentences_tuple_list:
    for tuple in sentence:
        all_words.append(tuple[0])



#mbledhja e etiketave
all_tags = []
for sentence in sentences_tuple_list:
    for tuple in sentence:
        all_tags.append(tuple[1])


#gjetja e fjaleve dhe etiketave unike
unique_words = list(set(all_words))
unique_tags = list(set(all_tags))


#kthimi i fjaleve ne indexe
word_index = {}
for index, word in enumerate(unique_words):
    word_index[word] = index


#kthimi i etiketave ne indexe
tag_index = {}
for index, tag in enumerate(unique_tags):
    tag_index[tag] = index



#ndarja e fjaleve nga etiketat
X = []
for sentence in sentences_tuple_list:
    sentence_words = []
    for word in sentence:
        sentence_words.append(word[0])
    X.append(sentence_words)


#ndarja e fjaleve nga etiketat
y = []
for sentence in sentences_tuple_list:
    sentence_tags = []
    for word in sentence:
        sentence_tags.append(word[1])
    y.append(sentence_tags)



#kodimi i fjaleve
X_encoded = []
for sentence in X:
    sentence_encoded = []
    for word in sentence:
        sentence_encoded.append(word_index[word])
    X_encoded.append(sentence_encoded)


X = X_encoded


#kodimi i etiketave
y_encoded = []
for sentence in y:
    sentence_encoded = []
    for tag in sentence:
        sentence_encoded.append(tag_index[tag])
    y_encoded.append(sentence_encoded)


y = y_encoded


# rrafshimi i listes se listave ne nje liste te madhe (per fjalet e koduara)
X_flattened = [word for sentence in X for word in sentence]
X = np.array(X_flattened).reshape(-1, 1)

# rrafshimi i listes se listave ne nje liste te madhe (per etiketat e koduara)
y_flattened = [tag for tags in y for tag in tags]
y = np.array(y_flattened)
##KTU VER GJITHE PARAPROCESIMIN E DIPLOMES

# Number of data points passed in each iteration
batch_size = 64
# Passes through entire dataset
epochs = 8
# Maximum length of review
max_len = 75
# Dimension of embedding vector
embedding = 40


#Getting unique words and labels from data
# words = list(df['Word'].unique())
# tags = list(df['Tag'].unique())
# Dictionary word:index pair
# word is key and its value is corresponding index
# word_to_index = {w : i + 2 for i, w in enumerate(words)}
# word_to_index["UNK"] = 1
# word_to_index["PAD"] = 0
#
# # Dictionary lable:index pair
# # label is key and value is index.
# tag_to_index = {t : i + 1 for i, t in enumerate(tags)}
# tag_to_index["PAD"] = 0

# idx2word = {i: w for w, i in word_to_index.items()}
# idx2tag = {i: w for w, i in tag_to_index.items()}

# Converting each sentence into list of index from list of tokens
# X = [[word_to_index[w[0]] for w in s] for s in sentences_tuple_list]

# Padding each sequence to have same length  of each word
X = pad_sequences(maxlen = max_len, sequences = X, padding = "post", value = word_to_index["PAD"])

# Convert label to index
# y = [[tag_to_index[w[2]] for w in s] for s in sentences_tuple_list]

# padding
y = pad_sequences(maxlen = max_len, sequences = y, padding = "post", value = tag_to_index["PAD"])

# num_tag = df['Tag'].nunique()
num_tag = len(unique_tags)
# One hot encoded labels
y = [to_categorical(i, num_classes = num_tag + 1) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

print("Size of training input data : ", X_train.shape)
print("Size of training output data : ", np.array(y_train).shape)
print("Size of testing input data : ", X_test.shape)
print("Size of testing output data : ", np.array(y_test).shape)


##Bidirectional LSTM-CRF Network
# num_tags = df['Tag'].nunique()
# Extract all labels from the sentences
all_labels = [label for sentence in sentences_tuple_list for _, label in sentence]

# Get the number of unique labels
num_tags = len(set(all_labels))

# Model architecture
input = Input(shape = (max_len,))
model = Embedding(input_dim = len(word_to_index) + 2, output_dim = embedding, input_length = max_len, mask_zero = True)(input)
model = Bidirectional(LSTM(units = 50, return_sequences=True, recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)
crf = CRF(num_tags+1)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()


checkpointer = ModelCheckpoint(filepath = 'model.h5',
                       verbose = 0,
                       mode = 'auto',
                       save_best_only = True,
                       monitor='val_loss')

start = time.time()
blstm_crf = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=epochs,
                    validation_split=0.1, callbacks=[checkpointer])
end= time.time()

# history.history.keys()


# Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y_test, -1)

# Reverse the tag_to_index dictionary to get idx2tag
idx2tag = {i: t for t, i in tag_index.items()}

# Convert the index to tag
y_pred = [[idx2tag[i] for i in row] for row in y_pred]
y_test_true = [[idx2tag[i] for i in row] for row in y_test_true]


print("F1-score is : {:.1%}".format(f1_score(y_test_true, y_pred)))

# Flatten the lists of predicted and true labels to calculate token-level accuracy
y_pred_flat = [tag for sentence in y_pred for tag in sentence]
y_true_flat = [tag for sentence in y_test_true for tag in sentence]

# Calculate token-level accuracy
accuracy = accuracy_score(y_true_flat, y_pred_flat)

print(f"Vlera e saktesise se blstm-crf eshte: {accuracy:.4f}")
# print(f"Vlera F1 e blstm-crf eshte: {f1:.4f}")
print(f"Koha e trajnimit te blstm-crf eshte: {end - start:.5f} sekonda")

report = flat_classification_report(y_pred=y_pred, y_true=y_test_true)
print(report)