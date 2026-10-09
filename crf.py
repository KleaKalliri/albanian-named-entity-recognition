import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_classification_report

import numpy as np
import sys, argparse, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error
from math import sqrt


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
parser.add_argument('-c', '--classifier', choices=['crf'], help='Modeli klasifikues', required=True)
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

# #rrafshimi i listes se listave ne nje liste te madhe (per fjalet e koduara)
# X_flattened = []
# for sentence in X:
#     for word in sentence:
#         X_flattened.append(word)
#
#
# X = X_flattened
#
#
# #rrafshimi i listes se listave ne nje liste te madhe (per etiketat e koduara)
# y_flattened = []
# for tags in y:
#     for tag in tags:
#         y_flattened.append(tag)
#
#
# y = y_flattened


# #konvertimi i te dhenave ne matrica numpy
# X = np.array(X)
# X = X.reshape(-1, 1)
# y = np.array(y)

def word2features(sent, i):
    word = sent[i][0]
    # postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        # 'postag': postag,
        # 'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        # postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            # '-1:postag': postag1,
            # '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        # postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            # '+1:postag': postag1,
            # '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]


X = [sent2features(s) for s in sentences_tuple_list]
y = [sent2labels(s) for s in sentences_tuple_list]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

crf = CRF(algorithm = 'lbfgs',
         c1 = 0.1,
         c2 = 0.1,
         max_iterations = 100,
         all_possible_transitions = False)

start = time.time()
crf.fit(X_train, y_train)
end = time.time()

#Predicting on the test set.
y_pred = crf.predict(X_test)

f1_sc = flat_f1_score(y_test, y_pred, average = 'weighted')
print(f1_sc)

report = flat_classification_report(y_test, y_pred)
print(report)


# #trajnimi i modelit dhe matja e kohes
# start = time.time()
# model.fit(X_train, y_train)
# end = time.time()

#rezultatet e klasifikimit
# y_prediction = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average='weighted')
#
#
# print(f"Vlera e saktesise se crf eshte: {accuracy:.4f}")
# print(f"Vlera F1 e crf eshte: {f1:.4f}")
# print(f"Koha e trajnimit te crf eshte: {end - start:.5f} sekonda")


# Flatten y_test and y_pred
y_test_flat = [label for seq in y_test for label in seq]
y_pred_flat = [label for seq in y_pred for label in seq]

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test_flat, y_pred_flat)
f1 = f1_score(y_test_flat, y_pred_flat, average='weighted')

print(f"Vlera e saktesise se crf eshte: {accuracy:.4f}")
print(f"Vlera F1 e crf eshte: {f1:.4f}")
print(f"Koha e trajnimit te crf eshte: {end - start:.5f} sekonda")


correctly_classified = sum([y_true == y_pred for y_true, y_pred in zip(y_test_flat, y_pred_flat)])
print(f"Correctly Classified Instances: {correctly_classified}")


incorrectly_classified = len(y_test_flat) - correctly_classified
print(f"Incorrectly Classified Instances: {incorrectly_classified}")


kappa = cohen_kappa_score(y_test_flat, y_pred_flat)
print(f"Kappa Statistic: {kappa:.4f}")

mae = np.mean(np.array(y_test_flat) != np.array(y_pred_flat))  # 1 if incorrect, 0 if correct
print(f"Mean Absolute Error: {mae:.4f}")

# rmse = mean_squared_error(y_test_flat, y_pred_flat, squared=False)
# Calculate Root Mean Squared Error (RMSE)

# Map the labels to numeric values
label_to_index = {label: idx for idx, label in enumerate(unique_tags)}  # assuming `unique_tags` holds the tag names
y_test_flat_numeric = [label_to_index[label] for label in y_test_flat]
y_pred_flat_numeric = [label_to_index[label] for label in y_pred_flat]

mse = mean_squared_error(y_test_flat_numeric, y_pred_flat_numeric)
rmse = sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"Root Mean Squared Error: {rmse:.4f}")

# mse = mean_squared_error(y_test_flat, y_pred_flat)  # Calculate Mean Squared Error
# rmse = np.sqrt(mse)  # Take the square root of MSE
# print(f"Root Mean Squared Error: {rmse:.4f}")

# Convert arrays to numeric values, assuming the arrays contain numeric data as strings
# y_test_flat = np.array(y_test_flat, dtype=float)
# y_pred_flat = np.array(y_pred_flat, dtype=float)
#
# # Now calculate the Relative Absolute Error
# relative_absolute_error = sum(abs(y_test_flat - y_pred_flat)) / sum(abs(y_test_flat - np.mean(y_test_flat)))
# print(f"Relative Absolute Error: {relative_absolute_error:.4f}")


# relative_absolute_error = sum(abs(np.array(y_test_flat) - np.array(y_pred_flat))) / sum(abs(np.array(y_test_flat) - np.mean(y_test_flat)))
# Calculate the absolute errors
absolute_errors = np.abs(np.array(y_test_flat_numeric) - np.array(y_pred_flat_numeric))

# Calculate the mean of the true values
mean_true_values = np.mean(y_test_flat_numeric)

# Calculate the total absolute error of the true values from the mean
total_absolute_error = np.sum(np.abs(np.array(y_test_flat_numeric) - mean_true_values))

# Calculate Relative Absolute Error (RAE)
rae = np.sum(absolute_errors) / total_absolute_error
print(f"Relative Absolute Error: {rae:.4f}")

# relative_squared_error = np.sqrt(sum((np.array(y_test_flat) - np.array(y_pred_flat)) ** 2) / sum((np.array(y_test_flat) - np.mean(y_test_flat)) ** 2))
# Calculate the squared errors
squared_errors = (np.array(y_test_flat_numeric) - np.array(y_pred_flat_numeric)) ** 2

# Calculate the mean of the true values
mean_true_values = np.mean(y_test_flat_numeric)

# Calculate the squared differences between the true values and the mean
squared_total_error = np.sum((np.array(y_test_flat_numeric) - mean_true_values) ** 2)

# Calculate Root Relative Squared Error (RRSE)
rrse = np.sqrt(np.sum(squared_errors) / squared_total_error)
print(f"Root Relative Squared Error: {rrse:.4f}")

# Now calculate the Root Relative Squared Error
# relative_squared_error = np.sqrt(sum((y_test_flat - y_pred_flat) ** 2) / sum((y_test_flat - np.mean(y_test_flat)) ** 2))
# print(f"Root Relative Squared Error: {relative_squared_error:.4f}")
