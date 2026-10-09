# Albanian Named Entity Recognition

An experimental Python project exploring Named Entity Recognition (NER) for Albanian text using statistical, neural, and transformer-based approaches.

## Overview

Named Entity Recognition identifies and labels entities within text, such as people, organizations, locations, and dates.

This project uses an annotated Albanian corpus to explore different approaches to token classification, including Conditional Random Fields (CRF), Bidirectional LSTM models, and BERT.

## Models

| Approach | Description |
|---|---|
| CRF | A feature-based model using token characteristics and neighboring words. |
| BiLSTM-CRF | A bidirectional LSTM combined with a CRF layer for sequence labeling. |
| BERT | Token classification experiments using `bert-base-uncased`. |
| BiLSTM | A TensorFlow/Keras sequence-labeling model with a softmax output layer. |

## Repository Contents

| File | Description |
|---|---|
| `crf.py` | Feature extraction, CRF training, and evaluation. |
| `blstm.py` | PyTorch BiLSTM-CRF training and evaluation. |
| `blstm_utils.py` | BiLSTM-CRF architecture and supporting functions. |
| `BLSTM-CRF.py` | An alternative experimental Keras-based BiLSTM-CRF implementation. |
| `BERT.py` | BERT token classification experiment. |
| `train_bilstm.py` | TensorFlow/Keras BiLSTM training script with SageMaker environment settings. |
| `txt_to_csv.py` | Conversion of the annotated text corpus to CSV. |
| `korpusi.txt` | Annotated Albanian corpus in token–label format. |
| `korpusi.csv` | CSV version of the corpus with `Token` and `Label` columns. |

## Dataset

The text corpus contains token–label pairs, with sentences separated by blank lines.

Annotations use BIO-style labels:

- `B-`: beginning of an entity.
- `I-`: continuation of an entity.
- `O`: token outside an entity.

The label set includes people, organizations, events, streets, squares, locations, dates, and additional corpus-specific categories.

## Technologies

- Python
- PyTorch
- TensorFlow / Keras
- Hugging Face Transformers
- scikit-learn
- sklearn-crfsuite
- seqeval
- pandas and NumPy

## Evaluation

The scripts include evaluation through classification reports, accuracy, and F1 scores. Metrics and train/test splits differ between implementations, so results require a consistent evaluation setup before direct comparison.

## Project Status

This repository contains experimental research code rather than a packaged application.

Some scripts require adjustments before execution:

- `BERT.py` contains notebook-specific syntax and requires code cleanup.
- The alternative Keras BiLSTM-CRF script contains incomplete variable definitions and uses `keras_contrib`.
- `train_bilstm.py` expects SageMaker environment variables and preprocessed training data.
- Dependency versions and dataset paths must be configured for the chosen implementation.

The BERT experiment uses `bert-base-uncased`, an English pretrained model, rather than an Albanian-specific or multilingual model.

## Purpose

The project explores statistical and deep learning approaches to named entity recognition in Albanian, covering corpus preprocessing, sequence labeling, model training, and evaluation.
