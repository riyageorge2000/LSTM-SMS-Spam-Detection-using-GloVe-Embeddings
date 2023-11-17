# LSTM-Spam-Detection-using-GloVe-Embeddings

This repository contains code for a Long Short-Term Memory (LSTM) model trained to detect spam messages using GloVe word embeddings. It aims to classify SMS messages as either spam or ham (non-spam) using deep learning techniques, specifically LSTM networks.

#### Data Collection and Preprocessing:
The dataset used is the SMSSpamCollection dataset.
Data preprocessing involves converting labels to numerical values and tokenizing the text data.

#### GloVe Embeddings:
GloVe word embeddings are utilized to represent words in a numerical format.
The embeddings are loaded and incorporated into the LSTM model.

#### Model Architecture:
The LSTM model architecture consists of an Embedding layer, LSTM layer, and a Dense output layer with a sigmoid activation function.

#### Training:
The model is trained on the preprocessed dataset with early stopping to prevent overfitting.

#### Evaluation:
Classification report and confusion matrix are generated to evaluate the model's performance on the test dataset.

#### Requirements
Python 3.x
Libraries: pandas, matplotlib, seaborn, numpy, TensorFlow, scikit-learn
