from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# Define the model class FIRST
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Then load the model
model = joblib.load('rnn_sentiment_model.pkl')
vocab = joblib.load('vocab.pkl')
model.eval()
