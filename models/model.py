import torch
import torch.nn as nn

class LSTMRecommender(nn.Module):
    def __init__(self, n_movies, emb_dim=64, hidden_dim=128):
        super().__init__()
        
        self.movie_emb = nn.Embedding(n_movies, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_movies)

    def forward(self, x):
        x = self.movie_emb(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))