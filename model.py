import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_dim=hidden_dim
        self.word_embedding=nn.Embedding(vocab_size,embedding_dim)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)

    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                    torch.zeros(2, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        return lstm_out

class Decoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, encoder):
        super(Encoder, self).__init__()
        self.hidden_dim=hidden_dim
        self.encoder=encoder
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,bidirectional=False)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                    torch.zeros(1, 1, self.hidden_dim))


class Model():

    def __init__(self,embedding_dim,vocab_size):
        self.encoder=Encoder(embedding_dim,512,vocab_size)

    def forward(self,sentence):
        hidden_state=self.encoder.forward(sentence)
        print(hidden_state)

