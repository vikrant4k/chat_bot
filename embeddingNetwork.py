import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, hidden_dim, embedding_dim, vocab_size, no_of_movies):
        #(256, 64, max_val + 1, no_of_movies)
        super(Encoder, self).__init__()
        self.hidden_dim=hidden_dim
        self.word_embedding=nn.Embedding(vocab_size, embedding_dim)
        self.movie_embedding=nn.Embedding(no_of_movies, embedding_dim)
        self.lstm_plot=nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        self.lstm_comments = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.lstm_review = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.linear_plot = nn.Linear(hidden_dim*2, embedding_dim)
        self.linear_comments = nn.Linear(hidden_dim*2, embedding_dim)
        self.linear_review = nn.Linear(hidden_dim*2, embedding_dim)
        self.hidden_plot = self.init_hidden()
        self.hidden_comments = self.init_hidden()
        self.hidden_review = self.init_hidden()
        #self.tanh = nn.Tanh()


    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                    torch.zeros(2, 1, self.hidden_dim))

    def forward(self, movie_index, knowledge_base):
        self.hidden_plot = self.init_hidden()
        self.hidden_comments = self.init_hidden()
        self.hidden_review = self.init_hidden()

        plot = knowledge_base[0]
        comments = knowledge_base[1]
        review = knowledge_base[2]

        embedded_movie = self.movie_embedding(movie_index)
        embedded_plot = self.word_embedding(plot)
        embedded_comment = self.word_embedding(comments)
        embedded_review = self.word_embedding(review)

        lstm_out_plot, self.hidden_plot = self.lstm_plot(embedded_plot.view(len(plot), 1, -1), self.hidden_plot)
        lstm_out_comments, self.hidden_comments = self.lstm_comments(embedded_comment.view(len(comments), 1, -1), self.hidden_comments)
        lstm_out_review, self.hidden_review = self.lstm_review(embedded_review.view(len(review), 1, -1), self.hidden_review)

        lstm_out_plot = self.linear_plot(lstm_out_plot)
        lstm_out_comments = self.linear_comments(lstm_out_comments)
        lstm_out_review = self.linear_review(lstm_out_review)

        lstm_out_kb = lstm_out_plot[-1] + lstm_out_comments[-1]  + lstm_out_review[-1]

        network_out = torch.matmul(lstm_out_kb[-1], embedded_movie.view(-1,1))

        lstm_norm = (torch.sqrt(torch.matmul(lstm_out_kb[-1].view(1,-1),lstm_out_kb[-1].view(-1,1)))).detach()
        lstm_embed_norm = (torch.sqrt(torch.matmul(embedded_movie.view(1,-1),embedded_movie.view(-1,1)))).detach()

        network_out = network_out / (lstm_norm*lstm_embed_norm)

        return network_out, lstm_out_kb, lstm_norm