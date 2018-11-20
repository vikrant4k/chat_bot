import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size,word_embedding):
        super(Encoder, self).__init__()
        self.word_embedding = word_embedding
        self.hidden_dim=hidden_dim
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)

    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                    torch.zeros(2, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        return lstm_out, self.hidden

class Decoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, word_embedding, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_dim=hidden_dim
        self.word_embedding = word_embedding
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,bidirectional=False)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, encoder_hidden_state):
        return (encoder_hidden_state,
                    torch.zeros(1, 1, self.hidden_dim))

    def forward(self,next_word_embedding):
        embeds = self.word_embedding(next_word_embedding)

        lstm_out, self.hidden = self.lstm(embeds.view(1, 1, -1),self.hidden )

        lstm_out = self.linear(lstm_out)

        return lstm_out, self.hidden



class Model(nn.Module):

    def __init__(self,embedding_dim,vocab_size):
        super(Model, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder=Encoder(embedding_dim,512,vocab_size, self.word_embedding)
        self.decoder = Decoder(embedding_dim, 1024, self.word_embedding, vocab_size)
        self.linear = nn.Linear(1024*2,vocab_size)

    def forward(self,enc_sent_indx,dec_sent_index,start_index,isTrain):
        self.encoder.hidden = self.encoder.init_hidden() #TODO WE CREATE HIDDEN HERE ACTUALLY

        lstm_out, hidden_state=self.encoder.forward(enc_sent_indx)

        print('hid', hidden_state[0])
        print(lstm_out.shape)

        # # projected_lstm_outs = []
        # #
        # # for ls in lstm_out:
        # #     projected_lstm_outs.append(self.projection(ls).view(1,1,-1))
        #
        # ini_dec_hidd_state=projected_lstm_outs[-1] #already concatenated final hidden of the encoder

        ini_dec_hidd_state = lstm_out[-1].view(1,1,-1)
        self.decoder.hidden=self.decoder.init_hidden(ini_dec_hidd_state)

        out_word_list = []
        if(isTrain):
            for i in range(-1,len(dec_sent_index)):
                if(i==-1):
                    out, hidden_state=self.decoder.forward(start_index)

                else:
                    out, hidden_state=self.decoder.forward(dec_sent_index[i])


                # attention_weights = F.softmax(torch.matmul(lstm_out.view(lstm_out.shape[0], lstm_out.shape[2]), hidden_state[0].view(-1,1)))  # todo bmm
                #
                # mult = torch.matmul(lstm_out.view(lstm_out.shape[2],lstm_out.shape[0]), attention_weights)
                # context = torch.sum(mult,dim=1)
                # concat = torch.cat((context.view(1,1,-1),hidden_state[0].view(1,1,-1)),2)
                # context = self.linear(concat.view(-1))
                #
                # out = F.tanh(context)
                out_word_list.append(out)


        return out_word_list