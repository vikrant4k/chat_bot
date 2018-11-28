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
        init_hidden= (torch.zeros(2, 1, self.hidden_dim,device="cuda:0"),
                    torch.zeros(2, 1, self.hidden_dim,device="cuda:0"))
        return init_hidden

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
        ##self.linear = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, encoder_hidden_state):
        return (encoder_hidden_state,
                    torch.zeros(1, 1, self.hidden_dim,device="cuda:0"))

    def forward(self,next_word_embedding):
        embeds = self.word_embedding(next_word_embedding)

        lstm_out, self.hidden = self.lstm(embeds.view(1, 1, -1),self.hidden )

        ##lstm_out = self.linear(lstm_out)

        return lstm_out, self.hidden

class Attention(nn.Module):

    def __init__(self,dim):
        super(Attention, self).__init__()
        self.linear1=nn.Linear(dim,256)
        self.linear2 = nn.Linear(256,1)

    def forward(self,encoder_hidden_state,decoder_hidden_state):
        input=torch.cat((encoder_hidden_state,decoder_hidden_state),1)
        out1=F.relu(self.linear1(input))
        out2=F.relu(self.linear2(out1))
        return out2

class Model(nn.Module):

    def __init__(self,embedding_dim,vocab_size):
        super(Model, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder=Encoder(embedding_dim,256,vocab_size, self.word_embedding)
        self.decoder = Decoder(embedding_dim, 512, self.word_embedding, vocab_size)
        self.linear = nn.Linear(512*2,vocab_size)
        self.attention=Attention(1024)
    def forward(self,enc_sent_indx,dec_sent_index,start_index,isTrain):
        self.encoder.hidden = self.encoder.init_hidden() #TODO WE CREATE HIDDEN HERE ACTUALLY

        lstm_out, hidden_state=self.encoder.forward(enc_sent_indx)

        ##print('hid', hidden_state[0])
        ##pint(lstm_out.shape)

        # # projected_lstm_outs = []
        # #
        # # for ls in lstm_out:
        # #     projected_lstm_outs.append(self.projection(ls).view(1,1,-1))
        #
        # ini_dec_hidd_state=projected_lstm_outs[-1] #already concatenated final hidden of the encoder

        ini_dec_hidd_state = lstm_out[-1].view(1,1,-1)
        self.decoder.hidden=self.decoder.init_hidden(ini_dec_hidd_state)
        encoder_out = lstm_out.view(lstm_out.shape[0], 1, 512)
        out_word_list = []
        if(isTrain):
            coverage=torch.zeros(len(dec_sent_index),lstm_out.shape[0],device="cuda:0")
            current_attention=torch.zeros(len(dec_sent_index),lstm_out.shape[0],device="cuda:0")
            for i in range(-1,len(dec_sent_index)-1):
                if(i==-1):
                    out, hidden_state=self.decoder.forward(start_index)

                else:
                    out, hidden_state=self.decoder.forward(dec_sent_index[i])


                decoder_out=hidden_state[0].view(1,512)
                attention_out=torch.zeros(lstm_out.shape[0],1,device="cuda:0")
                for j in range(0,lstm_out.shape[0]):
                       attention_out[j]=self.attention.forward(encoder_out[j],decoder_out)
                attention_weights=F.softmax(attention_out,dim=0)
                mult = torch.matmul(lstm_out.view(lstm_out.shape[2], lstm_out.shape[0]), attention_weights)
                context = torch.sum(mult, dim=1)
                concat = torch.cat((context.view(1, 1, -1), hidden_state[0].view(1, 1, -1)), 2)
                out_word_data = self.linear(concat.view(-1))
                if(i==-1):
                    temp_sum=torch.zeros(1,lstm_out.shape[0],device="cuda:0")
                else:
                    temp_sum=torch.sum(current_attention)
                coverage[i+1,:]=temp_sum
                ##print(coverage)
                current_attention[i+1,:]=attention_weights.view(-1)

                ##out_word = tout_word_data)
                ##attention_weights = F.softmax(torch.matmul(lstm_out.view(lstm_out.shape[0], lstm_out.shape[2]), hidden_state[0].view(-1,1)))  # todo bmm

                ##mult = torch.matmul(lstm_out.view(lstm_out.shape[2],lstm_out.shape[0]), attention_weights)
                ##context = torch.sum(mult,dim=1)
                ##concat = torch.cat((context.view(1,1,-1),hidden_state[0].view(1,1,-1)),2)
                ##context = self.linear(concat.view(-1))
                ##out = F.relu(context)
                out_word_list.append(out_word_data)


        return out_word_list,coverage,current_attention