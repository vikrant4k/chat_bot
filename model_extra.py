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

class KnowledgeRNN(nn.Module):

    def __init__(self,embedding_dim,hidden_dim,word_embedding):
        super(KnowledgeRNN,self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embedding=word_embedding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.resource_attention = Attention(512)

    def init_hidden(self):
        init_hidden= (torch.zeros(2, 1, self.hidden_dim,device="cuda:0"),
                    torch.zeros(2, 1, self.hidden_dim,device="cuda:0"))
        return init_hidden

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        return lstm_out, self.hidden

    def calculate_resource_attention(self,know_hidd,decoder_out):
        decoder_temp = decoder_out.repeat(know_hidd.shape[0], 1)
        attention_input=torch.cat((know_hidd,decoder_temp),dim=1)
        attention_out=self.resource_attention.forward(attention_input)
        attention_weights = F.softmax(attention_out, dim=0)
        attention_weights=attention_weights.squeeze()
        context_vector=torch.matmul(torch.t(know_hidd),attention_weights)
        return context_vector

class Attention(nn.Module):

    def __init__(self,dim):
        super(Attention, self).__init__()
        self.linear1=nn.Linear(dim,256)
        self.linear2 = nn.Linear(256,1)

    def forward(self,input):
        ##input=torch.cat((encoder_hidden_state,decoder_hidden_state),1)
        out1=F.relu(self.linear1(input))
        out2=F.relu(self.linear2(out1))
        return out2

class Pointer(nn.Module):

    def __init__(self,resource_dim,context_dim,dec_hidden_state_dim,prev_input_word_dim):
        super(Pointer, self).__init__()
        self.w_resource=nn.Linear(resource_dim,1,bias=False)
        self.w_context = nn.Linear(context_dim, 1, bias=False)
        self.w_dec_hidden=nn.Linear(dec_hidden_state_dim,1,bias=False)
        self.w_prev_input_word=nn.Linear(prev_input_word_dim,1,bias=False)
        self.bias=torch.randn(1)

    def forward(self,resource,context,hidden_state,prev_input_word):
        out=F.sigmoid(self.w_resource(resource)+self.w_context(context)+self.w_dec_hidden(hidden_state)+self.w_prev_input_word(prev_input_word)+self.bias)
        return out


class Model(nn.Module):

    def __init__(self,embedding_dim,vocab_size,prob_vocab):
        super(Model, self).__init__()
        self.vocab_size=vocab_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder=Encoder(embedding_dim,128,vocab_size, self.word_embedding)
        self.decoder = Decoder(embedding_dim, 256, self.word_embedding, vocab_size)
        self.plot_knowledge=KnowledgeRNN(embedding_dim,128,self.word_embedding)
        self.rev_knowledge = KnowledgeRNN(embedding_dim, 128, self.word_embedding)
        self.com_knowledge = KnowledgeRNN(embedding_dim, 128, self.word_embedding)
        self.linear1 = nn.Linear(256*5,vocab_size)
        ##self.linear2 = nn.Linear(2000, vocab_size)
        self.decoder_attention=Attention(512)
        self.p_gen=torch.zeros(1,device="cuda:0")
        self.prob_vocab=prob_vocab
        self.pointer=Pointer(768,256,256,512)

    def forward_knowledge_movie(self,plot,review,comment):
        data_out=[]
        self.plot_knowledge.hidden=self.plot_knowledge.init_hidden()
        lstm_out, hidden_state = self.plot_knowledge.forward(plot)
        data_out.append(lstm_out.squeeze())
        self.rev_knowledge.hidden = self.rev_knowledge.init_hidden()
        lstm_out1, hidden_state = self.rev_knowledge.forward(review)
        lstm_out1=lstm_out1.squeeze()
        data_out.append(lstm_out1)
        self.com_knowledge.hidden = self.com_knowledge.init_hidden()
        lstm_out2, hidden_state = self.com_knowledge.forward(comment)
        lstm_out2 = lstm_out2.squeeze()

        data_out.append(lstm_out2)
        return data_out



    def forward(self,enc_sent_indx,dec_sent_index,start_index,isTrain,know_hidd,isRely):
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
        encoder_out = lstm_out.view(lstm_out.shape[0], 1, 256)
        out_word_list = torch.zeros(len(dec_sent_index),self.vocab_size,device="cuda:0")
        if(isTrain):
            coverage=torch.zeros(len(dec_sent_index),lstm_out.shape[0],device="cuda:0")
            current_attention=torch.zeros(len(dec_sent_index),lstm_out.shape[0],device="cuda:0")

            for i in range(-1,len(dec_sent_index)-1):
                if(i==-1):
                    index=start_index
                    out, hidden_state=self.decoder.forward(start_index)

                else:
                    if(isRely):
                        ##out, hidden_state = self.decoder.forward(dec_sent_index[i])
                        probs=F.softmax(out_word_data,dim=0)
                        index = torch.argmax(probs)
                        out, hidden_state = self.decoder.forward(index)
                    else:
                        probs = F.softmax(out_word_data, dim=0)
                        index = torch.argmax(probs)
                        out, hidden_state = self.decoder.forward(index)
                decoder_out=hidden_state[0].view(1,256)
                resource_context_plot=self.plot_knowledge.calculate_resource_attention(know_hidd[0],decoder_out)
                resource_context_rev = self.rev_knowledge.calculate_resource_attention(know_hidd[1], decoder_out)
                resource_context_com = self.com_knowledge.calculate_resource_attention(know_hidd[2], decoder_out)
                resource_context=torch.cat((resource_context_plot,resource_context_rev,resource_context_com),dim=0)
                out_word_data,attention_weights,context=self.calculate_decoder_attention(lstm_out,encoder_out,hidden_state,decoder_out,resource_context)
                if(i==-1):
                    temp_sum=torch.zeros(1,lstm_out.shape[0],device="cuda:0")
                else:
                    temp_sum=temp_sum+last_attention_weight.view(attention_weights.shape[1],attention_weights.shape[0])

                coverage[i+1,:]=temp_sum
                current_attention[i+1,:]=attention_weights.view(-1)
                last_attention_weight=attention_weights
                ##print(coverage)
                ##print(current_attention)
                out_word_list[i,:]=out_word_data


        return out_word_list,coverage,current_attention


        ##attention_input = torch.cat((encoder_out[:, -1], decoder_temp), dim=1)

    def calculate_pointer(self,resource,context,hidden_state,prev_input_word):
        p_gen=self.pointer.forward(resource,context,hidden_state,prev_input_word)
        return p_gen

    def calculate_decoder_attention(self,lstm_out,encoder_out,hidden_state,decoder_out,resource_context):
        decoder_temp = decoder_out.repeat(encoder_out.shape[0], 1)
        attention_input = torch.cat((encoder_out[:, -1], decoder_temp), dim=1)
        ##for j in range(0,lstm_out.shape[0]):
        attention_out = self.decoder_attention.forward(attention_input)
        attention_weights = F.softmax(attention_out, dim=0)
        mult = torch.matmul(lstm_out.view(lstm_out.shape[2], lstm_out.shape[0]), attention_weights)
        context = torch.sum(mult, dim=1)
        concat = torch.cat((context.view(1, 1, -1), hidden_state[0].view(1, 1, -1),resource_context.view(1,1,-1)), 2)
        out_word_data = self.linear1(concat.view(-1))
        return out_word_data,attention_weights,context