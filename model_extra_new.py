import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

init_size=256
batch_size=1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class Encoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size,word_embedding):
        super(Encoder, self).__init__()
        self.word_embedding = word_embedding
        self.hidden_dim=hidden_dim
        self.lstm=nn.LSTM(embedding_dim,hidden_dim, batch_first=True,bidirectional=True)

    def init_hidden(self):
        init_hidden= (torch.zeros(2, batch_size, self.hidden_dim,device=device),
                    torch.zeros(2, batch_size, self.hidden_dim,device=device))
        return init_hidden

    def forward(self, sentence,enc_lengths):
        sorted_lengths,sorted_ids=torch.sort(enc_lengths,descending=True)
        embeds = self.word_embedding(sentence)
        embeds=embeds[sorted_ids]
        packed_input=nn.utils.rnn.pack_padded_sequence(embeds,sorted_lengths,batch_first=True)
        packed_output,self.hidden=self.lstm(packed_input,self.hidden)
        outputs,_=nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)
        ##lstm_out, self.hidden = self.lstm(
        ##    packed_input, self.hidden)#.view(len(sentence), 1, -1)
        ##return lstm_out, self.hidden
        _,reversed_idx=torch.sort(sorted_ids)
        outputs=outputs[reversed_idx]
        ##self.hidden=self.hidden[:,reversed_idx]
        return outputs,self.hidden
class Decoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, word_embedding, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_dim=hidden_dim
        self.word_embedding = word_embedding
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, batch_first=True,bidirectional=False)
        ##self.linear = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, encoder_hidden_state):
        encoder_hidden_state=encoder_hidden_state.view(1,batch_size,encoder_hidden_state.shape[2])
        return (encoder_hidden_state.contiguous(),
                    torch.zeros(1, batch_size, self.hidden_dim,device=device))

    def forward(self,next_word_embedding,dec_lengths):
        embeds = self.word_embedding(next_word_embedding)
        lstm_out, self.hidden = self.lstm(embeds,self.hidden )
        return lstm_out, self.hidden

class KnowledgeRNN(nn.Module):

    def __init__(self,embedding_dim,hidden_dim,word_embedding):
        super(KnowledgeRNN,self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embedding=word_embedding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.resource_attention = Attention(init_size*4)

    def init_hidden(self):
        init_hidden= (torch.zeros(2, 1, self.hidden_dim,device=device),
                    torch.zeros(2, 1, self.hidden_dim,device=device))
        return init_hidden

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        return lstm_out, self.hidden

    def calculate_resource_attention(self,know_hidd,decoder_out):
        decoder_out = decoder_out.view(decoder_out.shape[1],decoder_out.shape[0],decoder_out.shape[2])
        decoder_temp = decoder_out.repeat(1,know_hidd.shape[0], 1)
        know_hidd = know_hidd.unsqueeze(0).repeat(batch_size,1,1)
        attention_input=torch.cat((know_hidd,decoder_temp),dim=2)
        attention_out=self.resource_attention.forward(attention_input)
        attention_weights = F.softmax(attention_out, dim=1)
        attention_weights=attention_weights.transpose(2,1)
        context_vector=torch.bmm(attention_weights, know_hidd)
        return context_vector

class Attention(nn.Module):

    def __init__(self,dim):
        super(Attention, self).__init__()
        self.linear1=nn.Linear(dim,init_size*2)
        self.linear2 = nn.Linear(init_size*2,1)

    def forward(self,input):
        ##input=torch.cat((encoder_hidden_state,decoder_hidden_state),1)
        out1=F.relu(self.linear1(input))
        out2=F.relu(self.linear2(out1))
        return out2

class Pointer(nn.Module):

    def __init__(self,resource_dim,context_dim,dec_hidden_state_dim,prev_input_word_dim,vocab_size):
        super(Pointer, self).__init__()
        self.w_resource=nn.Linear(resource_dim,vocab_size,bias=False)
        self.w_context = nn.Linear(context_dim, vocab_size, bias=False)
        self.w_dec_hidden=nn.Linear(dec_hidden_state_dim,vocab_size,bias=False)
        self.w_prev_input_word=nn.Linear(prev_input_word_dim,vocab_size,bias=False)
        self.bias=torch.randn(vocab_size,device=device)

    def forward(self,resource,context,hidden_state,prev_input_word):
        out=torch.sigmoid(self.w_resource(resource)+self.w_context(context)+self.w_dec_hidden(hidden_state)+self.w_prev_input_word(prev_input_word)+self.bias)
        return out


class Model(nn.Module):

    def __init__(self,embedding_dim,vocab_size,prob_vocab):
        super(Model, self).__init__()
        self.vocab_size=vocab_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)
        self.encoder=nn.DataParallel(Encoder(embedding_dim,init_size,vocab_size, self.word_embedding))
        self.decoder = nn.DataParallel(Decoder(embedding_dim,init_size*2, self.word_embedding, vocab_size))
        self.plot_knowledge=KnowledgeRNN(embedding_dim,init_size,self.word_embedding)
        self.rev_knowledge = KnowledgeRNN(embedding_dim, init_size, self.word_embedding)
        self.com_knowledge = KnowledgeRNN(embedding_dim,init_size, self.word_embedding)
        self.linear1 = nn.DataParallel(nn.Linear(init_size*2*5,vocab_size))
        self.vocab_size=vocab_size
        ##self.linear2 = nn.Linear(2000, vocab_size)
        self.decoder_attention=nn.DataParallel(Attention(init_size*4))
        ##self.p_gen=torch.zeros(1,device="cuda:0")
        self.prob_vocab=prob_vocab
        ###self.pointer=Pointer(init_size*6,init_size*2,init_size*2,init_size,vocab_size)

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



    def forward(self,enc_sent_indx,dec_sent_index,start_index,isTrain,know_hidd,isRely,plot_sent_indx_arr,review_sent_indx_arr,
                comment_sent_indx_arr,enc_lengths,dec_lengths,know_base):
        self.encoder.module.hidden = self.encoder.module.init_hidden() #TODO WE CREATE HIDDEN HERE ACTUALLY
        att_sum=None
        lstm_out, hidden_state=self.encoder.forward(enc_sent_indx,enc_lengths)
        ##lstm_out=lstm_out.contiguous()
        mask_encoders = torch.ones(lstm_out.shape[0], lstm_out.shape[1],1).to(device)
        for k in range(0, len(enc_lengths)):
            mask_encoders[k] = torch.cat((torch.ones(enc_lengths[k],1), torch.zeros(lstm_out.shape[1] - enc_lengths[k],1)))

        lstm_out = lstm_out * mask_encoders
        ini_dec_hidd_state=lstm_out[:,-1,:].unsqueeze(1)
        ##print(ini_dec_hidd_state.shape,lstm_out[:,-1,:].shape)
        ##ini_dec_hidd_state=ini_dec_hidd_state.view(lstm_out.shape[0],1,lstm_out.shape[2])
        ##for k in range(0,batch_size):
        ##    ini_dec_hidd_state[k]=lstm_out[k,enc_lengths[k]-1,:]
        self.decoder.module.hidden=self.decoder.module.init_hidden(ini_dec_hidd_state)
        encoder_out= lstm_out#.view(lstm_out.shape[0], 1, init_size*2)\
        out_word_list = torch.zeros(batch_size,dec_sent_index.shape[1],self.vocab_size,device=device)
        if(isTrain):
            ##mask_encoders=torch.ones(lstm_out.shape[0],lstm_out.shape[1]).to(device)
            ##mask_decoder=torch.ones(lstm_out.shape[0],dec_sent_index.shape[1]).to(device)
            ##for k in range(0, len(dec_lengths)):
            ##    mask_decoder[k] = torch.cat(
            ##        (torch.ones(dec_lengths[k]), torch.zeros(dec_sent_index.shape[1] - dec_lengths[k])))
            ##for k in range(0,len(enc_lengths)):
            ##    mask_encoders[k]=torch.cat((torch.ones(enc_lengths[k]),torch.zeros(lstm_out.shape[1]-enc_lengths[k])))
            ##mask_encoders=mask_encoders.view(mask_encoders.shape[0],mask_encoders.shape[1],1)
            ##lstm_out=lstm_out*mask_encoders
            coverage=torch.zeros(batch_size,dec_sent_index.shape[1],lstm_out.shape[1],device=device)
            current_attention=torch.zeros(batch_size,dec_sent_index.shape[1],lstm_out.shape[1],device=device)
            ##print(coverage.shape,current_attention.shape)
            for i in range(-1,dec_sent_index.shape[1]-1):
                if(i==-1):
                    index=start_index
                    out, hidden_state=self.decoder.forward(start_index,dec_lengths)

                else:
                    if(isRely):
                        next_word=dec_sent_index[:,i]
                        next_word=next_word.reshape(next_word.shape[0],1)
                        ##out, hidden_state = self.decoder.forward(dec_sent_index[i])
                        out, hidden_state = self.decoder.forward(next_word,dec_lengths)
                    else:
                        ##probs = F.softmax(out_word_data, dim=0)
                        ##index = torch.argmax(probs)
                        out, hidden_state = self.decoder.forward(dec_sent_index[i])
                decoder_out=hidden_state[0]
                ##current_state_mask=mask_decoder[:,i+1]
                ##current_state_mask=current_state_mask.view(1,batch_size,1)
                ##ecoder_out=decoder_out*current_state_mask
                resource_context_plot=self.plot_knowledge.calculate_resource_attention(know_hidd[0],decoder_out)
                resource_context_rev = self.rev_knowledge.calculate_resource_attention(know_hidd[1], decoder_out)
                resource_context_com = self.com_knowledge.calculate_resource_attention(know_hidd[2], decoder_out)
                resource_context=torch.cat((resource_context_plot,resource_context_rev,resource_context_com),dim=2)
                ##resource_context = resource_context_plot
                out_word_data,attention_weights,context=self.calculate_decoder_attention(lstm_out,encoder_out,hidden_state,decoder_out,resource_context)
                ##out_word=self.calculate_pointer(resource_context,context,decoder_out,self.word_embedding(index),(plot_sent_indx_arr,resource_context_plot),(review_sent_indx_arr,resource_context_rev)
                ##                                ,(comment_sent_indx_arr,resource_context_com))

                attention_weights=attention_weights.squeeze()
                if(i==-1):
                    temp_sum=torch.zeros(batch_size,lstm_out.shape[1],device=device)
                    ##print(temp_sum.shape)
                else:
                    ##print(last_attention_weight.shape)
                    temp_sum=temp_sum+last_attention_weight
                coverage[:,i+1,:]=temp_sum
                current_attention[:,i+1,:]=attention_weights
                last_attention_weight=attention_weights
                ##att_sum=torch.sum(torch.min(coverage[i+1], current_attention[i+1]))+att_sum
                ##print(coverage)
                ##print(current_attention)
                out_word_data=out_word_data.squeeze()
                ##print(out_word_data.shape)
                out_word_list[:,i+1,:]=out_word_data
                ##probs = F.softmax(out_word_data, dim=0)
                ##index = torch.argmax(out_word_data)


        return out_word_list,coverage,current_attention


        ##attention_input = torch.cat((encoder_out[:, -1], decoder_temp), dim=1)

    def calculate_pointer(self,resource,context,hidden_state,prev_input_word,plot_data,review_data,comment_data):
        p_gen=self.pointer.forward(resource,context,hidden_state,prev_input_word)
        att_word_weights=torch.zeros(self.vocab_size,device="cuda:0")
        bases=[plot_data,review_data,comment_data]
        for base in bases:
            indxs=base[0]
            att=base[1]
            for i in range(0,len(indxs)):
                att_word_weights[indxs[i]]=att_word_weights[indxs[i]]+att[i]
        p_w=p_gen*self.prob_vocab+(1-p_gen)*att_word_weights
        return p_gen

    def calculate_decoder_attention(self,lstm_out,encoder_out,hidden_state,decoder_out,resource_context):
        decoder_out=decoder_out.transpose(0,1)
        ##decoder_temp = decoder_out.repeat(encoder_out.shape[0], 1)
        decoder_temp=decoder_out.repeat(1,encoder_out.shape[1], 1)
        attention_input = torch.cat((encoder_out, decoder_temp), dim=2)
        ##for j in range(0,lstm_out.shape[0]):
        attention_out = self.decoder_attention.forward(attention_input)
        attention_weights = F.softmax(attention_out, dim=1)
        ##att_transpose=attention_weights.transpose(2,1)
        context = torch.bmm(attention_weights.transpose(2,1),lstm_out)
        ##print(context.shape)
        concat = torch.cat((context,decoder_out,resource_context), 2)
        ##concat=torch.cat((context,decoder_out),2)
        out_word_data = self.linear1(concat)
        return out_word_data,attention_weights,context