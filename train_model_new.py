from model_extra_new import Model
import json
import pickle
import torch

import torch.nn as nn
import torch.nn.functional as F
import datetime
import time
import torch.optim as optim
from collections import deque
import math
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=2
def load_index_files():
    with open('w2i.json') as f:
         w2i= json.load(f)
    with open('i2w.json') as f:
         i2w= json.load(f)
    with open('w_freq.json') as f:
         w_freq= json.load(f)
    return w2i, i2w,w_freq

def create_vocab_distributions():
    max_val = 0
    for key in i2w:
        temp_val = int(key)
        if (max_val < temp_val):
            max_val = temp_val
    prob_vocab=torch.zeros(max_val+1,requires_grad=False,device=device)
    total_count=0
    for key in w_freq:
        total_count=total_count+w_freq[key]
    for key in i2w:
        index=int(key)
        if(key=="1" or key=="2" or key=="0" ):
            prob_vocab[index]=1/total_count
        else:
           prob_vocab[index]=w_freq[i2w[key]]/total_count
    return prob_vocab


def load_movie_data():
    with open('movie_data.pkl', 'rb') as input:
        movie_data = pickle.load(input)
    return movie_data

def convert_knowledge(knowledge):
    know_data=[]
    le=0
    if(type(knowledge)==list):

        for sentence in knowledge:
            data='<SOS> ' + sentence + ' <EOS>'
            indx_data=convert_sentence_to_index(data)
            le+=indx_data.shape[0]
            know_data.append(indx_data)
        index_data=torch.zeros(le,dtype=torch.long).to(device)
        le=0
        for data in know_data:
            for i in range(0,data.shape[0]):
                index_data[le]=data[i]
                le+=1
    else:
        data = '<SOS> ' + knowledge + ' <EOS>'
        index_data=convert_sentence_to_index(data)
        index_data=index_data.to(device)
        ##know_data.append(index_data)
    return index_data

def convert_sentence_to_index(sentence):
    sent_arr=sentence.split()
    sent_indx=torch.zeros(len(sent_arr),dtype=torch.long)
    sent_indx=sent_indx.to(device)
    for i in range(0,len(sent_arr)):
        sent_indx[i]=w2i[sent_arr[i]]
    return sent_indx

def convert2(sentence):
    sent_arr=sentence.split()
    # sent_indx=torch.zeros(len(sent_arr),dtype=torch.long)
    # sent_indx=sent_indx.to(device)
    sent_indx = []
    for i in range(0,len(sent_arr)):
        sent_indx.append(w2i[sent_arr[i]])
    return sent_indx

def churn_review(review):
    for i  in range(0,len(review)):
        str_arr=review[i].split()
        if(len(str_arr)>390):
            str=''
            for j in range(0,390):
                if(j==0):
                    str=str_arr[0]
                else:
                    str=str+' '+str_arr[j]
            review[i]=str
    return review
def churn_comments(comments):
    if(len(comments)>21):
        return comments[:21]
    return comments

w2i,i2w,w_freq=load_index_files()
movie_data=load_movie_data()
count=0
prob_vocab=create_vocab_distributions()
model=None
"""
count=0
avg=0
for data in movie_data:
    count=count+1
    movie=movie_data[data]
    avg+=(len(movie.comments))
print(avg/count)
"""
def save_model(epoch,loss,optimizer,model):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, "model.pkl")

def load_model(max_val):
    model = Model(256, max_val + 1, prob_vocab)
    optimizer = optim.Adam(model.parameters())

    checkpoint = torch.load("model.pkl")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model,optimizer,epoch,loss

def minibatch(data1, data2, batch_size=2):
   for i in range(0, len(data1), batch_size):
        yield data1[i:i+batch_size], data2[i:i+batch_size]

def preprocess(data1, data2, PAD=0):
    enc = [convert2(seq) for seq in data1]
    dec = [convert2(seq) for seq in data2]

    max_enc = max(map(len, enc))
    max_dec = max(map(len, dec))
    #'<SOS> =1','<EOS>=2'
    seq_enc = [[1]+ seq + [2]+ [PAD] * (max_enc - len(seq)) for seq in enc]
    seq_dec = [seq + [2]+ [PAD] * (max_dec - len(seq)) for seq in dec]    
    # print(seq_dec, seq_enc)
    seq_enc, seq_dec = np.array(seq_enc), np.array(seq_dec)
    ##print(seq_enc.shape, seq_dec.shape)
    return np.array(seq_enc), np.array(seq_dec)



def train_model():
    model_exist=False
    lamb=1e-4
    prob=0.6
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    txt_file=open("d","w")
    start_sent='<SOS>'
    start_index=convert_sentence_to_index(start_sent)
    max_val=0
    for key in i2w:
        temp_val=int(key)
        if(max_val<temp_val):
            max_val=temp_val
    if(model_exist):
        model,optimizer,epoch,loss=load_model(max_val)
        model.to(device)
        optimizer = optim.Adam(model.parameters())
    else:
        model = Model(256, max_val + 1, prob_vocab)
        model.to(device)
        optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    count=0
    chats_complted=0
    for epoch in range(200):
        for data in movie_data:
            tot_loss=0
            count = count + 1
            if(count>0):
                movie = movie_data[data]
                chats = movie.chat
                
                plot = movie.plot
                review = movie.review
                comments = churn_comments(movie.comments)
                review = churn_review(review)
                plot_sent_indx_arr = convert_knowledge(plot)
                
                review_sent_indx_arr = convert_knowledge(review)
                comment_sent_indx_arr = convert_knowledge(comments)
                # just the plot

                #model.knowledge.forward(plot_sent_indx)
                tot_loss = 0
                # if (len(comments) > 0 and len(review) > 0):
                encoder_in =[]
                decoder_ou =[]
                for chat in chats:
                    deq = deque(maxlen=2)
                    talk = chat.chat

                    if len(chat.chat)%2 !=0 and len(chat.chat)>2:
                        talk = talk.pop()

                    encoder_in=talk[0]
                    decoder_ou=talk[1]
                    ##encoder_in.extend(talk[0::1])
                    ##decoder_ou.extend(talk[1::2])

                    for enc , dec in minibatch(encoder_in, decoder_ou):
                        if(len(enc)<batch_size):
                            diff=batch_size-len(enc)
                            for k in range(diff):
                                enc.append("<PAD>")
                                dec.append("<PAD>")
                        enc_lengths=[]
                        dec_lengths=[]
                        for ea in enc:
                            enc_lengths.append(len(ea.split()))
                        for da in dec:
                            dec_lengths.append(len(da.split()))
                        input_sent, dec_sent_index,  = preprocess(enc,dec)
                        enc_lengths = torch.tensor(enc_lengths).long().to(device)
                        dec_lengths = torch.tensor(dec_lengths).long().to(device)
                        input_sent = torch.tensor(input_sent).long().to(device)
                        dec_sent_index = torch.tensor(dec_sent_index).long().to(device)
                        ##print(input_sent.shape,dec_sent_index.shape)
                        ##print(enc_lengths,dec_lengths)
                        know_hidd = model.forward_knowledge_movie(plot_sent_indx_arr, review_sent_indx_arr, comment_sent_indx_arr)
                        isRely = True
                        start_index = torch.tensor([1]).repeat(batch_size,1).long().to(device)
                        output, coverage, current_attention = model.forward(input_sent, dec_sent_index,
                                                                                        start_index,
                                                                                        True, know_hidd,
                                                                                        isRely, plot_sent_indx_arr,
                                                                                        review_sent_indx_arr,
                                                                                         comment_sent_indx_arr,enc_lengths,dec_lengths)
                        org_word_index=dec_sent_index.clone()
                        for j in range(0,dec_sent_index.shape[1]):
                            if (j == 0):
                                att_sum = torch.sum(torch.min(coverage[:,j,:], current_attention[:,j,:]),dim=1)
                            else:
                                att_sum = torch.sum(torch.min(coverage[:,j,:], current_attention[:,j,:]),dim=1) + att_sum
                        ##print(output.view(output.shape[0]*output.shape[1],output.shape[2]).shape,org_word_index[:-1].shape)
                        loss = criterion(output.view(output.shape[0]*output.shape[1],output.shape[2]), org_word_index.view(output.shape[0]*output.shape[1])) + att_sum
                        loss=torch.sum(loss)
                        tot_loss+=loss.item()
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()
                print(tot_loss)
        print("Epoch Completed")
    txt_file.close()
train_model()