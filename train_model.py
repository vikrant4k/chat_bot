from model_extra import Model
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
    prob_vocab=torch.zeros(max_val+1,requires_grad=False,device="cuda:0")
    total_count=0
    for key in w_freq:
        total_count=total_count+w_freq[key]
    for key in i2w:
        index=int(key)
        if(key=="1" or key=="0"):
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
        index_data=torch.zeros(le,dtype=torch.long,device="cuda:0")
        le=0
        for data in know_data:
            for i in range(0,data.shape[0]):
                index_data[le]=data[i]
                le+=1
    else:
        data = '<SOS> ' + knowledge + ' <EOS>'
        index_data=convert_sentence_to_index(data)
        index_data=index_data.cuda()
        ##know_data.append(index_data)
    return index_data

def convert_sentence_to_index(sentence):
    sent_arr=sentence.split()
    sent_indx=torch.zeros(len(sent_arr),dtype=torch.long)
    sent_indx=sent_indx.cuda()
    for i in range(0,len(sent_arr)):
        sent_indx[i]=w2i[sent_arr[i]]
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


def train_model():
    model_exist=True
    lamb=1e-4
    prob=0.6
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    txt_file=open(st,"w")
    start_sent='<SOS>'
    start_index=convert_sentence_to_index(start_sent)
    max_val=0
    for key in i2w:
        temp_val=int(key)
        if(max_val<temp_val):
            max_val=temp_val
    if(model_exist):
        model,optimizer,epoch,loss=load_model(max_val)
        model.cuda()
        optimizer = optim.Adam(model.parameters())
    else:
        model = Model(256, max_val + 1, prob_vocab)
        model.cuda()
        optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    count=0
    chats_complted=0
    for epoch in range(200):
        for data in movie_data:
            count = count + 1
            if(count>300):
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

                ##model.knowledge.forward(plot_sent_indx)
                tot_loss = 0
                if (len(comments) > 0 and len(review) > 0):
                    for chat in chats:
                        chats_complted += 1
                        deq = deque(maxlen=2)
                        for i in range(0, len(chat.chat), 2):
                            if ((i + 1) < len(chat.chat)):
                                encoder_sentence = chat.chat[i]  # TODO chat history
                                decoder_sentence = chat.chat[i + 1]

                                encoder_sentence = '<SOS> ' + encoder_sentence + ' <EOS>'
                                decoder_sentence = decoder_sentence + ' <EOS>'
                                enc_sent_indx = convert_sentence_to_index(encoder_sentence)
                                dec_sent_index = convert_sentence_to_index(decoder_sentence)
                                dec_sent_index=dec_sent_index.detach()
                                deq_dec_sent_index = convert_sentence_to_index('<SOS> ' + decoder_sentence)
                                if (len(dec_sent_index) < 350 and len(dec_sent_index) > 2):
                                    if (len(deq) > 0):
                                        input_sent = torch.cat((deq[0], deq[1], enc_sent_indx), dim=0)
                                    else:
                                        input_sent = enc_sent_indx
                                    know_hidd = model.forward_knowledge_movie(plot_sent_indx_arr, review_sent_indx_arr,
                                                                              comment_sent_indx_arr)
                                    ##prob_current = prob * math.exp(-lamb * chats_complted)
                                    ##select = np.random.choice([0, 1], p=[prob_current, 1 - prob_current])
                                    select = 0
                                    if (select == 0):
                                        isRely = True
                                    else:
                                        isRely = False
                                    output, coverage, current_attention = model.forward(input_sent, dec_sent_index,
                                                                                        start_index,
                                                                                        True, know_hidd,
                                                                                        isRely, plot_sent_indx_arr,
                                                                                        review_sent_indx_arr,
                                                                                        comment_sent_indx_arr)  # , plot_sent_indx)
                                    deq.append(enc_sent_indx)
                                    deq.append(deq_dec_sent_index)
                                    ##print(len(output))

                                    output_text = ""
                                    ##att_sum = torch.zeros(coverage.shape, device="cuda:0")
                                    ##org_word_index = torch.zeros(len(dec_sent_index), dtype=torch.long, device="cuda:0",
                                    ##                             requires_grad=False)
                                    org_word_index=dec_sent_index.clone()

                                    for j in range(0, len(dec_sent_index)):
                                    ##    org_word_index[j] = dec_sent_index[j]
                                    ##    if (chats_complted % 10 == 0):
                                    ##        index = torch.argmax(output[j])
                                    ##        output_text += (i2w[str(index.item())]) + " "

                                        if (j == 0):
                                            att_sum = torch.sum(torch.min(coverage[0], current_attention[0]))
                                        else:
                                            att_sum = torch.sum(torch.min(coverage[j], current_attention[j])) + att_sum
                                    loss = criterion(output, org_word_index) + att_sum
                                    tot_loss += loss.item()
                                    if (loss.item() < 90):
                                        ##print(loss.item())
                                        model.zero_grad()
                                        loss.backward()
                                        optimizer.step()
                                        """
                                        if (chats_complted%10 == 0):
                                            print(chats_complted)
                                            txt_file.write("Model: " + output_text.encode('utf-8').decode('utf-8'))
                                            txt_file.write("\n")
                                            txt_file.write("Encoder :" + encoder_sentence.encode('utf-8').decode('utf-8'))
                                            txt_file.write("\n")
                                            txt_file.write("Decoder :" + decoder_sentence.encode('utf-8').decode('utf-8'))
                                            txt_file.write("\n")
                                            if (isRely == False):
                                                txt_file.write("is Rely False")
                                                txt_file.write("\n")
                                        """
                                    else:
                                        h = 1
                                        ##print(loss.item())
                                        ##torch.cuda.empty_cache()
                                    loss = None
                                    know_hidd = None
                print(count,tot_loss / len(chats))
                att_sum = None
                coverage = None
                ##torch.cuda.empty_cache()
                txt_file.write("Movie Completed " + str(count))
                txt_file.write("\n")
                txt_file.write("Chats Completed " + str(chats_complted))
                txt_file.write("\n")
                if (count % 30 == 0):
                    save_model(epoch, loss, optimizer, model)

        print("Epoch Completed")
    txt_file.close()
train_model()