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
    print(prob_vocab)
    return prob_vocab


def load_movie_data():
    with open('movie_data.pkl', 'rb') as input:
        movie_data = pickle.load(input)
    return movie_data



def convert_sentence_to_index(sentence):
    sent_arr=sentence.split()
    sent_indx=torch.zeros(len(sent_arr),dtype=torch.long)
    sent_indx=sent_indx.cuda()
    for i in range(0,len(sent_arr)):
        sent_indx[i]=w2i[sent_arr[i]]
    return sent_indx

w2i,i2w,w_freq=load_index_files()
movie_data=load_movie_data()
prob_vocab=create_vocab_distributions()
model=None

def train_model():
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
    model=Model(512,max_val+1,prob_vocab)
    model.cuda()
    lis=model.parameters()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        for data in movie_data:
             #TODO ALSO FOR DECODER AND OTHER ENCODERS?
            movie=movie_data[data]
            chats=movie.chat
            plot=movie.plot
            #just the plot
            ##plot_sent_indx = convert_sentence_to_index(movie.plot)
            ##print(movie.plot)
             ##model.knowledge.forward(plot_sent_indx)

            for chat in chats:
                deq = deque(maxlen=2)
                for i in range(0,len(chat.chat),2):
                    if((i+1)<len(chat.chat)):
                        encoder_sentence = chat.chat[i]  # TODO chat history
                        decoder_sentence = chat.chat[i + 1]


                        encoder_sentence = '<SOS> ' + encoder_sentence + ' <EOS>'
                        decoder_sentence = decoder_sentence + ' <EOS>'
                        enc_sent_indx = convert_sentence_to_index(encoder_sentence)
                        dec_sent_index = convert_sentence_to_index(decoder_sentence)
                        deq_dec_sent_index=convert_sentence_to_index('<SOS> '+decoder_sentence)
                        if(len(dec_sent_index)<400 and len(dec_sent_index)>2):
                            if(len(deq)>0):
                              input_sent=torch.cat((deq[0],deq[1],enc_sent_indx),dim=0)
                            else:
                                input_sent=enc_sent_indx
                            output,coverage,current_attention = model.forward(input_sent, dec_sent_index, start_index,
                                                   True)  # , plot_sent_indx)
                            deq.append(enc_sent_indx)
                            deq.append(deq_dec_sent_index)
                            ##print(len(output))

                            output_text = ""
                            att_sum=torch.zeros(coverage.shape,device="cuda:0")
                            org_word_index = torch.zeros(len(dec_sent_index), dtype=torch.long, device="cuda:0", requires_grad=False)
                            for j in range(0, len(dec_sent_index)):
                                org_word_index[j] = dec_sent_index[j]
                                index = torch.argmax(output[j])
                                output_text += (i2w[str(index.item())]) + " "
                                if(j==0):
                                    att_sum = torch.sum(torch.min(coverage[0], current_attention[0]))
                                else:
                                    att_sum = torch.sum(torch.min(coverage[j], current_attention[j])) + att_sum
                            loss=criterion(output, org_word_index)+att_sum
                            """
                            for j in range(0, len(dec_sent_index)):
                                org_word_index = torch.zeros(1, dtype=torch.long, device="cuda:0",requires_grad=False)
                                org_word_index[0] = dec_sent_index[j]
                                index = torch.argmax(output[j])

                                output_text+=(i2w[str(index.item())])+" "
                                if (j == 0):
                                    loss = criterion(output[j].view(1, -1), org_word_index)
                                    att_sum=torch.sum(torch.min(coverage[0],current_attention[0]))
                                    loss+=att_sum
                                else:
                                    loss += criterion(output[j].view(1, -1), org_word_index)
                                    att_sum = torch.sum(torch.min(coverage[j], current_attention[j]))+att_sum
                                    loss+=att_sum
                            """
                            print(loss.item())
                            model.zero_grad()
                            loss.backward()
                            optimizer.step()
                            txt_file.write("Model: "+output_text)
                            txt_file.write("\n")
                            txt_file.write("Encoder :"+encoder_sentence)
                            txt_file.write("\n")
                            txt_file.write("Decoder :" + decoder_sentence)
                            txt_file.write("\n")
                            ##print(output_text, encoder_sentence, decoder_sentence)
        print("Epoch Completed")
    txt_file.close()
train_model()