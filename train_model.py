from model_extra import Model
import json
import pickle
import torch

import torch.nn as nn
import torch.nn.functional as F
import datetime
import time
import torch.optim as optim

def load_index_files():
    with open('w2i.json') as f:
         w2i= json.load(f)
    with open('i2w.json') as f:
         i2w= json.load(f)
    return w2i, i2w

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

w2i,i2w=load_index_files()
movie_data=load_movie_data()
model=None

def train_model():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    txt_file=open(st,"w")
    start_sent='<SOS>'
    start_index=convert_sentence_to_index(start_sent)
    model=Model(512,len(w2i))
    model.cuda()
    lis=model.parameters()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        for data in movie_data:
             #TODO ALSO FOR DECODER AND OTHER ENCODERS?
            movie=movie_data[data]
            chats=movie.chat

            #just the plot
            #plot_sent_indx = convert_sentence_to_index(movie.plot)

            for chat in chats:
                for i in range(0,len(chat.chat),2):
                    if((i+1)<len(chat.chat)):
                        encoder_sentence = chat.chat[i]  # TODO chat history
                        decoder_sentence = chat.chat[i + 1]

                        encoder_sentence = '<SOS> ' + encoder_sentence + ' <EOS>'
                        decoder_sentence = decoder_sentence + ' <EOS>'
                        enc_sent_indx = convert_sentence_to_index(encoder_sentence)
                        dec_sent_index = convert_sentence_to_index(decoder_sentence)
                        if(len(dec_sent_index)<400 and len(dec_sent_index)>2):
                            output,coverage,current_attention = model.forward(enc_sent_indx, dec_sent_index, start_index,
                                                   True)  # , plot_sent_indx)
                            ##print(len(output))

                            output_text = ""
                            att_sum=torch.zeros(coverage.shape,device="cuda:0")
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
                            print(loss.item() / len(dec_sent_index))
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