from model import Model
import json
import pickle
import torch

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
    for i in range(0,len(sent_arr)):
        sent_indx[i]=w2i[sent_arr[i]]
    return sent_indx
w2i,i2w=load_index_files()
movie_data=load_movie_data()
model=None

def train_model():
    model=Model(512,len(w2i))
    for epoch in range(200):
        for data in movie_data:
            model.encoder.zero_grad()
            model.encoder.hidden = model.encoder.init_hidden()
            movie=movie_data[data]
            chats=movie.chat
            for chat in chats:
                for i in range(0,len(chat.chat),2):
                    encoder_sentence=chat.chat[i]
                    decoder_sentence=chat.chat[i+1]
                    enc_sent_indx=convert_sentence_to_index(encoder_sentence)
                    dec_sent_index=convert_sentence_to_index(decoder_sentence)
                    model.forward(enc_sent_indx)


train_model()