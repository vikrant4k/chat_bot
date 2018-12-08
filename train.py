from embeddingNetwork import Encoder
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
        w2i = json.load(f)
    with open('i2w.json') as f:
        i2w = json.load(f)
    with open('w_freq.json') as f:
        w_freq = json.load(f)
    return w2i, i2w, w_freq


def create_vocab_distributions():
    max_val = 0
    for key in i2w:
        temp_val = int(key)
        if (max_val < temp_val):
            max_val = temp_val
    #prob_vocab = torch.zeros(max_val + 1, requires_grad=False, device="cuda:0")
    prob_vocab = torch.zeros(max_val + 1, requires_grad=False)
    total_count = 0
    for key in w_freq:
        total_count = total_count + w_freq[key]
    for key in i2w:
        index = int(key)
        if (key == "1" or key == "0"):
            prob_vocab[index] = 1 / total_count
        else:
            prob_vocab[index] = w_freq[i2w[key]] / total_count
    return prob_vocab


def load_movie_data():
    with open('movie_data.pkl', 'rb') as input:
        movie_data = pickle.load(input)
    return movie_data


def convert_knowledge(knowledge):
    know_data = []
    le = 0
    if (type(knowledge) == list):

        for sentence in knowledge:
            data = '<SOS> ' + sentence + ' <EOS>'
            indx_data = convert_sentence_to_index(data)
            le += indx_data.shape[0]
            know_data.append(indx_data)
        index_data = torch.zeros(le, dtype=torch.long)
        le = 0
        for data in know_data:
            for i in range(0, data.shape[0]):
                index_data[le] = data[i]
                le += 1
    else:
        data = '<SOS> ' + knowledge + ' <EOS>'
        index_data = convert_sentence_to_index(data)
        index_data = index_data
        ##know_data.append(index_data)
    return index_data


def convert_sentence_to_index(sentence):
    sent_arr = sentence.split()
    sent_indx = torch.zeros(len(sent_arr), dtype=torch.long)
    sent_indx = sent_indx
    for i in range(0, len(sent_arr)):
        sent_indx[i] = w2i[sent_arr[i]]
    return sent_indx


def churn_review(review):
    for i in range(0, len(review)):
        str_arr = review[i].split()
        if (len(str_arr) > 390):
            str = ''
            for j in range(0, 390):
                if (j == 0):
                    str = str_arr[0]
                else:
                    str = str + ' ' + str_arr[j]
            review[i] = str
    return review


def churn_comments(comments):
    if (len(comments) > 21):
        return comments[:21]
    return comments


def load_movie_indicies():
    with open('w2i_movie_names.json') as f:
        w2i_movies = json.load(f)
    with open('i2w_movie_names.json') as f:
        i2w_movies = json.load(f)
    return w2i_movies, i2w_movies

w2i,i2w,w_freq=load_index_files()
w2i_movies, i2w_movies= load_movie_indicies()
movie_data=load_movie_data()
prob_vocab=create_vocab_distributions()
model=None

count = 0
"""
count=0
avg=0
for data in movie_data:
    count=count+1
    movie=movie_data[data]
    avg+=(len(movie.comments))
print(avg/count)
"""


def save_model(epoch, loss, optimizer, model):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, "model.pkl")


def load_model(max_val):
    model = Encoder(256, 64,  max_val + 1, prob_vocab)
    optimizer = optim.Adam(model.parameters())

    checkpoint = torch.load("model.pkl")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train_model():
    model_exist = False
    lamb = 1e-4
    prob = 0.6
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    txt_file = open(st, "w")
    start_sent = '<SOS>'
    start_index = convert_sentence_to_index(start_sent)
    max_val = 0
    for key in i2w:
        temp_val = int(key)
        if (max_val < temp_val):
            max_val = temp_val
    if (model_exist):
        model, optimizer, epoch, loss = load_model(max_val)
        #model.cuda()
        optimizer = optim.Adam(model.parameters())
    else:
        no_of_movies = get_max_value(i2w_movies)
        model = Encoder(256, 64, max_val + 1, no_of_movies)
        #model.cuda()
        optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    count = 0
    chats_complted = 0
    for epoch in range(200):
        for data in movie_data:

            # TODO ALSO FOR DECODER AND OTHER ENCODERS?
            movie = movie_data[data]
            plot = movie.plot
            comment = movie.comments
            imdb_id = movie.imdb_id
            review = movie.review

            if '<p>' not in imdb_id:
                imdb_id = torch.tensor(w2i_movies[imdb_id])

                plot_indexed = convert_knowledge(plot)
                comment_indexed = convert_knowledge(comment)
                review_indexed = convert_knowledge([item for sublist in review for item in sublist])
                # knowledge_base = torch.cat((torch.cat((plot_indexed,comment_indexed)), review_indexed))

                knowledge_base = [plot_indexed, comment_indexed, review_indexed]

                # negative sampling of 10
                choice = np.random.choice(no_of_movies, 10)

                output, lstm_out_kb, lstm_norm = model.forward(imdb_id, knowledge_base)

                target = torch.tensor(1, dtype=torch.float)

                loss = criterion(output.squeeze(), target)

                choice = torch.from_numpy(choice).long()
                neg_embeds = model.movie_embedding(choice)

                neg_samples_losses = 0

                neg_target = torch.tensor(-1, dtype=torch.float)
                for ng in neg_embeds:
                    #print(ng)
                    network_out = torch.matmul(lstm_out_kb, ng.view(-1, 1))

                    lstm_embed_norm = (torch.sqrt(torch.matmul(ng.view(1, -1), ng.view(-1,1)))).detach()

                    network_out = network_out / (lstm_norm * lstm_embed_norm)

                    neg_loss = criterion(network_out.squeeze(), neg_target)

                    neg_samples_losses += neg_loss

                print(output)
                optimizer.zero_grad()
                loss += neg_samples_losses
                loss.backward()
                optimizer.step()


        torch.save(model.state_dict(), "model/")

def get_max_value(dict):
    max_val = 0
    for key in dict:
        temp_val = int(key)
        if (max_val < temp_val):
            max_val = temp_val
    return max_val

train_model()