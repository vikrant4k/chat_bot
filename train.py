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
from random import randint
from nltk.corpus import stopwords
from random import shuffle

stopwords_list = stopwords.words('english')

device_type="cuda:0"
num_movies=10
def load_index_files():
    with open('w2i.json') as f:
        w2i = json.load(f)
    with open('i2w.json') as f:
        i2w = json.load(f)
    with open('w_freq.json') as f:
        w_freq = json.load(f)
    with open('w2i_review_comments_plot.json') as f:
        w2i_rpc = json.load(f)
    with open('i2w_review_comments_plot.json') as f:
        i2w_rpc = json.load(f)
    return w2i, i2w, w_freq, w2i_rpc, i2w_rpc


def create_vocab_distributions():
    max_val = 0
    for key in i2w:
        temp_val = int(key)
        if (max_val < temp_val):
            max_val = temp_val
    #prob_vocab = torch.zeros(max_val + 1, requires_grad=False, device="cuda:0")
    prob_vocab = torch.zeros(max_val + 1, requires_grad=False,device=device_type)
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
        # index_data = torch.zeros(le, dtype=torch.long,device=device_type)
        index_data = know_data
        le = 0
        # for data in know_data:
        #     for i in range(0, data.shape[0]):
        #         index_data[le] = data[i]
        #         le += 1
    else:
        data = '<SOS> ' + knowledge + ' <EOS>'
        index_data = convert_sentence_to_index(data)
        index_data = index_data
        ##know_data.append(index_data)
    return index_data


def convert_sentence_to_index(sentence):
    sent_arr = sentence.split()
    sent_arr = [word for word in sent_arr if word not in stopwords_list]
    sent_indx = torch.zeros(len(sent_arr), dtype=torch.long,device=device_type)
    sent_indx = sent_indx
    for i in range(0, len(sent_arr)):
        sent_indx[i] = w2i_rpc[sent_arr[i]]
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

w2i,i2w,w_freq, w2i_rpc, i2w_rpc=load_index_files()
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
    }, "model_movie.pkl")


def load_model(max_val):
    model = Encoder(256, 64,  max_val + 1, prob_vocab)
    optimizer = optim.Adam(model.parameters())

    checkpoint = torch.load("model_movie.pkl")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train_model():
    model_exist = False
    ##lamb = 1e-4
    ##prob = 0.6
    ##ts = time.time()
    ##st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    ##txt_file = open(st, "w")
    start_sent = '<SOS>'
    ##start_index = convert_sentence_to_index(start_sent)
    max_val = 0
    for key in i2w_rpc:
        temp_val = int(key)
        if (max_val < temp_val):
            max_val = temp_val
    if (model_exist):
        model, optimizer, epoch, loss = load_model(max_val)
        model.cuda()
        optimizer = optim.Adam(model.parameters())
    else:
        no_of_movies = get_max_value(i2w_movies)
        model = Encoder(256, 64, max_val + 1, no_of_movies)
        model.cuda()
        optimizer = optim.Adam(model.parameters())
    criterion = nn.CosineEmbeddingLoss()
    count = 0
    for epoch in range(200):
        shuffle(movie_data)
        for data in movie_data:
            count+=1
            # TODO ALSO FOR DECODER AND OTHER ENCODERS?
            # movie = movie_data[data]
            # comment = movie.comments
            imdb_id = data[0]
            plot = data[1]
            # review = movie.review
            # comment = churn_comments(comment)
            # review = churn_review(review)

            if '<p>' not in imdb_id:
                ##imdb_id = torch.tensor(w2i_movies[imdb_id],device=device_type)
                choice = np.random.choice(no_of_movies, num_movies)
                choice = torch.from_numpy(choice).long()
                choice = choice.to(device_type)
                index=randint(0,num_movies-1)
                choice[index]=w2i_movies[imdb_id]
                plot_indexed = convert_knowledge(plot)
                # comment_indexed = convert_knowledge(comment)
                # review_indexed = convert_knowledge([item for item in review])
                # knowledge_base = torch.cat((torch.cat((plot_indexed,comment_indexed)), review_indexed))
                # knowledge_base = [plot_indexed, comment_indexed, review_indexed]

                # negative sampling of 10
                model_vector, movie_vector = model.forward(choice, plot_indexed, num_movies)

                target = torch.tensor(-1, dtype=torch.float, device=device_type, requires_grad=False)
                target = target.repeat(num_movies)
                target[index] = 1
                loss = criterion(model_vector, movie_vector, target)

                print(count, loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("********************** epoch finished **********************")
        save_model(epoch, loss, optimizer, model)

def get_max_value(dict):
    max_val = 0
    for key in dict:
        temp_val = int(key)
        if (max_val < temp_val):
            max_val = temp_val
    return max_val

train_model()