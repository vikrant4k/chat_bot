import pickle
import torch
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

with open('w2i_movie_names.json') as f:
    w2i_movies = json.load(f)

with open('i2w_movie_names.json') as f:
    i2w_movies = json.load(f)

with open('movie_data_separate.pkl', 'rb') as f:
    movie_data = pickle.load(f)

with open('neighbours.pkl', 'rb') as f:
    neighbours = pickle.load(f)

with open('w2i_review_comments_fact.json') as f:
    w2i_rpc = json.load(f)
with open('i2w_review_comments_fact.json') as f:
    i2w_rpc = json.load(f)

len(i2w_rpc.keys())
w2i_rpc['unknown']
stop_words = stopwords.words('english')
tknzr = TweetTokenizer()


def k_nearest_neighbors(k, embeddings):
    k += 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    return distances, indices


def get_average_utterance_embedding(utterance, embed_dim, stopwords, w2i, trained_word_embeddings):
    # obtain the average embedding for the whole utterance using the word embeddings
    # learned in the movie embedding training
    utterance_embedding = torch.zeros(embed_dim)

    count = 0
    for w in utterance:
        # skip stop words
        if w in stop_words or w in ['<SOS>', '<EOS>']:
            pass
        elif w in w2i:  # word in dictionary
            # print('word',w)
            word_em = trained_word_embeddings[w2i[w]]
            utterance_embedding += word_em
            count += 1
        else:
            word_em = trained_word_embeddings[w2i_rpc['unknown']]  # unk
            utterance_embedding += word_em
            count += 1

    # print(utterance_embedding, utterance_embedding/count, count)
    avg_utterances_embedding = utterance_embedding / count

    return avg_utterances_embedding


def get_similar_movie_responses(movie_id, n, model, w2i, i2w, utterance, neighbours, stop_words):
    '''
    w2i and i2w are vocabularies for plot-review-comments
    utterance is a tokenized sentence of strings
    returning a list of tokenized sentences of strings
    '''
    similar_movie_data = neighbours[movie_id]
    similar_movie_id = similar_movie_data.imdb_id
    print(similar_movie_id)
    similar_movie_chat = similar_movie_data.chat

    trained_word_embeddings = model['model_state_dict']['word_embedding.weight']
    # print(trained_word_embeddings.shape)
    embed_dim = trained_word_embeddings.shape[1]

    avg_utterance_embedding = get_average_utterance_embedding(utterance, embed_dim, stop_words, w2i, trained_word_embeddings)

    similar_responses = []

    all_chat_reps = []
    all_chat_reps.append(avg_utterance_embedding.numpy())
    all_chat_indices = [(-1, -1)]
    for c in range(len(similar_movie_chat)):
        chat = similar_movie_chat[c]
        enc = chat.encoder_chat
        dec = chat.decoder_chat

        for s in range(len(enc)):
            sent = enc[s]
            sent = tknzr.tokenize(sent)
            # print(sent)
            sent_avg_embedding = get_average_utterance_embedding(sent, embed_dim, stop_words, w2i, trained_word_embeddings)

            all_chat_reps.append(sent_avg_embedding.numpy())
            # chat index and then sentence index for speaker 1
            # so that we can get the related speaker 2 utterance
            all_chat_indices.append((c, s))

            # print(all_chat_reps[0])
    distances, indices = k_nearest_neighbors(n, all_chat_reps)
    print(indices)
    neighbours = indices[0]

    for n in neighbours:
        (c, s) = all_chat_indices[n]
        print(c, s)
        if c != -1:
            print(similar_movie_chat[c].encoder_chat[s])
            print(similar_movie_chat[c].decoder_chat[s])
            similar_responses.append(tknzr.tokenize(similar_movie_chat[c].decoder_chat[s]))

    return similar_responses


movie_id = 'tt0058150'
n = 5
utterance = ['which', 'is', 'your', 'favourite', 'character']  # speaker 1 utterance
model = torch.load('model_movie.pkl', map_location='cpu')
similar_responses = get_similar_movie_responses(movie_id, n, model, w2i_rpc, i2w_rpc, utterance, neighbours, stopwords)