import pickle
import torch
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

# INITIALIZATIONS ------------------------------------------------------------------

path = "model_movie.pkl"
with open(path, "rb") as f:
    model_movie = pickle.load(f)

with open('w2i_movie_names.json') as f:
    w2i_movies = json.load(f)

with open('i2w_movie_names.json') as f:
    i2w_movies = json.load(f)

with open('movie_data_separate.pkl', 'rb') as f:
    movie_data = pickle.load(f)

# LOAD MOVIE EMBEDDINGS AND NECESSARY VOCABULARIES (and some preprocessing)
model = torch.load(path, map_location='cpu')
print(len(model))
embeds = model['model_state_dict']['movie_embedding.weight']
print(len(embeds), len(w2i_movies), len(i2w_movies))  # all same lengths to be correct

embed_np = []
for em in embeds:
    embed_np.append(em.numpy().reshape(-1))

print(embed_np[0], type(embed_np[0]), embed_np[0].shape)

# create dict IMDB:name
movie_imdb2name = dict()
print(len(movie_data))  # doesn't have unk movie

for m in movie_data:
    movie_imdb2name[m] = movie_data[m].movie_name

len(movie_imdb2name)

# remove unknown from the embeddings, convert into list
embeddings = []
for m in movie_imdb2name:
    embed_id = w2i_movies[m]
    # print(m,embed_id,movie_imdb2name[m])
    embeddings.append(embed_np[embed_id])


# ------------------------------------------------------------------------------------------------


# FUNCTIONS-----------

def k_nearest_neighbors(k, embeddings):
    k += 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    return distances, indices


def get_similar_movie_obj(movie_id):
    curr_movie_index = w2i_movies[movie_id] - 1  # (-1 because is starts from 1 since we removed unk)
    # print(curr_movie_index)
    curr_name = movie_imdb2name[movie_id]

    distances, indices = k_nearest_neighbors(3, embeddings)  # TODO THIS FINDS NEIGHBOURS OF ALL MOVIES EVERY TIME
    # print(indices)
    closest_movie_index = indices[curr_movie_index][1] + 1
    closest_movie_id = i2w_movies[str(closest_movie_index)]
    closest_movie_name = movie_imdb2name[closest_movie_id]

    # print(curr_name, ' (', movie_id, ') matches with ', closest_movie_name, ' (', closest_movie_id, ' )')

    # commented part is for cosine similarity

    # cur_em = embeddings[curr_movie_index]
    #
    # cosine_similarity = -10
    # closest = 0
    #
    # for em in range(len(embeddings)):
    #
    #     # embeddings are not checked for unit length
    #     sim = 1 - spatial.distance.cosine(cur_em, embeddings[em])
    #
    #     if sim > cosine_similarity:
    #         if closest == curr_movie_index:
    #             pass
    #         else:
    #             closest = em + 1
    #             cosine_similarity = sim
    #
    # print('closest', closest, i2w_movies[str(closest)], movie_imdb2name[i2w_movies[str(closest)]])

    for m in movie_data:
        if m == closest_movie_id:
            return movie_data[m]


# closest_movie_object = return_similar_movie_obj('tt0061452')
# print(closest_movie_object)

closest_dict = dict()
 
for m in movie_data:
    closest_dict[m] = get_similar_movie_obj(m)
# 
with open('neighbours.pkl', 'wb') as f:
    pickle.dump(closest_dict, f)

with open('neighbours.pkl', 'rb') as f:
    nbs = pickle.load(f)

with open('neigbours_list.txt', 'w') as f:
    for n in nbs:
        string_movies = movie_imdb2name[n] + '\t' + nbs[n].movie_name + '\n'
        f.write(string_movies)
