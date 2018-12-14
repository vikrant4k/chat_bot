import pickle
import torch
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors


#INITIALIZATIONS ------------------------------------------------------------------

path = "model_movie.pkl"
with open("model_movie.pkl", "rb") as f:
    model_movie = pickle.load(f)

with open('w2i_movie_names.json') as f:
    w2i_movies = json.load(f)

with open('i2w_movie_names.json') as f:
    i2w_movies = json.load(f)
with open('movie_data.pkl', 'rb') as f:
    movie_data = pickle.load(f)

# LOAD MOVIE EMBEDDINGS AND NECESSARY VOVABULARIES (and some preprocess)

model = torch.load(path, map_location='cpu')
print(len(model))
embeds = model['model_state_dict']['movie_embedding.weight']
print(len(embeds), len(w2i_movies), len(i2w_movies))  # all same lengths to be correct

embed_np = []
for em in embeds:
    embed_np.append(em.numpy().reshape(-1))

print(embed_np[0], type(embed_np[0]), embed_np[0].shape)

# remove uknowwn
names = set(w2i_movies.keys())
print(len(names))
names.remove('unknown')
len(names)

# create dic IMDB:name
movie_imdb2name = dict()
print(len(movie_data))
x = 0
for m in movie_data:
    if '<p>' not in m:
        movie_imdb2name[m] = movie_data[m].movie_name
len(movie_imdb2name)

# remove uknown from the embeddings
embedings = []
for m in movie_imdb2name:
    embed_id = w2i_movies[m]
    #     print(m,embed_id,movie_imdb2name[m])
    embedings.append(embed_np[embed_id])

#------------------------------------------------------------------------------------------------


#FUNCTIONS-----------

def k_nearest_neighbors(k, embedings):
    k += 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    nbrs.fit(embedings)
    distances, indices = nbrs.kneighbors(embedings)
    return distances, indices


def return_similar_movie_obj(movie_id):
    curr_movie_index = w2i_movies[movie_id] - 1  # (-1 because is stars from 1)
    curr_name = movie_imdb2name[movie_id]

    distances, indices = k_nearest_neighbors(3, embedings)

    closest_movie_index = indices[curr_movie_index][1] + 1
    closest_movie_id = i2w_movies[str(closest_movie_index)]
    closest_movie_name = movie_imdb2name[closest_movie_id]

    print(curr_name, ' (', movie_id, ') matches with ', closest_movie_name, ' (', closest_movie_id, ' )')

    for m in movie_data:
        if '<p>' not in m:
            if m == 'closest_movie_id':
                return movie_data[m]


closest_movie_object = return_similar_movie_obj('tt0061452')