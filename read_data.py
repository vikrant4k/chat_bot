import json
from pprint import pprint
from movie_data import MovieData,Chat
import pickle
data=''
with open('../train_data.json') as f:
    data = json.load(f)
key=''
movie={}
tot=0

document_keys=['plot','review','comments']
other_keys=['movie_name','spans','chat']
w2i={}
i2w={}

w2i['<SOS>'] = 0
w2i['<EOS>'] = 1

i2w[0] = '<SOS>'
i2w[1] = '<EOS>'

index=2

def clean_data(data):
    for key in data:
        key['documents']['plot']=clear_quotes(key['documents']['plot'])
        key['documents']['review']=clear_quotes(key['documents']['review'])
        key['documents']['comments']=clear_quotes(key['documents']['comments'])
        key['spans']=clear_quotes(key['spans'])
        key['chat']=clear_quotes(key['chat'])
    return data


def clear_quotes(data):
    if isinstance(data,str):
        data=[data]

    for i in range(0, len(data)):
        val = data[i]
        val = val.replace('"', '')
        val = val.replace('', '')
        val = val.replace("'", "")
        val = val.replace("(", "")
        val = val.replace(")", "")
        val = val.replace("?", "")
        val = val.replace("!", "")
        val = val.replace(".", "")
        val = val.replace(",", "")
        val=val.replace("*","")
        val = val.replace("=", "")
        val=val.lower()
        data[i] = val
    return data

def create_word_to_ind(data):
    for key in data:
        for doc_key in document_keys:
            values=key['documents'][doc_key]
            helper_word_to_index(values)
        for other_key in other_keys:
            values=key[other_key]
            helper_word_to_index(values)

def helper_word_to_index(values):
    global index
    if(isinstance(values,str)):
        values=[values]
    for value in values:
        value_arr=value.split()
        for word in value_arr:
            if(word not in w2i):
                w2i[word]=index
                i2w[str(index)]=word
                index=index+1


def convert_data_to_obj(data):
    for key in data:
        imdb_id = key['imdb_id']
        if (imdb_id in movie):
            movie_data = movie[imdb_id]
            chat = key['chat']
            chat_id = key['chat_id']
            chat_data = Chat(chat_id, chat)
            movie_data.chat.append(chat_data)
            curr_review = key['documents']['review']
            if (curr_review not in movie_data.review):
                movie_data.review.append(curr_review)
            curr_fact_table = key['documents']['fact_table']
            for fact_key in curr_fact_table:
                if (fact_key not in movie_data.facts_table):
                    movie_data.facts_table[fact_key] = curr_fact_table[fact_key]
            curr_comments = key['documents']['comments']
            for comment in curr_comments:
                if (comment not in movie_data.comments):
                    movie_data.comments.append(comment)
            curr_span = key['spans']
            for span in curr_span:
                if (span not in movie_data.spans):
                    movie_data.spans.append(span)

        else:
            plot = key['documents']['plot']
            review = []
            review.append(key['documents']['review'])
            fact_table = key['documents']['fact_table']
            comments = key['documents']['comments']
            movie_name = key['movie_name']
            spans = key['spans']
            labels = key['labels']
            if ("chat" in key):
                chat = key["chat"]
            else:
                chat = None
            movie_data = MovieData(movie_name, imdb_id, plot, review, fact_table, comments, spans, labels, chat,
                                   key['chat_id'])
            movie[imdb_id] = movie_data
    return movie

data=clean_data(data)
create_word_to_ind(data)
convert_data_to_obj(data)
with open('w2i.json', 'w') as fp:
    json.dump(w2i, fp)
with open('i2w.json', 'w') as fp:
    json.dump(i2w, fp)
with open('movie_data.pkl', 'wb') as output:
     pickle.dump(movie, output, pickle.HIGHEST_PROTOCOL)
##print(key['documents']['plot'])
##print(key['documents']['review'])
##print(key['documents']['fact_table'])
##print(key['documents']['comments']) array
##print(key['movie_name'])
##print(key['spans'])  array
##print(key['labels'])   array
##print(key['imdb_id'])
##print(key['chat_id'])
##print(key['chat']) array