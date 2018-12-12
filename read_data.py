import json
from pprint import pprint
from movie_data import MovieData,Chat
import pickle
import nltk.tokenize.punkt
data=''
with open('/home/juan/data/main_data/train_data.json') as f:
    data = json.load(f)
key=''
movie={}
sentence_id=[]
tot=0

tokenizer = nltk.data.load('/home/juan/Downloads/punkt/english.pickle')
document_keys=['plot','review','comments']
other_keys=['movie_name','spans','chat']
w2i={}
i2w={}
dic_freq={}

w2i['<SOS>'] = 0
w2i['<EOS>'] = 1
w2i['unknown']=2

i2w[0] = '<SOS>'
i2w[1] = '<EOS>'
i2w[2]='unknown'

index=3

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
        data= tokenizer.tokenize(data)

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
        val = val.replace("-:", "")
        val = val.replace(":-", "")
        val = val.replace("r#16", "")
        val = val.replace("wham", "")
        val = val.replace(":", "")
        val = val.replace("http//www", "")
        val=val.lower()
        data[i] = val
    return data

def create_word_frequency(data):
    for key in data:
        for doc_key in document_keys:
            values=key['documents'][doc_key]
            helper_word_to_freq(values)
        for other_key in other_keys:
            values=key[other_key]
            helper_word_to_freq(values)

def helper_word_to_freq(values):
    if (isinstance(values, str)):
        values = [values]
    for value in values:
        value_arr = value.split()
        for word in value_arr:
            if (word not in dic_freq):
                dic_freq[word]=1
            else:
                dic_freq[word]=dic_freq[word]+1




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
                if(dic_freq[word]>15):
                    w2i[word] = index
                    i2w[str(index)] = word
                    index = index + 1
                else:
                    w2i[word] = 2


def convert_data_to_obj(data):
    all_ids = []
    for key in data:
        imdb_id = key['imdb_id']
        if (imdb_id in movie):
            movie_data = movie[imdb_id]
            #print(movie_data)
            chat = key['chat']
            #print(chat)
            chat_id = key['chat_id']
            chat_data = Chat(chat_id, chat)
            movie_data.chat.append(chat_data)

            curr_review = key['documents']['review']
            for reviews in curr_review:
                if (reviews not in movie_data.review):
                    movie_data.review.append(reviews)
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
            if all('' == s or s.isspace() for s in plot):
                continue
            review=(key['documents']['review'])
            if all('' == s or s.isspace() for s in review):
                continue

            fact_table = key['documents']['fact_table']
            if all('' == s or s.isspace()  for s in fact_table):
                continue

            comments = key['documents']['comments']
            if all('' == s or s.isspace() for s in comments):
                continue

            movie_name = key['movie_name']
            if all('' == s or s.isspace() or s.isdigit() for s in movie_name):
                continue

            spans = key['spans']
            if all('' == s or s.isspace() for s in spans):
                continue

            labels = key['labels']
            if all('' == s  for s in labels):
                continue

            if ("chat" in key):
                chat = key["chat"]
            else:
                chat = None
            movie_data = MovieData(movie_name, imdb_id, plot, review, fact_table, comments, spans, labels, chat,
                                   key['chat_id'])
            movie[imdb_id] = movie_data
    return movie

data=clean_data(data)
create_word_frequency(data)
create_word_to_ind(data)
convert_data_to_obj(data)
for element in movie:
    for sentence in movie[element].plot:
        if sentence != '{}' and sentence != 'r16':
            sentence_id.append((element, sentence))
    for sentence in movie[element].comments:
        if sentence != '{}' and sentence != 'r16':
            sentence_id.append((element, sentence))
    for sentence in movie[element].review:
        if sentence != '{}' and sentence != 'r16':
            sentence_id.append((element, sentence))

with open('w_freq.json', 'w') as fp:
    json.dump(dic_freq, fp)
with open('w2i_review_comments_plot.json', 'w') as fp:
    json.dump(w2i, fp)
with open('i2w_review_comments_plot.json', 'w') as fp:
    json.dump(i2w, fp)
with open('movie_data.pkl', 'wb') as output:
     pickle.dump(sentence_id, output, pickle.HIGHEST_PROTOCOL)
##print(key['documents']['review'])
##print(key['documents']['fact_table'])
##print(key['documents']['comments']) array
##print(key['movie_name'])
##print(key['spans'])  array
##print(key['labels'])   array
##print(key['imdb_id'])
##print(key['chat_id'])
##print(key['chat']) array