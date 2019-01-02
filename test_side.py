from model_side_know_test import Model
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
from pick_similar_sentences import get_similar_movie_responses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1


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
    prob_vocab = torch.zeros(max_val + 1, requires_grad=False, device=device)
    total_count = 0
    for key in w_freq:
        total_count = total_count + w_freq[key]
    for key in i2w:
        index = int(key)
        if (key == "1" or key == "2" or key == "0"):
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
        index_data = torch.zeros(le, dtype=torch.long).to(device)
        le = 0
        for data in know_data:
            for i in range(0, data.shape[0]):
                index_data[le] = data[i]
                le += 1
    else:
        data = '<SOS> ' + knowledge + ' <EOS>'
        index_data = convert_sentence_to_index(data)
        index_data = index_data.to(device)
        ##know_data.append(index_data)
    return index_data


def convert_sentence_to_index(sentence):
    sent_arr = sentence.split()
    sent_indx = torch.zeros(len(sent_arr), dtype=torch.long)
    sent_indx = sent_indx.to(device)
    for i in range(0, len(sent_arr)):
        sent_indx[i] = w2i[sent_arr[i]]
    return sent_indx


def convert2(sentence):
    sent_arr = sentence.split()
    # sent_indx=torch.zeros(len(sent_arr),dtype=torch.long)
    # sent_indx=sent_indx.to(device)
    sent_indx = []
    for i in range(0, len(sent_arr)):
        sent_indx.append(w2i[sent_arr[i]])
    return sent_indx


def churn_review(review):
    for i in range(0, len(review)):
        str_arr = review[i].split()
        if (len(str_arr) > 200):
            str = ''
            for j in range(0, 200):
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


w2i, i2w, w_freq = load_index_files()
movie_data = load_movie_data()
count = 0
prob_vocab = create_vocab_distributions()
model = None
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
    model = Model(256, max_val + 1, prob_vocab)
    optimizer = optim.Adam(model.parameters())

    checkpoint = torch.load("model.pkl", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def minibatch(data1, data2):
    for i in range(0, len(data1), batch_size):
        yield data1[i:i + batch_size], data2[i:i + batch_size]


def preprocess(data1, data2, PAD=0):
    enc = [convert2(seq) for seq in data1]
    dec = [convert2(seq) for seq in data2]

    max_enc = max(map(len, enc))
    max_dec = max(map(len, dec))
    # '<SOS> =1','<EOS>=2'
    seq_enc = [[1] + seq + [2] + [PAD] * (max_enc - len(seq)) for seq in enc]
    seq_dec = [seq + [2] + [PAD] * (max_dec - len(seq)) for seq in dec]
    ##seq_enc = [[1] + seq + [2]   for seq in enc]
    ##seq_dec = [seq + [2] for seq in dec]
    # print(seq_dec, seq_enc)
    seq_enc, seq_dec = np.array(seq_enc), np.array(seq_dec)
    ##print(seq_enc.shape, seq_dec.shape)
    return np.array(seq_enc), np.array(seq_dec)


def unique(tensor1d):
    t, idx = np.unique(tensor1d.numpy(), return_inverse=True)
    return torch.from_numpy(t), torch.from_numpy(idx)


def test_side():
    model_exist = True
    lamb = 1e-4
    prob = 0.6
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    txt_file = open("a.txt", "w")
    start_sent = '<SOS>'
    start_index = convert_sentence_to_index(start_sent)
    max_val = 0
    for key in i2w:
        temp_val = int(key)
        if (max_val < temp_val):
            max_val = temp_val
    if (model_exist):
        model, optimizer, epoch, loss = load_model(max_val)
        model.to(device)
        optimizer = optim.Adam(model.parameters())
    else:
        model = Model(256, max_val + 1, prob_vocab)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    count = 0
    chats_complted = 0
    with torch.no_grad():
        for data in movie_data:
            tot_loss = 0
            count = count + 1
            if (count >=0):
                movie = movie_data[data]
                chats = movie.chat
                movie_id=movie.imdb_id
                plot = movie.plot
                review = movie.review
                comments = churn_comments(movie.comments)
                review = churn_review(review)
                plot_sent_indx_arr = convert_knowledge(plot)

                review_sent_indx_arr = convert_knowledge(review)
                comment_sent_indx_arr = convert_knowledge(comments)

                tot_know_base = torch.cat((plot_sent_indx_arr, review_sent_indx_arr, comment_sent_indx_arr))
                uniq_indxs, inverse_indices = torch.unique(tot_know_base, return_inverse=True)
                indx_dic = {}
                for u in range(0, len(uniq_indxs)):
                    val = uniq_indxs[u].item()
                    for indy in range(0, len(tot_know_base)):
                        if (val == tot_know_base[indy]):
                            if (val in indx_dic):
                                indx_dic[val].append(indy)
                            else:
                                indx_dic[val] = []
                                indx_dic[val].append(indy)

                # just the plot
                # model.knowledge.forward(plot_sent_indx)
                tot_loss = 0
                # if (len(comments) > 0 and len(review) > 0):
                encoder_in = []
                decoder_ou = []
                for chat in chats:
                    deq = deque(maxlen=2)
                    talk = chat.chat

                    if len(chat.chat) % 2 != 0 and len(chat.chat) > 2:
                        talk = talk.pop()

                    encoder_in = talk[0]
                    decoder_ou = talk[1]
                    ##encoder_in.extend(talk[0::1])
                    ##decoder_ou.extend(talk[1::2])

                    for enc, dec in minibatch(encoder_in, decoder_ou):
                        last_index = enc[0].rfind("<EOS>")
                        if(last_index<0):
                            enc_sent=enc[0]
                        else:
                            enc_sent=enc[0][last_index+6:]

                        enc_sent_arr=enc_sent.split()
                        similar_reply_sent=get_similar_movie_responses(movie_id,1,enc_sent_arr)
                        similar_reply_sent=similar_reply_sent[0]
                        similar_reply_sent=similar_reply_sent.replace("'","")
                        ##print(similar_reply)
                        ##similar_reply_sent=" ".join(str(x) for x in similar_reply[0])
                        if(len(similar_reply_sent)==0):
                            similar_reply_sent="<PAD>"
                        similiar_sent_indx=convert2(similar_reply_sent)

                        ##print(enc_sent_arr)
                        if (len(enc) < batch_size):
                            diff = batch_size - len(enc)
                            for k in range(diff):
                                enc.append("<PAD>")
                                dec.append("<PAD>")
                        enc_lengths = []
                        dec_lengths = []
                        for ea in enc:
                            enc_lengths.append(len(ea.split()))
                        for da in dec:
                            dec_lengths.append(len(da.split()))
                        kj = 0
                        for lo in dec_lengths:
                            if (lo > 40):
                                kj = 1
                        if (kj == 1):
                            kj = 0
                            continue
                        input_sent, dec_sent_index, = preprocess(enc, dec)
                        enc_lengths = torch.tensor(enc_lengths).long().to(device)
                        dec_lengths = torch.tensor(dec_lengths).long().to(device)
                        input_sent = torch.tensor(input_sent).long().to(device)
                        dec_sent_index = torch.tensor(dec_sent_index).long().to(device)
                        similiar_sent_indx=torch.tensor(similiar_sent_indx).long().to(device)
                        ##dec_sent_index.requires_grad=False
                        ##print(input_sent.shape,dec_sent_index.shape)
                        ##print(enc_lengths,dec_lengths)
                        know_hidd = model.forward_knowledge_movie(plot_sent_indx_arr, review_sent_indx_arr,
                                                                  comment_sent_indx_arr,similiar_sent_indx)
                        isRely = True
                        start_index = torch.tensor([1]).repeat(batch_size, 1).long().to(device)
                        """
                        output, coverage, current_attention = model.forward(input_sent, dec_sent_index,
                                                                                start_index,
                                                                                True, know_hidd,
                                                                                isRely, plot_sent_indx_arr,
                                                                                review_sent_indx_arr,
                                                                                comment_sent_indx_arr, enc_lengths,
                                                                                dec_lengths,(uniq_indxs,indx_dic))
                        """
                        output, coverage, current_attention = model.forward(input_sent, dec_sent_index,
                                                                            start_index,
                                                                            False, know_hidd,
                                                                            isRely)
                        org_word_index = dec_sent_index.clone()
                        max_prob_index = torch.argmax(output, dim=1)
                        batch_sentences = []
                        ##print(dec_lengths)
                        """
                        for b in range(max_prob_index.shape[0]):
                            sentence_str = ''
                            actual_len = dec_lengths[b]
                            ##print(actual_len)
                            for w in range(actual_len):
                                ##word=i2w[str((max_prob_index[w]).item())]
                                word = i2w[str((max_prob_index[b][w]).item())]
                                sentence_str += word+' '
                        """
                        sentence_str = ''
                        for b in range(max_prob_index.shape[0]):
                            ##print(actual_len)
                            word = i2w[str((max_prob_index[b]).item())]
                            sentence_str += word + ' '

                        batch_sentences.append(sentence_str)
                        sent_index = 0
                        ##print(len(batch_sentences))
                        for sentence in batch_sentences:
                            ##print(sentence)
                            if (len(sentence) > 10):
                                index = enc[sent_index].rfind("<EOS>")
                                if (index == -1):
                                    sent = enc[sent_index]
                                else:
                                    sent = enc[sent_index][index + 6:]
                                txt_file.write("Speaker 1:" + sent)
                                txt_file.write("\n")
                                txt_file.write("Model: " + sentence.encode('utf-8').decode('utf-8'))
                                txt_file.write("\n")
                                txt_file.write("Speaker 2:" + dec[sent_index])
                                txt_file.write("\n")
                                txt_file.flush()
                            sent_index += 1

                        org_word_index = dec_sent_index.clone()
                        org_word_index = org_word_index.squeeze(0)

            txt_file.write("Movie Completed " + str(count))
            txt_file.write("\n")
    txt_file.close()


test_side()
