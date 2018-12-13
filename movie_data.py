class MovieData:

    def __init__(self,movie_name,imdb_id,plot,review,facts_table,comments,spans,labels,chat,chat_id):
        self.movie_name=movie_name
        self.imdb_id=imdb_id
        self.plot=plot
        self.review=review
        self.facts_table=facts_table
        self.comments=comments
        self.spans=spans
        self.labels=labels
        self.chat=[]
        if(chat is not None):
          self.chat.append(Chat(chat_id,chat))

class Chat:

    def __init__(self,chat_id,chats):
        self.chat=[]
        if(len(chats)%2!=0):
            le=len(chats)-1
        else:
            le=len(chats)
        self.chat_id=chat_id
        self.encoder_chat=[]
        self.decoder_chat=[]
        try:
            for i in range(0, le, 2):
                if(i>=2):
                    self.encoder_chat.append(chats[i-2]+" "+chats[i-1]+" "+chats[i])
                else:
                    self.encoder_chat.append(chats[i])
                self.decoder_chat.append(chats[i + 1])
            self.chat.append(self.encoder_chat)
            self.chat.append(self.decoder_chat)
        except:
            print("Error")
