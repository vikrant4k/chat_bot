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

    def __init__(self,chat_id,chat):
        self.chat_id=chat_id
        self.chat=chat
