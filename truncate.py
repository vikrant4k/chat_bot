def truncate(plot, comments, reviews, max_length):
    
    reviews_token = [c.split(" ") for c in reviews]
    reviews_token = [j for i in reviews_token for j in i]
    len_R = len(reviews_token)
    
    comments_token = [c.split(" ") for c in comments]
    comments_token = [j for i in comments_token for j in i]
    len_C = len(comments_token)
    
    
    plot = plot[0].split(" ")
    len_P = len(plot)
    
    
    p_ratio = (len_P / (len_P + len_R + len_C))*max_length
    r_ratio = (len_R / (len_P + len_R + len_C))*max_length
    c_ratio = (len_C / (len_P + len_R + len_C))*max_length
    
    plot = plot[:int(p_ratio)]
    comments = comments_token[:int(c_ratio)]
    reviews = reviews_token[:int(r_ratio)]
    
    knowledge_base = plot + comments + reviews
    sent=""
    for data in knowledge_base:
        sent+=data+" "
    #print(plot,"\n\n")
    #print(comments, "\n\n")
    #print(reviews, "\n\n")
    lis=[]
    lis.append(sent)
    return lis
    