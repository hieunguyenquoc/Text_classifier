def remove_stopwords(strings):
    print("strings: " + strings)
    strings = strings.split()
    f = open('stopwords.txt', 'r', encoding="utf-8")  
    stopwords = f.readlines()
    stop_words = [s.replace("\n", '') for s in stopwords]
    #print("mid: ", stop_words)
    doc_words = []
    #### YOUR CODE HERE ####
    
    for word in strings:
        if word not in stop_words:
            doc_words.append(word)

    #### END YOUR CODE #####
    doc_str = ' '.join(doc_words).strip()
    return doc_str