def preprocess_input(text):
    # Apply case folding
    text = case_folding(text)
    # Tokenize the input
    tokens = tokenize(text)
    # Normalize tokens
    tokens = normalize(tokens)
    # Remove stopwords
    tokens = stopword(tokens)
    # Stem tokens
    tokens = stemming(tokens)
    return tokens

def transform_to_tfidf(input_text):
    # Preprocess the input
    tokens = preprocess_input(input_text)
    
    # Convert the preprocessed tokens to a TF-IDF vector
    vector = np.zeros((len(word_set),))
    for word in tokens:
        if word in word_set:
            tf = termfreq(tokens, word)
            idf = inverse_doc_freq(word)
            vector[index_dict[word]] = tf * idf
    return vector