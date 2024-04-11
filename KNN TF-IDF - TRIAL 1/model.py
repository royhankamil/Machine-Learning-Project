import pandas as pd
from nltk.stem import PorterStemmer
import numpy as np

def Preprocessing(dataframe):
    dataframe = pd.DataFrame(dataframe)
    lower_cased_dataframe = dataframe.apply(lambda x: x.str.lower())
    no_punctuation_dataframe = lower_cased_dataframe.apply(lambda x: x.str.replace('[^\w\s]','', regex=True))
    tokenized_dataframe = no_punctuation_dataframe.iloc[:, 0].str.split()

    ps = PorterStemmer()

    stemmed_dataframe = []

    for document in tokenized_dataframe:
        stemmed_token = []
        for token in document:
            stemmed_token.append(ps.stem(token))
        stemmed_dataframe.append(stemmed_token)

    return stemmed_dataframe

def TermFrequencies(text_array = []):
    tf = {}
    for index, document in enumerate(text_array):
        matched_token = {}
        document_length = len(document)
        for token in document:
            if token not in matched_token:
                amount = 0
                for token_match in document:
                    if token_match == token:
                        amount+=1
                matched_token[token] = amount
        
        for key in matched_token:
            matched_token[key] /= document_length

        tf[index] = matched_token
    
    return tf

def InverseDocumentFrequencies(text_array = []):
    idf = {}
    terms_frequencies = {}
    possible_word = []
    text_length = len(text_array)
    for document in text_array:
        for word in document:
            if word not in possible_word:
                possible_word.append(word)
        
    for word in possible_word:
        terms_frequencies[word] = sum(1 for document in text_array if word in document)

    for term in terms_frequencies:
        terms_frequencies[term] = np.log(text_length/(terms_frequencies[term]))

    return terms_frequencies 

def tfidf_vectorizer(text_array, predicted_val = None):
    tf = TermFrequencies(text_array)
    idf = InverseDocumentFrequencies(text_array)
    tfidf = tf

    if predicted_val == None:
        for index, document in enumerate(tf.values()):
            for term in document:
                if idf[term] != 0:
                    tfidf[index][term] = document[term] / idf[term] 
                else:
                    tfidf[index][term] = 0
        return(pd.DataFrame(tfidf).transpose().fillna(0))
    else:
        for index, document in enumerate(tf.values()):
            for term in document:
                if idf[term] != 0:
                    tfidf[index][term] = document[term] / idf[term] 
                else:
                    tfidf[index][term] = 0
        return(pd.DataFrame(tfidf).transpose().fillna(0), predicted_val)

class KNN():
    def __init__(self, n_neighbors = 1):
        self.x_train = {}
        self.y_train = {}
        self.y_test = {}

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def voting(self):
        pass

    def predict(self, text):
        result = {}
        for document in self.x_train.items():
            print(document)

        
dataframe=pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN TF-IDF - TRIAL 1\my-test-data.csv")
preprocessed_dataframe = Preprocessing(dataframe["Review"])
preprocessed_dataframe_predict = ()
tfidf = tfidf_vectorizer(preprocessed_dataframe)
tfidf_predict = tfidf_vectorizer()