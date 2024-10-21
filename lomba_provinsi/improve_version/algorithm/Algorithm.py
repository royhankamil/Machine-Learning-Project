# # menyiapkan library
import numpy as np
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

class Preprocessing:
    def __init__(self, word_set, index_dict, word_count, total_documents):
        self.word_set = word_set
        self.index_dict = index_dict
        self.word_count = word_count
        self.total_documents = total_documents
        self.norm_word = {"emg" : "memang","saiz" : "size","cino" : "cina","jowo" : "jawa","kite" : "kita","lg" : "lagi","aj" : "aja","yg" : "yang","pdhl" : "padahal","napa" : "kenapa","dh" : "sudah","udh" : "sudah","tdk" : "tidak","sm" : "sama","ga" : "tidak","bgt" : "banget","mentri" : "menteri","embantu" : "membantu","dlm" : "dalam","bdang" : "bidang","msh" : "masih","ampe" : "sampai","ky" : "kaya","nnya" : "tanya","krn" : "karena","jir" : "anjing","ajg" : "anjing","anjir" : "anjing","gak" : "tidak","ak" : "aku","dasr" : "dasar","lgsg" : "langsung","skrg" : "sekarang","gw" : "gua","w" : "gua","engga" : "tidak","dgn" : "dengan","orng" : "orang","org" : "orang","ni" : "ini","jgn" : "jangan","mbahas" : "bahas","krna" : "karena","ma" : "sama","sblm" : "sebelum","tp" : "tapi","sbg" : "sebagai","kl" : "kalau"}

    def case_folding(self, text):
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\n+', '', text)
        text = re.sub(r'\r+', '', text)
        text = re.sub(r'\d+', '', text)

        text = text.translate(str.maketrans("", "", string.punctuation))

        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"
                                u"\U0001F300-\U0001F5FF"
                                u"\U0001F680-\U0001F6FF"
                                u"\U00010000-\U0010ffff"
                                "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r"", text)

        return text.lower()
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def stopword(self, text):
        stopw = stopwords.words("indonesian")
        return [word for word in text if word not in stopw]
    
    def normalize(self, text):
        return [self.norm_word[word] if word in self.norm_word else word for word in text]
    
    def stemming(self, text):
        stemmer = StemmerFactory().create_stemmer()
        return [stemmer.stem(word) for word in text]
    
    def termfreq(self, document, word):
        N = len(document)
        occurance = len([token for token in document if token==word])
        return occurance/N
    
    def inverse_doc_freq(self, word):
        try:
            word_occurance = self.word_count[word] + 1
        except:
            word_occurance = 1
        return np.log(self.total_documents/word_occurance)
    
    def preprocess_input(self, text):
        # Apply case folding
        text = self.case_folding(text)
        # Tokenize the input
        tokens = self.tokenize(text)
        # Normalize tokens
        tokens = self.normalize(tokens)
        # Remove stopwords
        tokens = self.stopword(tokens)
        # Stem tokens
        tokens = self.stemming(tokens)
        return tokens
    
    def transform_to_tfidf(self,input_text):
        # Preprocess the input
        tokens = self.preprocess_input(input_text)
        
        # Convert the preprocessed tokens to a TF-IDF vector
        vector = np.zeros((len(self.word_set),))
        for word in tokens:
            if word in self.word_set:
                tf = self.termfreq(tokens, word)
                idf = self.inverse_doc_freq(word)
                vector[self.index_dict[word]] = tf * idf
        return vector
    

class KNN:
    def __init__(self, n_neighbors=3, weight="uniform", distance_metric="euclidean", p=2):
        self.n_neighbors = n_neighbors
        self.weight = weight
        self.distance_metric = distance_metric
        self.p = p  
        self.x = None
        self.y = None

    def fit(self, x, y):
        if len(x) != len(y):
            raise f"length of x is different with length of y, x = ({len(x)}) and y = ({len(y)})"
        
        self.x = np.array(x)
        self.y = np.array(y)

    def predict(self, x_predict):
        x_predict = np.array(x_predict)
        predicted = [self._predict(x) for x in x_predict]
        return np.array(predicted)

    def _predict(self, x):
        if self.distance_metric == "euclidean":
            distance = np.sqrt(np.sum((self.x - x) ** 2, axis=1))
        elif self.distance_metric == "manhattan":
            distance = np.sum(np.abs(self.x - x), axis=1)
        elif self.distance_metric == "minkowski":
            distance = np.sum(np.abs(self.x - x) ** self.p, axis=1) ** (1 / self.p)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        nearest = np.argpartition(distance, self.n_neighbors)[:self.n_neighbors]
        labels = [self.y[i] for i in nearest]

        if self.weight == "uniform":
            return np.bincount(labels).argmax()
        
        if self.weight == "distance":
            weight = [(1/(distance[i]+1e-10)) for i in nearest]
            return np.bincount(labels, weights=weight).argmax()
        
        raise ValueError("Can only use 'uniform' or 'distance' for weight")

    def accuracy(self, y_pred, y_true):
        return np.sum(y_pred == y_true) / len(y_true)
