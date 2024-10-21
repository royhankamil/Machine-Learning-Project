# # menyiapkan library
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import string
# import re


# def case_folding(text):
#     text = re.sub(r'@[A-Za-z0-9_]+', '', text)
#     text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
#     text = re.sub(r'#\w+', '', text)
#     text = re.sub(r'https?://\S+', '', text)
#     text = re.sub(r'\n+', '', text)
#     text = re.sub(r'\r+', '', text)
#     text = re.sub(r'\d+', '', text)

#     text = text.translate(str.maketrans("", "", string.punctuation))

#     emoji_pattern = re.compile("["
#                                u"\U0001F600-\U0001F64F"
#                                u"\U0001F300-\U0001F5FF"
#                                u"\U0001F680-\U0001F6FF"
#                                u"\U00010000-\U0010ffff"
#                                "]+", flags=re.UNICODE)
#     text = emoji_pattern.sub(r"", text)

#     return text.lower()

# def tokenize(text):
#     return word_tokenize(text)

# norm_word = {
#     "emg" : "memang",
#     "saiz" : "size",
#     "cino" : "cina",
#     "jowo" : "jawa",
#     "kite" : "kita",
#     "lg" : "lagi",
#     "aj" : "aja",
#     "yg" : "yang",
#     "pdhl" : "padahal",
#     "napa" : "kenapa",
#     "dh" : "sudah",
#     "udh" : "sudah",
#     "tdk" : "tidak",
#     "sm" : "sama",
#     "ga" : "tidak",
#     "bgt" : "banget",
#     "mentri" : "menteri",
#     "embantu" : "membantu",
#     "dlm" : "dalam",
#     "bdang" : "bidang",
#     "msh" : "masih",
#     "ampe" : "sampai",
#     "ky" : "kaya",
#     "nnya" : "tanya",
#     "krn" : "karena",
#     "jir" : "anjing",
#     "ajg" : "anjing",
#     "anjir" : "anjing",
#     "gak" : "tidak",
#     "ak" : "aku",
#     "dasr" : "dasar",
#     "lgsg" : "langsung",
#     "skrg" : "sekarang",
#     "gw" : "gua",
#     "w" : "gua",
#     "engga" : "tidak",
#     "dgn" : "dengan",
#     "orng" : "orang",
#     "org" : "orang",
#     "ni" : "ini",
#     "jgn" : "jangan",
#     "mbahas" : "bahas",
#     "krna" : "karena",
#     "ma" : "sama",
#     "sblm" : "sebelum",
#     "tp" : "tapi",
#     "sbg" : "sebagai",
#     "kl" : "kalau"
# }

# # mengembalikan text yang sudah dinormalisasi 
# def normalize(text):
#     return [norm_word[word] if word in norm_word else word for word in text]


# # mengembalikan text yang sudah di stopwords
# def stopword(text):
#     stopw = stopwords.words("indonesian")
#     return [word for word in text if word not in stopw]


# def stemming(text):
#     stemmer = StemmerFactory().create_stemmer()
#     return [stemmer.stem(word) for word in text]


# def preprocess_input(text):
#     # Apply case folding
#     text = case_folding(text)
#     # Tokenize the input
#     tokens = tokenize(text)
#     # Normalize tokens
#     tokens = normalize(tokens)
#     # Remove stopwords
#     tokens = stopword(tokens)
#     # Stem tokens
#     tokens = stemming(tokens)
#     return tokens

# def transform_to_tfidf(input_text):
#     # Preprocess the input
#     tokens = preprocess_input(input_text)
    
#     # Convert the preprocessed tokens to a TF-IDF vector
#     vector = np.zeros((len(word_set),))
#     for word in tokens:
#         if word in word_set:
#             tf = termfreq(tokens, word)
#             idf = inverse_doc_freq(word)
#             vector[index_dict[word]] = tf * idf
#     return vector

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
