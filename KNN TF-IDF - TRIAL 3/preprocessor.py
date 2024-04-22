import math
import re
from nltk.stem import PorterStemmer

class TFIDFVectorizer:
    def __init__(self):
        self.documents = {}
        self.documents_length = 0
        self.preprocessed_documents = {}
        self.tf = {}
        self.idf = {}
        self.idf_tokens = []
        self.tfidf = {}
        
    # Preprocess text
    def preprocess(self, documents):
        stemmer = PorterStemmer()
        preprocessed_documents = {}
        for index, document in documents.items():
            preprocessed_documents[index] = [stemmer.stem(word) for word in re.sub("[,.:)/(!'1234567890&-@]", '', str(document)).lower().split() if len(word) != 1]
        return preprocessed_documents

    # Calculate Term Frequency
    def term_frequency(self):
        for index, document in self.preprocessed_documents.items():
            term_frequency = {}
            document_length = len(document)
            for word in document:
                if word in term_frequency:
                    term_frequency[word] +=1
                else:
                    term_frequency[word] = 1
            self.tf[index] = {key: value/document_length for key, value in term_frequency.items()}

    # Calculate IDF
    def inverse_document_frequency(self):
        self.documents_length = len(self.documents)
        self.idf_tokens = set()
        
        for document in self.preprocessed_documents.values():
            self.idf_tokens.update(document)

        for word in self.idf_tokens:
            self.idf[word] = 0
            for document in self.preprocessed_documents.values():
                if word in document:
                    self.idf[word] += 1

        self.idf = {key: math.log(self.documents_length/value) for key, value in self.idf.items()}

    # Calculate TFIDF
    def calculate_tfidf(self):
        for index, document in self.tf.items():
            self.tfidf[index] = {key: value*self.idf[key] for key, value in document.items()}
            for word in self.idf_tokens:
                if word not in self.tfidf[index]:
                    self.tfidf[index][word] = 0

    # Calculate TF-IDF for test data
    def calculate_predict_tfidf(self, text = ""):
        text = self.preprocess({0:text})
        text = [word for word in text.values() for word in word if word in self.idf_tokens]
        predict_tf = {}
        predict_tfidf = {}

        for word in text:
            if word in predict_tf:
                predict_tf[word] +=1
            else:
                predict_tf[word] = 1

        predict_tf = {term: value/sum(predict_tf.values()) for term, value in predict_tf.items()}
        predict_tfidf = {word: value*self.idf[word] for word, value in predict_tf.items()}
        
        for word in self.idf_tokens:
            if word not in predict_tfidf.keys():
                predict_tfidf[word] = 0

        return predict_tfidf

    def train(self, documents):
        self.documents = documents
        self.preprocessed_documents = (self.preprocess(self.documents))
        self.term_frequency()
        self.inverse_document_frequency()
        self.calculate_tfidf()

        return self.tfidf