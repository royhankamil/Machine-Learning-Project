import re
import math
from nltk.stem import PorterStemmer

class TextCleaner:
    def __init__(self):
        self.documents = {}

    def RemoveSpecialCharacter(self):
        for key, value in self.documents.items():
            cleaned_value = re.sub("[!@#$%^&*()'`,.’1234567890]", '', value)
            self.documents[key] = cleaned_value

    def LowerCase(self):
        for key, value in self.documents.items():
            lowercased_value = value.lower()
            self.documents[key] = lowercased_value
    
    def Tokenizer(self):
        for key, value in self.documents.items():
            self.documents[key] = value.split()

    def Stem(self):
        stemmer = PorterStemmer()
        for key, value in self.documents.items():
            tokens = []
            for word in value:
                tokens.append(stemmer.stem(word))
            self.documents[key] = tokens
                

    def Clean(self, documents):
        self.documents = documents
        self.RemoveSpecialCharacter()
        self.LowerCase()
        self.Tokenizer()
        self.Stem()

        return self.documents
        
class TFIDFVectorizer:
    def __init__(self):
        self.documents = {}
        self.documents_length = 0

        self.tf = {}
        self.idf = {}
        self.tfidf = {}
        self.max_tfidf = 0
        self.min_tfidf = 0
        

    def fit(self, documents):
        self.documents = documents
        self.documents_length = len(documents)

    def TermFrequency(self):
        for key, value in self.documents.items():
            document_tf = {}
            document_length = len(value)
            for token in value:
                if token not in document_tf:
                    document_tf[token] = 1
                else:
                    document_tf[token] +=1

            for doctf_key, doctf_val in document_tf.items():
                document_tf[doctf_key] = doctf_val/document_length

            self.tf[key] = document_tf

    def InverseDocumentFrequency(self):
        for value in self.documents:
            for token in value:
                if token not in self.idf:
                    self.idf[token] = 0

        for key, value in self.idf.items():
            for document in self.documents:
                if key in document:
                    self.idf[key] +=1

        for key, value in self.idf.items():
            self.idf[key] = math.log(self.documents_length/value)

    def calculate_tfidf(self):
        for index, document in self.tf.items():
            self.tfidf[index] = document
            for key, value in document.items():
                self.tfidf[index][key] = value*self.idf[key]

    def normalize(self):
        numeric_tfidf = []

        for index, document in self.tfidf.items():
            for value in document.values():
                    numeric_tfidf.append(value)

        self.max_tfidf = max(numeric_tfidf)
        self.min_tfidf = min(numeric_tfidf)

        for index, document in self.tfidf.items():
            for key, value in document.items():
                self.tfidf[index][key] = (value-self.min_tfidf)/(self.max_tfidf-self.min_tfidf)
                

    def Vectorize(self):
        self.TermFrequency()
        self.InverseDocumentFrequency()
        self.calculate_tfidf()
        self.normalize()
        
        return self.tfidf