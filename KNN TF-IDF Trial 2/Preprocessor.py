import re
import math
from nltk.stem import PorterStemmer
import pandas as pd

class TextCleaner:
    def __init__(self):
        self.documents = {}

    def RemoveSpecialCharacter(self):
        try:
            for key, value in self.documents.items():
                self.documents[key] = str(value)
                cleaned_value = re.sub("[!@#$%^&*()'`,.’1234567890]", '', value)
                self.documents[key] = cleaned_value
        except:
            if not isinstance(self.documents, pd.Series):
                cleaned_value = re.sub("[!@#$%^&*()'`,.’1234567890]", '', self.documents)
                self.documents = cleaned_value
            else:
                print('"its an instance of pd.series')
    def LowerCase(self):
        try:
            for key, value in self.documents.items():
                lowercased_value = value.lower()
                self.documents[key] = lowercased_value
        except:
            lowercased_value = self.documents.lower()
            self.documents = lowercased_value
    
    def Tokenizer(self):
        try:
            for key, value in self.documents.items():
                self.documents[key] = value.split()
        except:
            self.documents = self.documents.split()

    def Stem(self):
        stemmer = PorterStemmer()
        try:
            for key, value in self.documents.items():
                tokens = []
                for word in value:
                    tokens.append(stemmer.stem(word))
                self.documents.loc[key] = tokens
        except:
            tokens = []
            for word in self.documents:
                tokens.append(stemmer.stem(word))
                

    def Clean(self, documents):
        self.documents = documents
        print("Removing Special Character (1/4)")
        self.RemoveSpecialCharacter()
        print("Lower Case (2/4)")
        self.LowerCase()
        print("Tokenizing (3/4)")
        self.Tokenizer()
        print("Stemming All Term (4/4)")
        self.Stem()
        print("Cleaning Completed")

        return self.documents
        
class TFIDFVectorizer:
    def __init__(self):
        self.documents = {}
        self.documents_length = 0
        self.tf = {}
        self.idf = {}
        self.idf_token = {}
        self.tfidf = {}
        self.max_tfidf = 0
        self.min_tfidf = 0
        self.trained = False

    def fit(self, documents):
        self.documents = documents
        self.documents_length = len(documents)

    def TermFrequency(self):
        if self.trained == False:
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
        else:
            document_tf = {}
            document_length = len(self.documents)
            for token in self.documents:
                if token not in document_tf:
                    document_tf[token] = 1
                else:
                    document_tf[token] +=1
            for doctf_key, doctf_val in document_tf.items():
                document_tf[doctf_key] = doctf_val/document_length

            self.tf = document_tf
                    

    def InverseDocumentFrequency(self):
        if self.trained == False:
            for value in self.documents:
                for token in value:
                    if token not in self.idf:
                        self.idf_token[token] = 0

            for key, value in self.idf_token.items():
                for document in self.documents:
                    if key in document:
                        self.idf_token[key] +=1

            for key, value in self.idf_token.items():
                self.idf[key] = math.log(self.documents_length/value)
        else:
            for key in self.documents:
                if key not in self.idf_token:
                    self.idf_token[key] = 1
                else:
                    self.idf_token[key] += 1

            for key, value in self.idf_token.items():
                self.idf[key] = math.log(self.documents_length/value)

    def calculate_tfidf(self):
        if self.trained == False:
            for index, document in self.tf.items():
                self.tfidf[index] = document
                for key, value in document.items():
                    self.tfidf[index][key] = value*self.idf[key]
        else:
            for key, value in self.tf.items():
                self.tfidf[key] = value*self.idf[key]

    def calculate_trained_tfidf(self, xtest):
        self.trained = True
        self.documents = xtest
        self.tf = {}
        self.tfidf = {}
        self.TermFrequency()
        self.InverseDocumentFrequency()
        self.calculate_tfidf()
        self.Fill()
        #self.normalize()
    
        return self.tfidf

    def normalize(self):
        numeric_tfidf = []
        if self.trained == False:
            for index, document in self.tfidf.items():
                for value in document.values():
                        numeric_tfidf.append(value)

            self.max_tfidf = max(numeric_tfidf)
            self.min_tfidf = min(numeric_tfidf)

            for index, document in self.tfidf.items():
                for key, value in document.items():
                    self.tfidf[index][key] = (value-self.min_tfidf)/(self.max_tfidf-self.min_tfidf)
        else:
            for value in self.tfidf.values():
                numeric_tfidf.append(value)
            
            current_maxtfidf = max(numeric_tfidf)
            current_mintfidf = min(numeric_tfidf)

            if self.max_tfidf < current_maxtfidf:
                self.max_tfidf = current_maxtfidf
            if self.min_tfidf > current_mintfidf:
                self.min_tfidf = current_mintfidf

            for key, value in self.tfidf.items():
                self.tfidf[key] = (value-self.min_tfidf)/(self.max_tfidf-self.min_tfidf)

    def Fill(self):
        if self.trained == False:
            for index, documents in self.tfidf.items():
                for key in self.idf_token:
                    if key not in documents:
                        self.tfidf[index][key] = 0
        else:
            for key in self.idf_token:
                if key not in self.tfidf:
                    self.tfidf[key] = 0

    def Vectorize(self):
        self.trained = False
        self.TermFrequency()
        self.InverseDocumentFrequency()
        self.calculate_tfidf()
        self.Fill()
        #self.normalize()
        self.trained = True

        return self.tfidf