import pandas as pd
from nltk.stem import PorterStemmer
import numpy as np

class TFIDF_Vectorizer:
    def __init__(self):
        self.dataframe = {}
        self.tf = {}
        self.idf = {}
        self.tfidf = {}
        self.terms_frequencies = {}
        self.cleaned_dataframe = {}

    def fit(self, dataframe):
        self.dataframe = dataframe

    def Preprocessing(self, predict = None):
        selected_dataframe = self.dataframe[["Review", "Sentiment"]]
        lower_cased_dataframe = selected_dataframe.apply(lambda x: x.str.lower())
        no_punctuation_dataframe = lower_cased_dataframe.apply(lambda x: x.str.replace('[^\w\s]','', regex=True))
        tokenized_dataframe = no_punctuation_dataframe["Review"].str.split()

        ps = PorterStemmer()

        stemmed_dataframe = []

        for document in tokenized_dataframe:
            stemmed_token = []
            for token in document:
                stemmed_token.append(ps.stem(token))
            stemmed_dataframe.append(stemmed_token)

        self.cleaned_dataframe = stemmed_dataframe


    def TermFrequencies(self):
        for index, document in enumerate(self.cleaned_dataframe):
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

            self.tf[index] = matched_token

    def InverseDocumentFrequencies(self):
        possible_word = []
        text_length = len(self.cleaned_dataframe)
        for document in self.cleaned_dataframe:
            for word in document:
                if word not in possible_word:
                    possible_word.append(word)
            
        for word in possible_word:
            self.terms_frequencies[word] = sum(1 for document in self.cleaned_dataframe if word in document)

        for term in self.terms_frequencies:
            self.terms_frequencies[term] = np.log(text_length/(self.terms_frequencies[term]))

    def tfidf_vectorizer(self):
        for index, document in enumerate(self.tf.values()):
            for term in document:
                if self.idf[term] != 0:
                    self.tfidf[index][term] = document[term] / idf[term] 
                else:
                    self.tfidf[index][term] = 0
        self.tfidf = pd.DataFrame(tfidf).transpose().fillna(0)

    def Calculate_TFIDF(self, text_array = {}):
        self.Preprocessing()
        self.TermFrequencies()
        self.InverseDocumentFrequencies()
        self.tfidf_vectorizer()

        return self.tfidf