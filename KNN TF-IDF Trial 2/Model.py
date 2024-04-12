import Preprocessor
import math
import numpy

class KNN:
    def __init__(self, k=3):
        self.x_test = {}
        self.y_test = {}
        self.x_train = {}
        self.y_train = {}
        self.range = {}
        self.voted = {}
        self.voting_result = {}
        self.class_result = 0
        self.k = k

    def Fit(self, xtrain, ytrain):
        if len(xtrain) == len(ytrain):
            self.x_train = xtrain
            self.y_train = ytrain
        else:
            print("X Train length doesn't match Y Train Length")

    def Vectorize(self):
        Cleaner = Preprocessor.TextCleaner()
        Vectorizer = Preprocessor.TFIDFVectorizer()
        
        Vectorizer.fit(self.x_train)
        self.x_train = Cleaner.Clean(self.x_train)
        self.x_train = Vectorizer.Vectorize()

        self.x_test = Cleaner.Clean(self.x_test)
        self.x_test = Vectorizer.calculate_trained_tfidf(xtest=self.x_test)

    def Vote(self):
        sorted = numpy.argsort(list(self.range.values()))
        for total in range(self.k):
            self.voted[sorted[total]] = self.range[sorted[total]]
        
        for index in self.voted.keys():
            self.voted[index] = self.y_train[index]

        for index, values in self.voted.items():
            if values not in self.voting_result:
                self.voting_result[values] =1
            else:
                self.voting_result[values] +=1

        self.class_result = numpy.argmax(list(self.voting_result.values()))

    def Predict(self, xtest):
        self.x_test = xtest
        self.Vectorize()
        
        for index, document in self.x_train.items():
            range = []
            for key, value in document.items():
                range.append((self.x_test[key]-value)**2)
            self.range[index] = math.sqrt(sum(range))

        self.Vote()
