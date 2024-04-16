import math

class KNN:
    def __init__(self, n_neighbor = 3):
        self.xtrain = {}
        self.ytrain = {}
        self.n_neighbor = n_neighbor

    def fit(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def predict(self, ytest = {}):
        distances = sorted({index: math.sqrt(sum({key: (value-document[key])**2 for key, value in ytest.items() if value !=0 or document[key] !=0}.values())) for index, document in self.xtrain.items()}.items(), key=lambda x:x[1])
        shortest_distance = [distances[index] for index in range(self.n_neighbor)]
        shortest_distance_class = [self.ytrain[index[0]] for index in shortest_distance]
        voted_distance_class = {key: shortest_distance_class.count(key) for key in shortest_distance_class}

        return max(voted_distance_class.items(),key=lambda x:x[1])[0]