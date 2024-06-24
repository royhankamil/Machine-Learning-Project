import numpy as np
import pandas

class KNN:
    def __init__(self, n_neighbors=3, metric='jaccard'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.x = None
        self.y = None
    
    def fit(self, x, y):
        if len(x) != len(y):
            raise ValueError(f"length of x is different with length of y, x = ({len(x)}) and y = ({len(y)})")
        
        self.x = np.array(x)
        self.y = np.array(y)
    
    def _jaccard_distance(self, a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        intersection = np.sum(np.minimum(a, b))
        union = np.sum(np.maximum(a, b))
        return 1 - intersection / union
    
    def _cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
    
    def _hamming_distance(self, a, b):
        return np.sum(a != b) / len(a)
    
    def __predict(self, x):
        x = np.array(x)
        if self.metric == 'jaccard':
            distance_func = self._jaccard_distance
        elif self.metric == 'cosine':
            distance_func = self._cosine_similarity
        elif self.metric == 'hamming':
            distance_func = self._hamming_distance
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        distances = [distance_func(x, x1) for x1 in self.x]
        
        if self.metric == 'cosine':
            nearest = np.argsort(distances)[::-1][:self.n_neighbors]  
        else:
            nearest = np.argsort(distances)[:self.n_neighbors]
        
        labels = [self.y[i] for i in nearest]
        return np.bincount(np.array(labels)).argmax()

    def predict(self, x):
        if len(x.shape) == 1:
            return self.__predict(x)
        
        x = np.array(x)
        predicts_value = []
        for val in x:
            predicts_value.append(self.__predict(val))
        return predicts_value

                    

    