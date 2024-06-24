import numpy as np

class MKNN:
    def __init__(self, n_neighbors=3, metric='jaccard', alpha=1.0):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.alpha = alpha
        self.x = None
        self.y = None
    
    def fit(self, x, y):
        if len(x) != len(y):
            raise ValueError(f"Length of x is different from length of y, x = ({len(x)}) and y = ({len(y)})")
        
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
    
    def _validity(self, x_idx):
        x_i = self.x[x_idx]
        if self.metric == 'jaccard':
            distances = [self._jaccard_distance(x_i, x_j) for x_j in self.x]  
        elif self.metric == 'cosine':
            distances = [self._cosine_similarity(x_i, x_j) for x_j in self.x]  
        elif self.metric == 'hamming':
            distances = [self._hamming_distance(x_i, x_j) for x_j in self.x]  
        nearest_neighbors = np.argsort(distances)[:self.n_neighbors]
        same_class_neighbors = sum(1 for idx in nearest_neighbors if self.y[idx] == self.y[x_idx])
        return same_class_neighbors / self.n_neighbors
    
    def _weighted_vote(self, distances, indices):
        weights = []
        for dist, idx in zip(distances, indices):
            validity = self._validity(idx)
            weight = validity * (1 / (dist + self.alpha))
            weights.append(weight)
        return weights
    
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
        
        nearest_distances = [distances[i] for i in nearest]
        labels = [self.y[i] for i in nearest]
        weights = self._weighted_vote(nearest_distances, nearest)
        
        weighted_votes = {}
        for label, weight in zip(labels, weights):
            if label not in weighted_votes:
                weighted_votes[label] = 0
            weighted_votes[label] += weight
        
        return max(weighted_votes, key=weighted_votes.get)

    def predict(self, x):
        if len(x.shape) == 1:
            return self.__predict(x)
        
        x = np.array(x)
        predicts_value = []
        for val in x:
            predicts_value.append(self.__predict(val))
        return predicts_value