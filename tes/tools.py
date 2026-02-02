import numpy as np

# class yang meyimpan algoritma knn + accuracy
class KNN:
  # constructor
    def __init__(self, n_neighbors, weight="uniform", distance_metric="euclidean", p=2):
        self.n_neighbors = n_neighbors # jumlah tetangga / neighbor
        self.weight = weight # bobbot
        self.distance_metric = distance_metric # metode jarak
        self.p = p
        self.x = None
        self.y = None
        self.cm = None
        self.precision = []
        self.recall = []
        self.f1score = []

    # fungsi training
    def fit(self, x, y):
      # jika panjang x dan y itu tidak sama (sifatnya itu optional untuk menambahkan error handling)
        if len(x) != len(y):
            raise f"length of x is different with length of y, x = ({len(x)}) and y = ({len(y)})"

        # menyimpan variable dalam attribut kelas
        self.x = np.array(x)
        self.y = np.array(y)

    # fungsi untuk melakukan prediksi (untuk multiple data)
    def predict(self, x_predict):
        x_predict = np.array(x_predict) # melakukan konversi ke numpy array agar bisa melakukan komputasi
        predicted = [self._predict(x) for x in x_predict] # melakukan prediksi untuk setiap data
        return np.array(predicted)

    # untuk single data
    def _predict(self, x):
      # menghitung jarak sesuai metode yang dipilih sebelumnya
        if self.distance_metric == "euclidean":
            distance = np.sqrt(np.sum((self.x - x) ** 2, axis=1))
        elif self.distance_metric == "manhattan":
            distance = np.sum(np.abs(self.x - x), axis=1)
        elif self.distance_metric == "minkowski":
            distance = np.sum(np.abs(self.x - x) ** self.p, axis=1) ** (1 / self.p)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # mencari tetangga terdekat sebanyak jumlah neighbor
        nearest = np.argpartition(distance, self.n_neighbors)[:self.n_neighbors]
        labels = [self.y[i] for i in nearest] # melabeli semua hasil

        # jika tanpa menggunakan perhitungan bobot
        if self.weight == "uniform":
            return np.bincount(labels).argmax() # mengembalikan label dengan hasil vote terbanyak

        # jika menggunakan bobot
        if self.weight == "distance":
            weight = [(1/(distance[i]+1e-10)) for i in nearest] # menghitung bobot
            return np.bincount(labels, weights=weight).argmax() # menghasilkan label dg vote dan bobot terbanyak

        # error handling (optional)
        raise ValueError("Can only use 'uniform' or 'distance' for weight")

    # perhitungan akurasi
    def accuracy(self, y_pred, y_true):
        return np.sum(y_pred == y_true) / len(y_true)

    # menampilkan akurasi, prrecision, recall, dan f1 score
    def metric(self, y_pred, y_true, label):
        y_true = y_true.tolist()

        n_label = len(label) # banyak label

        confusion_matrix = np.zeros((n_label, n_label)) # confusion matriks

        # menghitung masing masing cell
        for i in range(len(y_pred)):
            confusion_matrix[label.index(y_pred[i])][label.index(y_true[i])] += 1
        self.cm = confusion_matrix # menyimpan di object
        print("Confusion Matrix:\n", confusion_matrix)

        # menghitung akurasi model
        accuracy = np.sum([confusion_matrix[i][i] for i in range(n_label)]) / len(y_pred)

        # perhitungan precision
        for i in range(n_label):
            print("\nclass:", label[i])

            # menghitung precision
            precision = confusion_matrix[i][i] / np.sum([confusion_matrix[i][k] for k in range(n_label)])
            self.precision.append(precision)
            print("precision:", precision)

            # menghitung recall
            recall = confusion_matrix[i][i] / np.sum([confusion_matrix[k][i] for k in range(n_label)])
            self.recall.append(recall)
            print("recall:", recall)

            # menghitung f1 score
            f1score = (2 * precision * recall) / (precision + recall)
            self.f1score.append(f1score)
            print("f1Score:", f1score)

        print("\naccuracy:", accuracy)
        return confusion_matrix
