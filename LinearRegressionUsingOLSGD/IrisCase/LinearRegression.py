import numpy as np

class OLSClassifier:
    def __init__(
        self,
        fit_intercept=True,     # apakah model menggunakan bias/intercept
        alpha=0.0,              # kekuatan regularisasi
        l1_ratio=0.0,           # 0 = Ridge (L2), 1 = Lasso (L1)
        max_iter=1000,          # jumlah iterasi maksimum (gradient descent)
        lr=0.01,                # learning rate
        solver="closed_form",   # metode optimasi: closed_form atau gradient_descent
        threshold=0.5,          # ambang batas klasifikasi
        clip_proba=True,         # membatasi output agar berada di rentang [0, 1]
        n_classes=2
    ):
        # Menyimpan parameter sebagai atribut objek
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.lr = lr
        self.solver = solver
        self.threshold = threshold
        self.clip_proba = clip_proba
        self.n_classes = n_classes

    def _add_intercept(self, X):
        # Menambahkan kolom 1 sebagai bias (intercept)
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        # Konversi input ke numpy array
        X = np.asarray(X)
        y = np.asarray(y).astype(float)

        # Tambahkan intercept jika diaktifkan
        if self.fit_intercept:
            X = self._add_intercept(X)

        # Inisialisasi koefisien dengan nol
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        # Memilih metode pelatihan
        if self.solver == "closed_form":
            self._fit_closed_form(X, y)
        elif self.solver == "gradient_descent":
            self._fit_gd(X, y)
        else:
            raise ValueError("solver must be 'closed_form' or 'gradient_descent'")

        return self

    def _fit_closed_form(self, X, y):
        # Closed-form hanya mendukung regularisasi L2 (Ridge)
        if self.l1_ratio != 0:
            raise ValueError("Closed-form solution does not support L1 regularization")

        # Matriks identitas untuk regularisasi
        I = np.eye(X.shape[1])

        # Intercept tidak ikut diregularisasi
        if self.fit_intercept:
            I[0, 0] = 0

        # Rumus OLS + Ridge:
        # w = (XᵀX + αI)⁻¹ Xᵀy
        self.coef_ = np.linalg.pinv(
            X.T @ X + self.alpha * I
        ) @ X.T @ y

    def _fit_gd(self, X, y):
        # Optimasi menggunakan gradient descent
        for _ in range(self.max_iter):
            # Prediksi nilai kontinu
            y_pred = X @ self.coef_

            # Error
            error = y_pred - y

            # Gradien loss OLS
            grad = (X.T @ error) / len(y)

            # Regularisasi L2 (Ridge)
            grad += self.alpha * (1 - self.l1_ratio) * self.coef_

            # Regularisasi L1 (Lasso) menggunakan subgradient
            grad += self.alpha * self.l1_ratio * np.sign(self.coef_)

            # Intercept tidak diregularisasi
            if self.fit_intercept:
                grad[0] = grad[0] - self.alpha * self.coef_[0]

            # Update bobot
            self.coef_ -= self.lr * grad

    def decision_function(self, X):
        # Menghasilkan skor linear (belum diklasifikasikan)
        X = np.asarray(X)
        if self.fit_intercept:
            X = self._add_intercept(X)
        return X @ self.coef_

    def predict_proba(self, X):
        # Mengubah skor linear menjadi pseudo-probabilitas
        scores = self.decision_function(X)

        # Membatasi nilai agar berada pada rentang [0, 1]
        if self.clip_proba:
            scores = np.clip(scores, 0.0, 1.0)

        # Format output seperti classifier sklearn
        return np.vstack([1 - scores, scores]).T

    def predict(self, X):
        # Klasifikasi berdasarkan threshold
        scores = self.decision_function(X)
        preds = np.round(scores)
        preds = np.clip(preds, 0, self.n_classes - 1)

        return preds.astype(int)
    
    def predict_percentage(self, X):
        # Mengembalikan probabilitas kelas terpilih dalam persen
        score = self.decision_function(X)[0]
        cls = int(np.clip(round(score), 0, self.n_classes - 1))
        score = self.decision_function(X)[0]
        confidence = 1 - abs(score - cls)

        return f"{confidence * 100:.2f}%"
    
    def score(self, X, y):
        # Menghitung akurasi
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
