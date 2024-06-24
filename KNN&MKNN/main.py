# import model
from MKNN import MKNN
import pandas as pd
import os

# fungsi untuk meminta input gejala dari pengguna
def get_symptom_input(prompt):
    return 2 if input(prompt + " (y/t): ").strip().lower() == 'y' else 1

# mengambil data latih yang telah menjadi model
X_train = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/x-train.csv'))
y_train = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/y-train.csv')).iloc[:, 0]

# Inisialisasi dan latih model
model = MKNN(n_neighbors=12)
model.fit(X_train, y_train)

# melabeli hasil numerik
label_class = {1: 'Abdominal Pain', 2: 'GI Haemorrhage', 3: 'Gastritis', 4: 'Gastroenteritis', 5: 'Dispepsia', 
               6: 'Apenditis Akut', 7: 'Cholelitiasis', 8: 'GERD'}

# Daftar gejala
symptoms = [
    "Apakah muncul terasa mual", "Apakah sering muntah", "Apakah ketika muntah muncul darah",
    "Apakah mulut menjadi asam", "Apakah muncul benjolan dalam perut", "Apakah dada menjadi panas",
    "Apakah jantung berdebar dengan kencang", "Apakah terasa nyeri di dada", "Apakah seringkali bersendawa secara berlebihan",
    "Apakah pernafasan menjadi sesak", "Apakah sering merasa nyeri di bagian ulu hati", "Apakah sering merasa nyeri di bagian perut",
    "Apakah perut terasa kembung", "Apakah sering merasa nyeri di perut bagian kiri", "Apakah sering merasa nyeri di perut bagian kanan bawah",
    "Apakah sering merasa nyeri di perut bagian kanan atas", "Apakah terasa sakit kepala", "Apakah terasa pusing",
    "Apakah badan menjadi demam", "Apakah BAB masih normal", "Apakah BAB tidak lancar", "Apakah BAB menjadi cair",
    "Apakah BAB menjadi hitam", "Apakah ketik BAB muncul darah", "Apakah muncul gejala flatus", "Apakah badan menjadi lemas",
    "Apakah nafsu makan menjadi menurun"
]

# kmpulkan input dari pengguna
input_data = pd.Series([get_symptom_input(symptom) for symptom in symptoms])

# prediksi dan cetak hasil
prediction = model.predict(input_data)
print("Gejala yang mungkin anda alami adalah: ", label_class[prediction])
