# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1 ) 
# data akan di shuffle

"""
Dengan fungsi train_test_split dari library sklearn, kita membagi array X dan y ke dalam 20% data
testing (test_size=0.2 ). Misal total dataset A yang kita miliki adalah 1000 record, dengan 
test_size=0.2, maka data testing kita berjumlah 200 record dan jumlah data training sebesar 800 (80%).
"""

from sklearn.model_selection import train_test_split

x_data = range(10)
y_data = range(10)

print("random state yang ditentukan ")
for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 42)
    print(y_test)


print("random state yang tidak ditentukan ")
for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=None)
    print(y_test)

"""
Melalui parameter random_state, fungsi train_test_split menyediakan random seed yang tetap untuk 
internal pseudo-random generator yang digunakan pada proses shuffling. Umumnya, nilai yang digunakan 
adalah 0, atau 1, atau ada juga yang menggunakan 42. Menentukan parameter random_state bertujuan untuk 
dapat memastikan bahwa hasil pembagian dataset konsisten dan memberikan data yang sama setiap kali model 
dijalankan. Jika tidak ditentukan, maka tiap kali melakukan split, kita akan mendapatkan data train dan 
tes berbeda, yang juga akan membuat akurasi model ML menjadi berbeda tiap kali di-run.  
"""
