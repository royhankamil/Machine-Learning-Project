import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import math
import numpy as np
from wordcloud import WordCloud
from PIL import Image

data = pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN - Game Twitter Sentiment\twitter_training.csv")

# Seleksi data yang akan digunakan [ Cleaning ]
selected_data = data[["review", "sentiment"]]
selected_data.dropna()
selected_data = selected_data.iloc[1:2000]
original_data = selected_data

# Mengubah semua alfabet menjadi alfabet kecil [ Case Folding ]
selected_data['review'] = selected_data['review'].str.lower()
selected_data['review'] = selected_data['review'].astype(str)
selected_data['review'] = selected_data['review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Tokenize
selected_data['tokenize'] = selected_data['review'].apply(word_tokenize)

# Stemming
porter = PorterStemmer()
selected_data['stemmed'] = selected_data['tokenize'].apply(lambda y: [porter.stem(token) for token in y])

# Remove Stopword
stopwords = set(stopwords.words('english'))
selected_data['filtered'] = selected_data['stemmed'].apply(lambda tokens: [word for word in tokens if word not in stopwords])

# Remove single character
selected_data['filtered_singlechar'] = selected_data['filtered'].apply(lambda tokens: [word for word in tokens if len(word) != 1])

# Calculate how many times token is present in a document
present_documents = []
for tokens in selected_data['filtered_singlechar']:
    token_counts = {}
    for token in tokens:
        if token in token_counts:
            token_counts[token] += 1
        else:
            token_counts[token] = 1
    present_documents.append(token_counts)

# Calculate the term frequencies
tf_documents = []
for document in present_documents:
    total_words = sum(document.values())
    tf_document = {word: freq/total_words for word, freq in document.items()}
    tf_documents.append(tf_document)


# Calculate idf
idf_documents = {}
n = len(selected_data['filtered_singlechar'])
for document in selected_data['filtered_singlechar']:
    for term in document:
        if term not in idf_documents:
            df = sum(1 for doc in selected_data['filtered_singlechar'] if term in doc)
            idf_documents[term] = math.log(n/df) if df!= 0 else 0

# Calculate TF-IDF
tf_idf = []
for document in tf_documents:
    tf_idf_document = {}
    for term, tf_value in document.items():
        idf_value = idf_documents.get(term, 0)
        tf_idf_document[term] = tf_value * idf_value
    tf_idf.append(tf_idf_document)

tf_idf_dataframe = pd.DataFrame(tf_idf)
tf_idf_dataframe.fillna(0, inplace=True)

def input(input_text):
    # Preprocess the input text
    input_text = input_text.lower()
    input_text = re.sub(r'[^\w\s]', '', input_text)
    input_tokens = word_tokenize(input_text)
    input_stemmed = [porter.stem(token) for token in input_tokens]
    input_filtered = [word for word in input_stemmed if word not in stopwords]
    input_filtered_singlechar = [word for word in input_filtered if len(word) != 1]

    # Calculate TF-IDF for the input text
    input_token_counts = {}
    for token in input_filtered_singlechar:
        if token in input_token_counts:
            input_token_counts[token] += 1
        else:
            input_token_counts[token] = 1

    total_words_input = sum(input_token_counts.values())
    tf_input = {word: freq/total_words_input for word, freq in input_token_counts.items()}
    tf_idf_input = {term: tf_value * idf_documents.get(term, 0) for term, tf_value in tf_input.items()}

    # Convert TF-IDF vector into array-like object
    input_vector = np.zeros((1, len(tf_idf_dataframe.columns)))
    for idx, term in enumerate(tf_idf_dataframe.columns):
        input_vector[0, idx] = tf_idf_input.get(term, 0)

    return(input_vector)

# KNN Models
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_Train = y_train

    def predict(self, X_test):
        y_pred = []
        for i in range(X_test.shape[0]):
            distances = np.linalg.norm(self.X_train - X_test[i], axis=1)
            nearest_neighbors = distances.argsort()[:self.k]
            nearest_labels = self.Y_Train[nearest_neighbors]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])
        return(y_pred)


knn = KNN(k=3)

x = tf_idf_dataframe.values
y = selected_data['sentiment'].values

knn.fit(x, y)

def acc():
    result = []
    for index, document in original_data['review'].items():
        result.append(knn.predict(input(document)))

    result_dataframe = pd.DataFrame({'Review' : original_data['review'], 'Prediction' : result})

    result_dataframe['Actual'] = original_data['sentiment']
    result_dataframe['Prediction'] = result_dataframe['Prediction'].apply(lambda x: x[0])

    result_dataframe['Correct'] = (result_dataframe['Prediction'] == result_dataframe['Actual'])

    accuracy = result_dataframe['Correct'].sum() / len(result_dataframe)

    print(accuracy)

print(knn.predict(input("for a game, this is too good")))