import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

data = pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN - Game Twitter Sentiment\twitter_training.csv")

# Seleksi data yang akan digunakan [ Cleaning ]
selected_data = data[["review", "sentiment"]]
selected_data = selected_data.iloc[1:7]

# Mengubah semua alfabet menjadi alfabet kecil [ Case Folding ]
selected_data['review'] = selected_data['review'].str.lower()
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

#Calculate Term Frequencies
tf = []
for tokens in selected_data['filtered_singlechar']:
    token_counts = {}
    for token in tokens:
        if token in token_counts:
            token_counts[token] += 1
        else:
            token_counts[token] = 1
    
df = pd.da