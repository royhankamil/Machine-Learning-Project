import pandas as pd
from preprocessor import TFIDFVectorizer
from model import KNN

data = pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN TF-IDF - TRIAL 3\twitter_training.csv")
review = data[["review", "sentiment"]].iloc[:100]

vectorizer = TFIDFVectorizer()
knn = KNN()

train_data = vectorizer.train(review["review"])
test_data = vectorizer.calculate_predict_tfidf("fuck you")

knn.fit(train_data, review["sentiment"])
print(knn.predict(test_data))


