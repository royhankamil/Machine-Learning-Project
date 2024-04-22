import pandas as pd
from preprocessor import TFIDFVectorizer
from model import KNN
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

data = pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN TF-IDF - TRIAL 3\twitter_training.csv")
review = data[["review", "sentiment"]].iloc[:100]

vectorizer = TFIDFVectorizer()
knn = KNN()

x_train, x_test, y_train, y_test = train_test_split(review["review"], review["sentiment"], test_size=0.4, random_state=142)

train_data = vectorizer.train(x_train)
knn.fit(train_data, y_train)

def predict(text):
    text = vectorizer.calculate_predict_tfidf(text)
    return knn.predict(text)

predicted = {}

for index, doc in x_test.items():
    test_data = vectorizer.calculate_predict_tfidf(doc)
    predicted[index] = knn.predict(test_data)

right_answer = 0
wrong_answer = 0

for index, value in predicted.items():
    if value == y_test[index]:
        right_answer+=1
    else:
        wrong_answer+=1

accuracy = right_answer/(wrong_answer+right_answer)*100

def wordcloud():
    return WordCloud(width=800, height=400, background_color="white").generate(x_train.values.split())

#wordcloud()