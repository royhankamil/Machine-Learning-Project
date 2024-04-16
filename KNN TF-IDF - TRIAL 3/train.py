import pandas as pd
from preprocessor import TFIDFVectorizer
from model import KNN

data = pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN TF-IDF - TRIAL 3\twitter_training.csv")
review = data[["review", "sentiment"]].iloc[:5000]

vectorizer = TFIDFVectorizer()
knn = KNN()

train_data = vectorizer.train(review["review"])
knn.fit(train_data, review["sentiment"])

def predict(text):
    text = vectorizer.calculate_predict_tfidf(text)
    return knn.predict(text)

#predict = {}


#for index, doc in review["review"].items():
#    test_data = vectorizer.calculate_predict_tfidf(doc)
#    predict[index] = knn.predict(test_data)

#right_answer = 0
#wrong_answer = 0
#for index, value in predict.items():
#    if value == review["game"][index]:
#        right_answer+=1
#    else:
#        wrong_answer+=1

#print("accuracy = ", right_answer/(wrong_answer+right_answer)*100, "%")