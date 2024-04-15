import pandas as pd
import Model

# Comment Sentiment. A value of zero represents a negative sentiment, whereas values of one and two represent neutral
df = pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN TF-IDF Trial 2\twitter_training.csv")
sample_df = df.iloc[:400, :]
xtrain = sample_df["review"]
ytrain = sample_df["sentiment"]

KNN_Model = Model.KNN(k=11)

KNN_Model.Fit(xtrain, ytrain)
print(KNN_Model.Predict("that was the first borderlands session in a long time where i actually had a really satisfying combat experience. i got some really good kills"))