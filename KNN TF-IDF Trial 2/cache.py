import pandas as pd
import Model

# Comment Sentiment. A value of zero represents a negative sentiment, whereas values of one and two represent neutral
df = pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN TF-IDF Trial 2\comments.csv")
sample_df = df.iloc[0:4, :]
xtrain = sample_df["Comment"]
ytrain = sample_df["Sentiment"]

KNN_Model = Model.KNN()

KNN_Model.Fit(xtrain, ytrain)
KNN_Model.Predict("This is a good pocket")