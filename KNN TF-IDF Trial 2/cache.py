import pandas as pd
import Preprocessor

# Comment Sentiment. A value of zero represents a negative sentiment, whereas values of one and two represent neutral
df = pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN TF-IDF Trial 2\comments.csv")
sample_df = df.iloc[0:4, :]
sample_df = sample_df["Comment"]

cleaner = Preprocessor.TextCleaner()
vectorizer = Preprocessor.TFIDFVectorizer()

cleaned = cleaner.Clean(sample_df)

vectorizer.fit(cleaned)
vectorized = vectorizer.Vectorize()
vectorized = pd.DataFrame(vectorized)
print(vectorized.values)