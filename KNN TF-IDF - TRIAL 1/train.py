import pandas as pd
from text_preprocessor import TFIDF_Vectorizer


dataframe=pd.read_csv(r"C:\Users\Folive\Documents\Python\AI\Basic-Machine-Learning\KNN TF-IDF - Simple\my-test-data.csv")


Text_Preprocessor = TFIDF_Vectorizer()

tfidf = TFIDF_Vectorizer()
tfidf.fit(dataframe)
tfidf = tfidf.Calculate_TFIDF()
print(tfidf)