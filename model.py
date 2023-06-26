import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import gensim
from gensim.parsing.preprocessing import STOPWORDS


df = pd.read_csv("Cleaned_Indian_Food_Dataset.csv")


def preprocess_text(text):
    
    tokens = [token for token in gensim.utils.simple_preprocess(text) if token not in STOPWORDS]
    
    return " ".join(tokens)


df["preprocessed_text"] = (df["Cuisine"] + " " + df["TotalTimeInMins"].astype(str)).apply(preprocess_text)

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["preprocessed_text"])



model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, activation='relu', solver='adam', random_state=42)
model.fit(X, df["TranslatedRecipeName"])

# Save the trained model as a pickle file
with open('indian_food_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the CountVectorizer as a pickle file
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

