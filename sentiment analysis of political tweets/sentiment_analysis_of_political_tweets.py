
!pip install pandas numpy matplotlib seaborn scikit-learn nltk

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('finalSentimentdata2.csv')

print(df.head())

def preprocess_text(text):
   text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df['lemmatized_text'] = df['cleaned_text'].apply(lemmatize_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['lemmatized_text'])

sentiment_mapping = {'fear': 0, 'sad': 1, 'anger': 2, 'joy': 3}
df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)

y = df['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['fear', 'sad', 'anger', 'joy']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['fear', 'sad', 'anger', 'joy'], yticklabels=['fear', 'sad', 'anger', 'joy'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

import joblib
joblib.dump(model, 'sentiment_analysis_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    lemmatized_text = lemmatize_text(cleaned_text)
    text_vector = vectorizer.transform([lemmatized_text])

    prediction = model.predict(text_vector)

    sentiment_labels = ['fear', 'sad', 'anger', 'joy']
    return sentiment_labels[prediction[0]]

test_text = "I am feeling very sad about the lockdown situation."
print(f"Predicted Sentiment: {predict_sentiment(test_text)}")

