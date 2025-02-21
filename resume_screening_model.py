from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pandas as pd

# Load and preprocess your data
data = pd.read_csv(r"C:\Users\shyam\Downloads\UpdatedResumeDataSet.csv.csv")
X = data['Resume']
y = data['Category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Save the model
joblib.dump(model, "resume_screening_model.pkl")
