import joblib

# Load the trained model and vectorizer
model = joblib.load("resume_screening_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# New resume text to predict
new_resume_text = "I have extensive experience in bitcoin development and Blockchain technology."

# Transform the new resume text
new_resume_vectorized = vectorizer.transform([new_resume_text])  # Creates a 2D array

# Predict the category
predicted_category = model.predict(new_resume_vectorized)[0]

print(f"The predicted category for the resume is: {predicted_category}")
