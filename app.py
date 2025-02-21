from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer saved from the training script
model = joblib.load("resume_screening_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/screen_resume', methods=['POST'])
def screen_resume():
    # Parse the JSON data from the request
    data = request.get_json()
    if not data or 'category_label' not in data or 'resume_text' not in data:
        return jsonify({'error': 'Missing required fields'}), 400

    category_label = data['category_label']
    resume_text = data['resume_text']
    
    # Use the loaded vectorizer to transform the input resume text
    resume_vectorized = vectorizer.transform([resume_text])
    
    # Predict the category using the trained model
    predicted_category = model.predict(resume_vectorized)[0]
    
    # Build the response message based on the prediction
    if predicted_category == category_label:
        result = f"Resume is selected for the {category_label} role!"
    else:
        result = f"Resume is NOT selected for the {category_label} role. Predicted category: {predicted_category}"
    
    return jsonify({'message': result})

if __name__ == '__main__':
    app.run(debug=True)
