from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
import io
import sys


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


model_path = r"C:\Users\Vatsa U\OneDrive\Desktop\skin disease detection\Dataset_3\model\skin_disease_model_2.keras"
model = tf.keras.models.load_model(model_path)


class_labels = [
    "Athlete Foot",
    "Cellulitis",
    "Chickenpox",
    "Herpes Zoster",
    "Keratosis",
    "Melanoma",
    "No disease detected",
    "Onychomycosis",
    "Ringworm"
]


app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'  


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def introduction():
    return render_template('introduction.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No selected file")

    if file:
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        
        top_predictions, top_probs = predict_disease(file_path)
        
        
        session['top_predictions'] = top_predictions
        session['top_probs'] = top_probs
        
        
        return render_template('result.html', 
                               top_predictions=top_predictions, 
                               top_probs=top_probs)

def predict_disease(img_path):
    try:
        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0  

        
        predictions = model.predict(img_array)[0]  
        top_indices = np.argsort(predictions)[-4:]  

        
        top_predictions = [class_labels[i] for i in reversed(top_indices)]
        top_probs = [predictions[i] * 100 for i in reversed(top_indices)]  
        
        return top_predictions, top_probs
    except Exception as e:
        print(f"An error occurred while predicting the disease: {e}")
        return ["Error"], [0]

@app.route('/disease/<disease_name>')
def disease_advice(disease_name):
    # Ensure the disease_name is sanitized or validated to prevent unauthorized access
    return render_template(f'diseases/{disease_name}.html')

@app.route('/back_to_results')
def back_to_results():
    
    top_predictions = session.get('top_predictions', [])
    top_probs = session.get('top_probs', [])
    
    
    return render_template('result.html', 
                           top_predictions=top_predictions, 
                           top_probs=top_probs)

if __name__ == "__main__":
    app.run(debug=True)
