import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"

# ================== LOAD MODEL ==================
model = load_model("flower_model.keras")
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# ================== HOME PAGE ==================
@app.route('/')
def index():
    return render_template("index.html")

# ================== PREDICTION ROUTE ==================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        
        input_image = tf.keras.utils.load_img(filepath, target_size=(180,180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array,0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        predicted_class = flower_names[np.argmax(result)]
        Accuracy = round(100 * np.max(result), 2)

        return render_template("result.html", 
                               filename=filename,
                               flower=predicted_class,
                               Accuracy=Accuracy)

# ================== SERVE UPLOADED FILES ==================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)





