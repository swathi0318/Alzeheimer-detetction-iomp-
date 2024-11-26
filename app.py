from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Class labels
class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_path = os.path.join("static", image_file.filename)
            image_file.save(image_path)
            
            # Preprocess the image
            img = load_img(image_path, target_size=(224, 224), color_mode="grayscale")
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(img)
            result = np.argmax(prediction)
            predicted_class = class_names[result]

            return render_template("index.html", prediction=predicted_class, image_loc=image_path)

    return render_template("index.html", prediction=None, image_loc=None)

if __name__ == "__main__":
    app.run(debug=True)
