import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask import Flask, render_template
import os

app = Flask(__name__, template_folder='templates')


# ✅ Define Flask App
#app = Flask(__name__)

# ✅ Create "uploads" folder if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Load the trained model
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        attention_weights = tf.keras.backend.softmax(tf.keras.backend.dot(score, self.u), axis=1)
        context_vector = attention_weights * inputs
        return tf.keras.backend.sum(context_vector, axis=1)

model_path = "deepfake_model.h5"
model = tf.keras.models.load_model(model_path, custom_objects={"Attention": Attention})

# ✅ Function to extract MFCC features
def extract_features(file_path, max_len=100):
    signal, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T

    if mfccs.shape[0] < max_len:
        mfccs = np.pad(mfccs, ((0, max_len - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:max_len, :]

    return np.array([mfccs])

# ✅ Home Route
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Prediction Route
@app.route("/predict", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(file_path)

    # Extract features & make prediction
    features = extract_features(file_path)
    prediction = model.predict(features)
    result = "FAKE" if prediction[0][0] > 0.5 else "REAL"
    confidence = prediction[0][0] if result == "FAKE" else 1 - prediction[0][0]

    return jsonify({"prediction": result, "confidence": round(float(confidence), 2)})

if __name__ == "__main__":
    app.run(debug=True)
