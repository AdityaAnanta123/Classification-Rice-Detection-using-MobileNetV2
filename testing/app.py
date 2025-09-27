# testing/app.py
"""
Flask app to upload a rice image and get predicted class.
Usage: python app.py
"""

import io
import json
from pathlib import Path
from flask import Flask, request, render_template_string, send_file
import tensorflow as tf
from PIL import Image
from training.utils import load_class_map

# Paths
BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "artifacts" / "models" / "final_model.keras"
CLASS_MAP_PATH = BASE / "artifacts" / "class_map.json"

app = Flask(__name__)

# Load model & class map once
print("Loading model from", MODEL_DIR)
model = tf.keras.models.load_model(str(MODEL_DIR))
class_map = {}
if Path(CLASS_MAP_PATH).exists():
    class_map = load_class_map(str(CLASS_MAP_PATH))
    # invert
    idx_to_class = {int(v):k for k,v in class_map.items()}
else:
    idx_to_class = None

HTML = """
<!doctype html>
<title>Rice Classifier</title>
<h2>Upload an image of Nasi (rice) to classify</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*">
  <input type=submit value="Predict">
</form>
{% if filename %}
  <h3>Prediction</h3>
  <img src="{{ url_for('uploaded_file') }}" style="max-width:400px;"><br>
  <p>Predicted: <b>{{ pred }}</b> (prob: {{ prob }})</p>
{% endif %}
"""

uploaded_img = None

@app.route('/', methods=['GET','POST'])
def index():
    global uploaded_img
    pred_text = None
    prob = None
    if request.method == 'POST':
        if 'file' not in request.files:
            pred_text = "No file part"
        else:
            file = request.files['file']
            img_bytes = file.read()
            uploaded_img = img_bytes
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            image = image.resize((224,224))
            import numpy as np
            x = np.array(image)/255.0
            x = x[None, ...]
            preds = model.predict(x)[0]
            top_idx = int(preds.argmax())
            prob = float(preds[top_idx])
            class_name = idx_to_class.get(top_idx, str(top_idx))
            pred_text = class_name
    return render_template_string(HTML, filename=uploaded_img is not None, pred=pred_text, prob=f"{prob:.3f}" if prob else None)

@app.route('/uploaded')
def uploaded_file():
    global uploaded_img
    if uploaded_img is None:
        return "No image", 404
    return send_file(io.BytesIO(uploaded_img), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
