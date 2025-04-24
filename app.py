from flask import Flask, request, jsonify, render_template
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import logging
import os

app = Flask(__name__)

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/prediction_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

model_dir = r"C:\Users\priya\Projects\Deepfake Image Detection\model"

model = ViTForImageClassification.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True
)

processor = ViTImageProcessor.from_pretrained(model_dir)
print("Model and Processor loaded successfully!")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logging.warning("No file part in the request")
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()

        predicted_label = model.config.id2label[predicted_class_id]
        confidence = torch.softmax(logits, dim=1)[0][predicted_class_id].item()

        # Log the prediction
        logging.info(f"Prediction: {predicted_label}, Confidence: {confidence:.4f}")

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
