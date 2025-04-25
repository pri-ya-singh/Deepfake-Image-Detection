# 🧠 Deepfake Image Detection

## 🚀 Project Aim
This project aims to detect whether an uploaded image is a real or deepfake image using a Vision Transformer (ViT)-based image classification model.

![Deefake Image Detection](https://github.com/pri-ya-singh/Deepfake-Image-Detection/blob/main/images/Screenshot%202025-04-25%20051458.png)

---

## 📋 How to Use

To get started with this project, follow these steps:

1. **Clone the Repository:**  
   Download the project repository to your local machine:
   ```bash
   git clone https://github.com/pri-ya-singh/deepfake-image-detection.git
   cd deepfake-image-detection
   ```

2. **Install Required Packages:**  
   - `torch`
   - `transformers`
   - `flask`
   - `Pillow`

3. **Prepare Model Files:**  
   Place the pre-trained model files inside a folder named `model/` in the root directory. Required files include:
   - `config.json`
   - `model.safetensors`
   - `preprocessor_config.json`
   - `training_args.bin`

4. **Run the Application:**  
   Execute the `app.py` file to launch the web application:
   ```bash
   python app.py
   ```


## 📌 Project Description

### 🎯 What Does This Project Do?
This project allows users to upload an image and receive a classification result — either **Real** or **Fake** — along with the confidence score.  
If the image is predicted as `Fake`, the result is displayed in **red** color for better visibility.

![Deefake Image Detection](https://github.com/pri-ya-singh/Deepfake-Image-Detection/blob/main/images/Screenshot%202025-04-25%20051717.png)
![Deefake Image Detection](https://github.com/pri-ya-singh/Deepfake-Image-Detection/blob/main/images/Screenshot%202025-04-25%20051740.png)


---

## 🛠️ How Does This Project Work?

### 1. **Model Loading:**  
A Vision Transformer (ViT) model, fine-tuned for binary image classification, is loaded using Hugging Face’s `transformers` library.

### 2. **Image Upload:**  
Users can upload an image in `.jpg`, `.png`, or `.jpeg` format using the browser-based interface built with Flask.

### 3. **Preprocessing:**  
The uploaded image is converted to RGB and transformed into tensor format using `ViTImageProcessor`.

### 4. **Prediction:**  
The image tensor is passed to the model, which outputs the predicted label (`Real` or `Fake`) along with the softmax confidence score.

### 5. **Result Display:**  
The result is returned via the web page. If the image is detected as `Fake`, the prediction text appears in red.

### 6. **Logging:**  
All predictions, including filename, timestamp, label, and confidence, are saved in:
```
logs/predictions.log
```

---

## 🧪 Example Output

When an image is uploaded, the system returns:
```
Prediction: Fake
Confidence: 0.9745
```
