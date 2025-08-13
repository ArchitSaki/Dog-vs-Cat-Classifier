# Dog-vs-Cat-Classifier
🐶🐱 Dog vs Cat Classifier
A Convolutional Neural Network (CNN) based image classifier that predicts whether an image contains a dog or a cat.
This project demonstrates deep learning techniques for binary image classification using TensorFlow/Keras.

📌 Project Overview
This project aims to classify images of dogs and cats using a deep learning model trained on the Kaggle Dogs vs Cats Dataset.
The notebook includes:

Data preprocessing and augmentation

CNN model building and training

Model evaluation and accuracy visualization

Making predictions on new images

📂 Dataset
Source: Kaggle Dogs vs Cats Dataset

Classes: Dog and Cat

Format: JPG images

⚙️ Installation & Setup
1️⃣ Clone the repository
bash
Copy
Edit
git clone https://github.com/ArchitSaki/Dog-vs-Cat-Classifier.git
cd Dog-vs-Cat-Classifier
2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
🛠 Dependencies
Python 3.x

TensorFlow / Keras

NumPy

Matplotlib

Pandas

Pillow

🚀 Training the Model
Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook dig_vs_cat_classifier.ipynb
Steps included in the notebook:

Load and preprocess data

Data augmentation for better generalization

Build CNN model

Train and validate

Evaluate performance

📊 Results
Achieved ~XX% accuracy on the validation dataset

Training and validation accuracy graphs included in the notebook

📸 Predictions Example
python
Copy
Edit
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("test_image.jpg", target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
📌 Future Improvements
Use Transfer Learning (e.g., VGG16, ResNet50)

Deploy as a web app using Flask or Streamlit

Optimize model for mobile deployment with TensorFlow Lite
