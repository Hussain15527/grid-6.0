import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('package_classification_model_improved.h5')
model.summary()

# Function to preprocess an input image
def preprocess_image(img_path, img_width=224, img_height=224):  # Fixed dimensions to 224x224
    img = image.load_img(img_path, target_size=(img_width, img_height))  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Scale pixel values to [0, 1]
    return img_array

# Function to predict whether the image is damaged or intact
def classify_image(model, img_path):
    processed_img = preprocess_image(img_path)  # Preprocess the image to 224x224
    prediction = model.predict(processed_img)  # Get prediction from the model
    if prediction < 0.5:
        return "Damaged"
    else:
        return "Intact"

# Example usage
intact = './data/intact/0198143165548_top.png'  # Replace with the path to your image
damaged = './data/damaged/0509324410896_top.png'
result1 = classify_image(model, intact)
result2 = classify_image(model, damaged)

print(f'The package is: {result1}')

print(f'The package is: {result2}')
