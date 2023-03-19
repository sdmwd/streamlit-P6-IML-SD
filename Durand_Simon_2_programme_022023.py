import json

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

# Charger le modèle ResNet50 pré-entraîné
model = tf.keras.applications.ResNet50(weights='imagenet')

# Load the fine-tuned InceptionV3 model
model = load_model("model_120.h")

# Define the file name to load from
file_name = "dog_classes.json"

# Open the file in read mode
with open(file_name, 'r') as file:
    # Decode the JSON data in the file and load it into a dictionary
    dog_classes = json.load(file)

# Définir une fonction pour prétraiter l'image d'entrée
def preprocess_image(image):
    # Resize the image to the size required by the model
    image = image.resize((299, 299))
    # Convert the PIL image to a numpy array
    img_array = np.array(image)
    # Preprocess the image for InceptionV3
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    # Add a batch dimension to the array
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_dog_breed(image):
    # Prétraiter l'image d'entrée
    img_array = preprocess_image(image)
    # Utiliser le modèle personnalisé pour faire une prédiction
    predictions = model.predict(img_array)
    # Décoder la prédiction et obtenir la classe prédite et la probabilité
    class_index = np.argmax(predictions[0])
    predicted_class = dog_classes[class_index]
    prediction_accuracy = round(predictions[0][class_index]*100, 2)
    return predicted_class, prediction_accuracy

# Configurer l'application Streamlit
st.title("Prédicteur de race de chien")

# Permettre à l'utilisateur de télécharger une image
uploaded_file = st.file_uploader("Importez votre image", type="jpg")

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    # Prédire la race de chien et afficher le résultat
    predicted_class, prediction_accuracy = predict_dog_breed(image)
    st.write("Race prédite :", predicted_class)
    st.write("Précision de la prédiction :", prediction_accuracy, "%")
