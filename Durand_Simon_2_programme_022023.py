import json

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

# Charger le modèle ResNet50 pré-entraîné
model = tf.keras.applications.ResNet50(weights='imagenet')

# Define the file name to load from
file_name = "dog_classes.json"

# Open the file in read mode
with open(file_name, 'r') as file:
    # Decode the JSON data in the file and load it into a dictionary
    dog_classes = json.load(file)

# Définir une fonction pour prétraiter l'image d'entrée
def preprocess_image(image):
    # Redimensionner l'image à la taille d'entrée requise par le modèle
    image = image.resize((224, 224))
    # Convertir l'image PIL en tableau numpy
    img_array = np.array(image)
    # Prétraiter l'image pour le modèle ResNet50
    img_array = preprocess_input(img_array)
    # Ajouter une dimension de lot (batch) au tableau
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Définir une fonction pour prédire la race de chien
def predict_dog_breed(image):
    # Prétraiter l'image d'entrée
    img_array = preprocess_image(image)
    # Utiliser le modèle ResNet50 pré-entraîné pour faire une prédiction
    predictions = model.predict(img_array)
    # Décoder les 5 meilleures classes et leurs probabilités
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    # Obtenir la classe prédite et la probabilité
    predicted_class = decoded_predictions[0][1].split('-')[-1].title()
    prediction_accuracy = round(decoded_predictions[0][2]*100, 2)
    return predicted_class, prediction_accuracy

# Configurer l'application Streamlit
st.title("Prédicteur de race de chien")

# Permettre à l'utilisateur de télécharger une image
uploaded_file = st.file_uploader("Importez votre image", type="jpg")

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)

    # Prédire la race de chien et afficher le résultat
    predicted_class, prediction_accuracy = predict_dog_breed(image)
    st.write("Race prédite :", predicted_class)
    st.write("Précision de la prédiction :", prediction_accuracy, "%")
