import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Charger le modèle InceptionV3 pré entraîné et ajusté finement
model = load_model("model_120.h5")

# Définir le fichier contenant les classes des résultats
file_name = "dog_classes_name.json"

# Ouvrir le fichier json avec les classes des résultats
with open(file_name, 'r') as file:
    # Decode the JSON data in the file and load it into a dictionary
    dog_classes = json.load(file)

# Définir les dimensions de l'image
img_size = (224, 224)

# Définir une fonction pour prétraiter l'image d'entrée
def preprocess_image(image):
    # Convertir l'image PIL en tableau numpy
    image = np.array(image)
    # Convertir le tableau numpy en tenseur TensorFlow
    image = tf.convert_to_tensor(image)
    # Redimensionner l'image à la taille requise par le modèle
    image = tf.image.resize(image, img_size)
    # Normaliser les valeurs de pixels pour qu'elles soient dans la plage [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    # Ajouter une dimension de lot (batch) au tableau
    image = np.expand_dims(image, axis=0)
    return image

# Définir la fonction de prédiction
def predict_dog_breed(image):
    # Prétraiter l'image d'entrée
    img_array = preprocess_image(image)
    # Utiliser le modèle personnalisé pour faire une prédiction
    predictions = model.predict(img_array)
    # Décoder la prédiction et obtenir la classe prédite et la probabilité
    class_index = np.argmax(predictions[0])
    predicted_class = dog_classes[str(class_index)]
    prediction_accuracy = round(predictions[0][class_index]*100, 2)
    return predicted_class, prediction_accuracy

# Configurer l'application Streamlit
st.set_page_config(page_title="Prédicteur de race de chien", page_icon=":dog:")
st.title("Prédicteur de race de chien")


# Permettre à l'utilisateur de téléverser une image
uploaded_file = st.file_uploader("Importez votre image", type="jpg")

if uploaded_file is not None:
    # Afficher l'image téléversée
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    # Prédire la race de chien et afficher le résultat
    predicted_class, prediction_accuracy = predict_dog_breed(image)
    st.write("Race prédite :", predicted_class)
    st.write("Précision de la prédiction :", prediction_accuracy, "%")
