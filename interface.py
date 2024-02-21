import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image, ImageTk
import datetime
import json
import utils

dict = r'C:\Users\Moi\venv\tag\moi.json'
try:
    with open(dict, 'r') as file:
        dict_matrix = json.load(file)
except FileNotFoundError:
    dict_matrix = {}
print(dict_matrix)


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_height = 200
    img_width = 200
    resized_image = cv2.resize(image, (img_width, img_height))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.uint8)  # Convert to 8-bit format
    return resized_image

def reset_result_label():
    global result_label
    result_label.config(text="")


def predict_image(image_path):
    global result_label
    image_to_predict = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Redimensionner l'image
    img_height = 200
    img_width = 200
    resized_image = cv2.resize(image_to_predict, (img_width, img_height))

    # Afficher l'image redimensionnée
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()

    image9 = cv2.imread(image_path)
    image = cv2.cvtColor(image9, cv2.COLOR_BGR2RGB)

    img_1 = cv2.bilateralFilter(image, d=7, sigmaSpace=75, sigmaColor=75)
    image1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)

    R, G, B = cv2.split(image)

    a = image1.max()

    _, thresl = cv2.threshold(G, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresl1 = cv2.threshold(image1, a / 2 + 60, a, cv2.THRESH_BINARY_INV)
    contour = cv2.Canny(thresl, 100, 200)

    contour_with_contour = contour.copy()

    contours, _ = cv2.findContours(contour_with_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(contour_with_contour, contours, -1, (255, 255, 255), 2)

    colors_matrix = np.zeros((len(contours), 256, 3), dtype=np.uint8)

    for i, contour in enumerate(contours):
        for point in contour:
            x, y = point[0][1], point[0][0]
            colors_matrix[i, y, :] = image[x, y]

    matrice4 = {"tomi": colors_matrix.tolist()}

    matrice_trouvee = False

    for matrice_key, matrice_value in dict_matrix.items():
        if matrice_value == matrice4["tomi"]:
            matrice_trouvee = True
            break

    if matrice_trouvee:
        print("c'est une tomate")
    



        img_to_predict = np.expand_dims(resized_image, axis=0)

        # Charger votre modèle pré-entraîné
        model = tf.keras.models.load_model('C:/Users/Moi/venv/tag/model1.h5')
        

        predictions = model.predict(img_to_predict)
        res = np.argmax(predictions, axis=1)
    
        if res == 1:
                plt.imshow(cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2RGB))
                plt.show()
                print("La tomate est en santé.")
                result_label.config(text="cette tomate est  en santé")
        else:
                plt.imshow(cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2RGB))
                print("cette tomate n'est pas en santé.Veuillez revoir cette zone de culture")
                result_label.config(text="cette tomate n'est pas en santé.Veuillez revoir cette zone de culture")


    else:
        print("c'est pas une tomate. Revoyer votre selection")
        result_label.config(text="c'est pas une tomate. Revoyer votre selection")



def show_interface_graphique():
    root = tk.Tk()
    root.title("Prédiction d'images")
    root.geometry("1000x500")  # Set the size of the interface

    # Load the background image
    background_image_path = "C:/Users/Moi/venv/tag/1.png"
    background_image = Image.open(background_image_path)
    background_image = background_image.resize((1000, 500), Image.LANCZOS)  # Use LANCZOS for antialiasing
    background_photo = ImageTk.PhotoImage(background_image)

    # Create a canvas and display the background image
    canvas = tk.Canvas(root, width=1000, height=500)
    canvas.create_image(0, 0, anchor=tk.NW, image=background_photo)
    canvas.pack()

    def open_image():
        if verify_credentials():
            reset_result_label()
            image_path = filedialog.askopenfilename(initialdir='C:/Users/Moi/venv/tag/test',
                                                    title='Select Image',
                                                    filetypes=[('Image Files', ('*.png', '*.jpg', '*.jpeg'))])
            if image_path:
                predict_image(image_path)
            else:
                result_label.config(text="Aucune image sélectionnée.")
        else:
            result_label.config(text="Accès refusé. Veuillez vérifier vos informations d'identification.", fg="red")

    def verify_credentials():
        # Récupérer les informations d'identification saisies par l'utilisateur
        username = username_entry.get()
        password = password_entry.get()

        # Vérifier si les informations d'identification sont correctes
        return username == "1111" and password == "1111"

    # Entrées pour le nom d'utilisateur et le mot de passe
    username_label = tk.Label(root, text="Nom d'utilisateur:", fg="black")
    username_label.pack()
    username_entry = tk.Entry(root)
    username_entry.pack()

    password_label = tk.Label(root, text="Mot de passe:", fg="black")
    password_label.pack()
    password_entry = tk.Entry(root, show="*")
    password_entry.pack()

    # Bouton de connexion
    login_button = tk.Button(root, text="Connexion", command=open_image)
    login_button.pack()

    global result_label
    result_label = tk.Label(root, text="")
    result_label.pack()

    root.mainloop()

if __name__ == "__main__":
    show_interface_graphique()