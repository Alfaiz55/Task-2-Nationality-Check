import tkinter as tk
from tkinter import filedialog, Label, Button
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk

# Load the trained model
from tensorflow.keras.models import model_from_json

# Load the architecture of the model 
with open("nationality_model_arch.json", "r") as json_file:
    model = model_from_json(json_file.read())

# Load the model weights(.h5 file)
model.load_weights("nationality_model.weights.h5")

print("✅ Model loaded successfully in TF 2.x!")





class_names = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic',
               'Middle Eastern', 'Southeast Asian', 'White']

# Function to upload and display image and reading the path
def upload_image():
    global file_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp")])
    if file_path:
        img = Image.open(file_path).resize((250, 250))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        result_label.config(text="")  # Clear previous result

# Function to predict nationality of the input image
def predict_image():
    if not file_path:
        result_label.config(text="Please image first!", fg="red")
        return

    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    idx = np.argmax(prediction)
    # creating the output/prediction line so that model looking Easy to understand
    result = f"This person belongs to maybe: {class_names[idx]} ({prediction[idx]*100:.2f}%)"
    result_label.config(text=result, fg="green")

# Tkinter GUI setup for Nationality Detection
root = tk.Tk()
root.title("Nationality Detection")# name of the GUI
root.geometry("400x400")

panel = Label(root)
panel.pack()

upload_btn = Button(root, text="Upload Image", command=upload_image)# upload button for selecting the images for prediction
upload_btn.pack()

predict_btn = Button(root, text="Predict", command=predict_image)# creating the predict button to ensure that the model starting the prediction
predict_btn.pack()

result_label = Label(root, text="", font=("Arial", 12))
result_label.pack()

file_path = ""

root.mainloop()
