# Nationality Detection using Deep Learning (Task 2 - NullClass Internship)

## 📌 Project Overview
This project is part of the NullClass internship. It is a GUI-based deep learning application that detects a person's **nationality** from their image using a CNN model trained on the **FairFace dataset**.

---

## 🚀 Features
- Predicts nationality from uploaded face images.
- Simple Tkinter-based GUI.
- Pre-trained model for instant predictions.
- Batch testing support for multiple images.

---

## 🗂️ File Structure
- `app.py` → Main GUI application.
- `nationality_model_arch.json` → Model architecture.
- `nationality_model.weights.h5` → Model weights.
- `nationality_model_fixed.h5` → Combined model file (backup).
- `nationality-check.ipynb` → Model training notebook.
- `requirements.txt` → Required dependencies.

---

## 🔧 Installation & Setup
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>

Create a virtual environment and activate it:
python -m venv venv_tf3
venv_tf3\Scripts\activate      # (for the Windows)

Install dependencies:
pip install -r requirements.txt

How to run::
python app.gui #Run this cmd on the terminal or cmd promot

✅ Requirements
Python 3.10
TensorFlow / Keras
OpenCV
Pillow (PIL)

