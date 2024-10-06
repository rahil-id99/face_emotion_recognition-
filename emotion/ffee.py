import cv2
import os
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox
from fer import FER  # Import FER for emotion detection

# List of four members to recognize
members = ["tharun", "praveen", "bhaskar", "rahil"]

# Dictionary to map member names to specific IDs (1 to 4)
names = {i+1: members[i] for i in range(len(members))}

# Step 1: Capture Faces
def capture_faces(name, save_dir='faces'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if name not in members:
        messagebox.showwarning("Warning", "Name not in the list of recognized members!")
        return

    id = members.index(name) + 1
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    sample_num = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_num += 1
            cv2.imwrite(f"{save_dir}/{id}_{sample_num}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sample_num >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()

# Step 2: Train the Recognizer
def train_faces(data_dir='faces'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []

    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    for image_path in image_paths:
        filename = os.path.basename(image_path)

        try:
            id = int(filename.split("_")[0])
        except ValueError:
            print(f"Filename {filename} is not in the expected format!")
            continue

        gray_img = Image.open(image_path).convert('L')
        img_numpy = np.array(gray_img, 'uint8')

        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    ids = np.array(ids, dtype=np.int32)
    recognizer.train(face_samples, ids)
    recognizer.write('trainer.yml')

# Step 3: Recognize Faces and Emotions in Real-time
def recognize_faces_and_emotions():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    emotion_detector = FER()  # Initialize the emotion detector

    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100 and id in names:
                name = names[id]
                confidence_text = f"  {round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"  {round(100 - confidence)}%"

            # Predict emotions using FER
            emotion_prediction = emotion_detector.detect_emotions(img[y:y+h, x:x+w])
            emotion_text = "Emotion: Unknown"
            if emotion_prediction:
                top_emotion = emotion_prediction[0]['emotions']
                emotion_text = f"Emotion: {max(top_emotion, key=top_emotion.get)}"

            # Display the name, confidence, and emotion
            cv2.putText(img, f"{name} {confidence_text}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, emotion_text, (x+5, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Face Recognition and Emotion Detection', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Step 4: Tkinter GUI
def start_capture():
    name = name_entry.get()
    if name:
        capture_faces(name)
        messagebox.showinfo("Info", "Face captured successfully!")
    else:
        messagebox.showwarning("Warning", "Please enter a name.")

def start_training():
    train_faces()
    messagebox.showinfo("Info", "Training completed!")

def start_recognition():
    recognize_faces_and_emotions()

# Set up the GUI
root = tk.Tk()
root.title("Face Recognition and Emotion Detection System")
root.configure(bg='light blue')

tk.Label(root, text="Enter your name:", bg='light blue', fg='white').pack(pady=10)
name_entry = tk.Entry(root)
name_entry.pack(pady=5)

capture_button = tk.Button(root, text="Capture Face", command=start_capture, bg='blue', fg='white')
capture_button.pack(pady=10)

train_button = tk.Button(root, text="Train Faces", command=start_training, bg='blue', fg='white')
train_button.pack(pady=10)

recognize_button = tk.Button(root, text="Recognize Faces and Emotions", command=start_recognition, bg='blue', fg='white')
recognize_button.pack(pady=10)

root.mainloop()
