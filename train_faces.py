import cv2
import os
import numpy as np
from imutils import paths
import pickle
import face_recognition
from db_config import get_db_connection

def train_faces():
    print("[INFO] Encodage des visages...")
    imagePaths = list(paths.list_images("dataset"))
    
    knownEncodings = []
    knownNames = []
    
    for (i, imagePath) in enumerate(imagePaths):
        print(f"Traitement de l'image {i+1}/{len(imagePaths)}")
        name = imagePath.split(os.path.sep)[-2].split("_")[1]
        
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    
    print("[INFO] Sérialisation des encodages...")
    data = {"encodings": knownEncodings, "names": knownNames}
    
    with open("encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))
    
    print("[INFO] Mise à jour des face_id dans la base de données...")
    update_face_ids(knownEncodings, knownNames)

def update_face_ids(encodings, names):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            for name, encoding in zip(names, encodings):
                # Convertir l'encodage en format binaire pour MySQL
                encoding_bytes = pickle.dumps(encoding)
                cursor.execute("UPDATE employees SET face_encoding = %s WHERE name = %s", 
                              (encoding_bytes, name))
            conn.commit()
            print("Mise à jour des face_id terminée.")
        except Exception as e:
            print(f"Erreur lors de la mise à jour: {e}")
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    train_faces()