import cv2
import os
from db_config import get_db_connection  # Import de votre fonction existante

def capture_faces(employee_id, employee_name, num_samples=50):
    # Création du dossier de destination
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    employee_dir = os.path.join(dataset_dir, f"{employee_id}_{employee_name}")
    os.makedirs(employee_dir, exist_ok=True)
    
    # Initialisation de la webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print(f"\nCapture en cours pour {employee_name} (ID: {employee_id})")
    print("Instructions:")
    print("- Appuyez sur 's' pour enregistrer l'image actuelle")
    print("- Appuyez sur 'q' pour terminer la capture\n")
    print(f"Images à capturer: {num_samples} | Images capturées: 0", end='')

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = gray[y:y+h, x:x+w]
            
            # Affichage des instructions
            cv2.putText(frame, "Press 's' to save", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Sauvegarde de l'image
                img_path = os.path.join(employee_dir, f"{count}.jpg")
                cv2.imwrite(img_path, face_roi)
                count += 1
                print(f"\rImages à capturer: {num_samples} | Images capturées: {count}", end='', flush=True)
        
        cv2.imshow('Capture Faciale - Mode Manuel', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Enregistrement dans la base de données
    conn = get_db_connection()  # Utilisation de votre fonction existante
    if conn:
        try:
            cursor = conn.cursor()
            query = "INSERT INTO employees (employee_id, name) VALUES (%s, %s)"
            cursor.execute(query, (employee_id, employee_name))
            conn.commit()
            print(f"\n\nEmployé {employee_name} enregistré avec succès dans la base de données.")
        except Exception as e:
            print(f"\nErreur lors de l'enregistrement en base: {str(e)}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

if __name__ == "__main__":
    print("=== Enregistrement d'un nouvel employé ===")
    emp_id = input("ID employé: ").strip()
    emp_name = input("Nom complet: ").strip()
    capture_faces(emp_id, emp_name)