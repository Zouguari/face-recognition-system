import cv2
import face_recognition
import pickle
import datetime
from db_config import get_db_connection
import numpy as np

# Pour éviter les doublons
last_detected = {}

def load_encodings():
    print("[INFO] Chargement des encodages...")
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.loads(f.read())
        return data
    except Exception as e:
        print(f"[ERREUR] Chargement des encodages: {e}")
        return {"encodings": [], "names": []}

def employee_exists(name):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM employees WHERE name = %s", (name,))
            return cursor.fetchone() is not None
        except Exception as e:
            print(f"Erreur vérification employé: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    return False

def record_attendance(employee_name):
    current_time = datetime.datetime.now()
    
    # Vérifier si déjà détecté il y a moins de 5 minutes
    if employee_name in last_detected:
        time_diff = (current_time - last_detected[employee_name]).total_seconds()
        if time_diff < 300:  # 5 minutes
            print(f"{employee_name} déjà pointé récemment")
            return False
    
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            # Vérifier d'abord si l'employé existe
            if not employee_exists(employee_name):
                print(f"ERREUR: {employee_name} non enregistré dans la base")
                return False
                
            cursor.execute("""
                INSERT INTO attendance (employee_name, check_in_time)
                VALUES (%s, %s)
            """, (employee_name, current_time))
            conn.commit()
            
            last_detected[employee_name] = current_time
            print(f"ACCÈS AUTORISÉ: {employee_name}")
            return True
            
        except Exception as e:
            print(f"Erreur d'enregistrement: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    return False

def main():
    data = load_encodings()
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Inconnu"
            
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                
                name = max(counts, key=counts.get)
            
            names.append(name)
            
            if name != "Inconnu":
                record_attendance(name)
        
        for ((top, right, bottom, left), name) in zip(boxes, names):
            color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            status = "ACCES AUTORISE" if name != "Inconnu" else "ACCES REFUSE"
            cv2.putText(frame, f"{name} - {status}", (left, top - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("Systeme de Pointage", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()