import mysql.connector

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="",
            password="",
            database="database_name"
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Erreur de connexion: {err}")
        return None