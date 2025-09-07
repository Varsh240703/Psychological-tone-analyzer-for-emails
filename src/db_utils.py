import mysql.connector
from mysql.connector import Error

def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',        # or your cloud db address
            user='root',
            password='Varsh@24',
            database='tone_analyzer_db'
        )
    except Error as e:
        print(f"Error: '{e}' occurred while connecting to MySQL")
    return connection

def insert_email_analysis(email_text, detected_emotion, detected_tone, sentiment_score, formality_score, suggestions):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        query = """
        INSERT INTO email_analysis (email_text, detected_emotion, detected_tone, sentiment_score, formality_score, suggestions)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (email_text, detected_emotion, detected_tone, sentiment_score, formality_score, suggestions)
        try:
            cursor.execute(query, values)
            connection.commit()
            print("Data inserted successfully!")
        except Error as e:
            print(f"Error: '{e}' occurred while inserting data")
        finally:
            cursor.close()
            connection.close()
