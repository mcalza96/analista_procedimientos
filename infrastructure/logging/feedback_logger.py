import csv
import os
from datetime import datetime
from config.settings import settings

class FeedbackLogger:
    @staticmethod
    def log_feedback(query, response, rating, details=""):
        file_exists = os.path.isfile(settings.FEEDBACK_FILE)
        
        with open(settings.FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Pregunta", "Respuesta", "Calificación", "Detalle"])
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, query, response[:500], rating, details])
