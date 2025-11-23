import csv
import os
import logging
from datetime import datetime
from config.settings import settings
from core.interfaces.feedback_repository import FeedbackRepository
from infrastructure.constants import FEEDBACK_HEADERS, DATE_FORMAT

logger = logging.getLogger(__name__)

class FeedbackLogger(FeedbackRepository):
    def log_feedback(self, query: str, response: str, rating: str, details: str = ""):
        try:
            file_exists = os.path.isfile(settings.FEEDBACK_FILE)
            
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(settings.FEEDBACK_FILE), exist_ok=True)
            
            with open(settings.FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(FEEDBACK_HEADERS)
                
                timestamp = datetime.now().strftime(DATE_FORMAT)
                writer.writerow([timestamp, query, response[:500], rating, details])
                
            logger.info(f"Feedback registrado: {rating}")
        except Exception as e:
            logger.error(f"Error registrando feedback: {e}")
