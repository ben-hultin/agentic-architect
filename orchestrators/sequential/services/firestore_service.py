import os
from google.cloud import firestore
from dotenv import load_dotenv

load_dotenv()

class FirestoreService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirestoreService, cls).__new__(cls)
            project_id = os.getenv("NEXT_PUBLIC_FIREBASE_PROJECT_ID")
            cls._instance.db = firestore.Client(project=project_id)
        return cls._instance

    def get_db(self):
        return self.db
