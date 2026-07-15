import os
from google.cloud import firestore
from dotenv import load_dotenv

load_dotenv()

class FirestoreService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirestoreService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        project_id = os.getenv("NEXT_PUBLIC_FIREBASE_PROJECT_ID")
        if project_id:
            self.db = firestore.Client(project=project_id)
        else:
            try:
                self.db = firestore.Client()
            except Exception as e:
                print(f"Warning: Could not initialize Firestore client. {e}")
                self.db = None

    def get_db(self):
        return self.db
