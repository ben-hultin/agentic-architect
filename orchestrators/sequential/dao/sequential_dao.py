from typing import Optional
from ..models import SequentialReport, SequentialJobConfig
from ..services.firestore_service import FirestoreService

class SequentialDAO:
    def __init__(self):
        self.db = FirestoreService().get_db()

    def save_report(self, report: SequentialReport):
        doc_ref = self.db.collection("reports").document(report.id)
        doc_ref.set(report.dict())

    def update_job_status(self, job_id: str, status: str, metrics: Optional[dict] = None):
        doc_ref = self.db.collection("jobs").document(job_id)
        update_data = {"status": status, "updatedAt": firestore.SERVER_TIMESTAMP}
        if metrics:
            update_data.update(metrics)
        doc_ref.update(update_data)

    def get_job_config(self, job_id: str) -> Optional[SequentialJobConfig]:
        doc_ref = self.db.collection("jobs").document(job_id)
        doc = doc_ref.get()
        if doc.exists:
            return SequentialJobConfig(**doc.to_dict())
        return None
