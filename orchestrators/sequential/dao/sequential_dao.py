from typing import Optional
from models import SequentialReport, SequentialJobConfig, StepMetric
from services.firestore_service import FirestoreService
from google.cloud import firestore

class SequentialDAO:
    def __init__(self):
        self.db = FirestoreService().get_db()

    def save_report(self, report: SequentialReport):
        doc_ref = self.db.collection("reports").document(report.id)
        doc_ref.set(report.dict())

    def create_job(self, config: SequentialJobConfig):
        doc_ref = self.db.collection("jobs").document(config.job_id)
        job_data = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
        job_data["status"] = "QUEUED"
        job_data["createdAt"] = firestore.SERVER_TIMESTAMP
        job_data["updatedAt"] = firestore.SERVER_TIMESTAMP
        doc_ref.set(job_data)

    def update_job_status(self, job_id: str, status: str, metrics: Optional[dict] = None):
        doc_ref = self.db.collection("jobs").document(job_id)
        update_data = {"status": status, "updatedAt": firestore.SERVER_TIMESTAMP}
        if metrics:
            update_data.update(metrics)
        doc_ref.update(update_data)

    def update_live_metrics(self, job_id: str, step_metric: StepMetric, current_step: int):
        doc_ref = self.db.collection("jobs").document(job_id)
        update_data = {
            "liveMetrics": firestore.ArrayUnion([step_metric.model_dump() if hasattr(step_metric, 'model_dump') else step_metric.dict()]),
            "currentStep": current_step,
            "updatedAt": firestore.SERVER_TIMESTAMP
        }
        doc_ref.update(update_data)

    def get_job_config(self, job_id: str) -> Optional[SequentialJobConfig]:
        doc_ref = self.db.collection("jobs").document(job_id)
        doc = doc_ref.get()
        if doc.exists:
            return SequentialJobConfig(**doc.to_dict())
        return None
