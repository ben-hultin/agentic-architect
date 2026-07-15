from typing import Optional
from models import CyclicalReport, CyclicalJobConfig, NodeMetric
from services.firestore_service import FirestoreService
from google.cloud import firestore

class CyclicalDAO:
    def __init__(self):
        self.db = FirestoreService().get_db()

    def save_report(self, report: CyclicalReport):
        if not self.db:
            print("Firestore DB not initialized. Skipping save_report.")
            return
        doc_ref = self.db.collection("reports").document(report.id)
        
        report_dict = report.model_dump() if hasattr(report, 'model_dump') else report.dict()
        
        metrics = report_dict.get("metrics", {})
        mapped_metrics = {
            "performance": {
                "latency": metrics.get("total_latency", 0),
                "ttft": 0, 
                "tokenCount": metrics.get("total_tokens", 0),
                "cost": metrics.get("total_cost", 0)
            },
            "appLayer": {
                "trajectoryDrift": metrics.get("trajectory_drift", 0),
                "cascadingTokenConsumption": metrics.get("cascading_token_consumption", 0),
                "threadRecovery": metrics.get("thread_recovery_latency", 0)
            }
        }
        
        report_data = {
            "id": report.id,
            "jobId": report.job_id,
            "metrics": mapped_metrics,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        
        doc_ref.set(report_data)

    def create_job(self, config: CyclicalJobConfig):
        if not self.db:
            print("Firestore DB not initialized. Skipping create_job.")
            return
        doc_ref = self.db.collection("jobs").document(config.job_id)
        job_data = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
        job_data["status"] = "QUEUED"
        job_data["createdAt"] = firestore.SERVER_TIMESTAMP
        job_data["updatedAt"] = firestore.SERVER_TIMESTAMP
        doc_ref.set(job_data)

    def update_job_status(self, job_id: str, status: str, metrics: Optional[dict] = None):
        if not self.db:
            print("Firestore DB not initialized. Skipping update_job_status.")
            return
        doc_ref = self.db.collection("jobs").document(job_id)
        update_data = {"status": status, "updatedAt": firestore.SERVER_TIMESTAMP}
        if metrics:
            update_data.update(metrics)
        doc_ref.update(update_data)

    def update_live_metrics(self, job_id: str, step_metric: NodeMetric, current_step: int):
        if not self.db:
            print("Firestore DB not initialized. Skipping update_live_metrics.")
            return
        doc_ref = self.db.collection("jobs").document(job_id)
        update_data = {
            "liveMetrics": firestore.ArrayUnion([step_metric.model_dump() if hasattr(step_metric, 'model_dump') else step_metric.dict()]),
            "currentStep": current_step,
            "updatedAt": firestore.SERVER_TIMESTAMP
        }
        doc_ref.update(update_data)

    def get_job_config(self, job_id: str) -> Optional[CyclicalJobConfig]:
        if not self.db:
            print("Firestore DB not initialized. Skipping get_job_config.")
            return None
        doc_ref = self.db.collection("jobs").document(job_id)
        doc = doc_ref.get()
        if doc.exists:
            return CyclicalJobConfig(**doc.to_dict())
        return None
