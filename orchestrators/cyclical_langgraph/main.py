import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from models import CyclicalJobConfig
from orchestrator import CyclicalOrchestrator
from dao.cyclical_dao import CyclicalDAO

app = FastAPI()

class JobRequest(BaseModel):
    jobId: str

def run_orchestrator(job_id: str):
    dao = CyclicalDAO()
    
    config = dao.get_job_config(job_id)
    if not config:
        print(f"Job config for {job_id} not found.")
        return

    dao.update_job_status(job_id, "RUNNING")

    try:
        orchestrator = CyclicalOrchestrator(config)
        report = orchestrator.run()

        dao.save_report(report)

        last_metric = report.metrics.node_metrics[-1] if report.metrics.node_metrics else None
        final_status = "COMPLETED" if last_metric and last_metric.status != "FAILED" else "FAILED"

        dao.update_job_status(job_id, final_status, {
            "successRate": 100 if final_status == "COMPLETED" else 0,
            "cost": report.metrics.total_cost
        })
        
        print(f"Job {job_id} finished with status: {final_status}")

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        dao.update_job_status(job_id, "FAILED")

@app.post("/run")
async def run_job(request: JobRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_orchestrator, request.jobId)
    return {"message": "Job enqueued for execution", "jobId": request.jobId}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
