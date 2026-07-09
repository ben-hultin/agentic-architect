import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from models import SequentialJobConfig
from orchestrator import SequentialOrchestrator
from dao.sequential_dao import SequentialDAO

app = FastAPI()

class JobRequest(BaseModel):
    jobId: str

def run_orchestrator(job_id: str):
    dao = SequentialDAO()
    
    # 1. Fetch Job Config
    config = dao.get_job_config(job_id)
    if not config:
        print(f"Job config for {job_id} not found.")
        return

    # 2. Update status to RUNNING
    dao.update_job_status(job_id, "RUNNING")

    try:
        # 3. Initialize Orchestrator
        orchestrator = SequentialOrchestrator(config)

        # 4. Run Benchmark
        report = orchestrator.run()

        # 5. Save Report
        dao.save_report(report)

        # 6. Update Job status to COMPLETED with high-level metrics
        dao.update_job_status(job_id, "COMPLETED", {
            "successRate": report.metrics.chain_recovery_efficiency * 100,
            "cost": report.metrics.total_cost
        })
        
        print(f"Job {job_id} completed successfully.")

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        dao.update_job_status(job_id, "FAILED")

@app.post("/run")
async def run_job(request: JobRequest, background_tasks: BackgroundTasks):
    """
    Endpoint triggered by Cloud Tasks to start a benchmarking job.
    """
    background_tasks.add_task(run_orchestrator, request.jobId)
    return {"message": "Job enqueued for execution", "jobId": request.jobId}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
