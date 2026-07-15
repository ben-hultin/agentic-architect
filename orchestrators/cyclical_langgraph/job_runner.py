import argparse
import json
import datetime
from models import CyclicalJobConfig
from orchestrator import CyclicalOrchestrator
from dao.cyclical_dao import CyclicalDAO

def main():
    parser = argparse.ArgumentParser(description="Run a Cyclical LangGraph Orchestrator Job from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the static job config JSON file.")
    args = parser.parse_args()

    print(f"Loading job configuration from: {args.config}")
    try:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
    except Exception as e:
        print(f"Failed to load config file: {e}")
        return

    base_job_id = config_dict.get("job_id", "Unknown-Job")
    run_id = f"{base_job_id}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    config_dict["job_id"] = run_id

    config = CyclicalJobConfig(**config_dict)
    dao = CyclicalDAO()

    print(f"Creating job {run_id} in Firestore...")
    dao.create_job(config)

    print(f"Executing job {run_id}...")
    dao.update_job_status(run_id, "RUNNING")

    try:
        orchestrator = CyclicalOrchestrator(config)
        report = orchestrator.run()

        dao.save_report(report)
        
        last_metric = report.metrics.node_metrics[-1] if report.metrics.node_metrics else None
        final_status = "COMPLETED" if last_metric and last_metric.status != "FAILED" else "FAILED"
        
        dao.update_job_status(run_id, final_status, {
            "successRate": 100 if final_status == "COMPLETED" else 0,
            "cost": report.metrics.total_cost
        })
        print(f"Job {run_id} finished with status: {final_status}")

    except Exception as e:
        print(f"Job {run_id} failed: {e}")
        dao.update_job_status(run_id, "FAILED")

if __name__ == "__main__":
    main()
