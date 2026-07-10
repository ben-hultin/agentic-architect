import argparse
import json
import uuid
import datetime
from models import SequentialJobConfig
from orchestrator import SequentialOrchestrator
from dao.sequential_dao import SequentialDAO

def main():
    parser = argparse.ArgumentParser(description="Run a Sequential Orchestrator Job from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the static job config JSON file.")
    args = parser.parse_args()

    print(f"Loading job configuration from: {args.config}")
    try:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
    except Exception as e:
        print(f"Failed to load config file: {e}")
        return

    # Generate a unique execution run ID if desired, or use the one from config
    # We will append a timestamp to the base job_id to make each run unique in the dashboard
    base_job_id = config_dict.get("job_id", "Unknown-Job")
    run_id = f"{base_job_id}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    config_dict["job_id"] = run_id

    config = SequentialJobConfig(**config_dict)
    dao = SequentialDAO()

    print(f"Creating job {run_id} in Firestore...")
    dao.create_job(config)

    print(f"Executing job {run_id}...")
    dao.update_job_status(run_id, "RUNNING")

    try:
        orchestrator = SequentialOrchestrator(config)
        report = orchestrator.run()

        dao.save_report(report)
        dao.update_job_status(run_id, "COMPLETED", {
            "successRate": report.metrics.chain_recovery_efficiency * 100,
            "cost": report.metrics.total_cost
        })
        print(f"Job {run_id} completed successfully.")

    except Exception as e:
        print(f"Job {run_id} failed: {e}")
        dao.update_job_status(run_id, "FAILED")

if __name__ == "__main__":
    main()