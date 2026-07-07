#!/bin/bash

# Exit on error
set -e

# Load environment variables if .env.local exists
if [ -f dashboard/.env.local ]; then
  export $(grep -v '^#' dashboard/.env.local | xargs)
fi

PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
LOCATION=${GCP_LOCATION:-"us-central1"}
QUEUE_NAME=${GCP_QUEUE_NAME:-"benchmarking-queue"}

echo "🚀 Deploying GCP Infrastructure for Project: $PROJECT_ID"

# 1. Enable APIs
echo "Enabling necessary APIs..."
gcloud services enable \
  cloudtasks.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  --project "$PROJECT_ID"

# 2. Create Cloud Tasks Queue
echo "Creating Cloud Tasks queue: $QUEUE_NAME..."
if gcloud tasks queues describe "$QUEUE_NAME" --location "$LOCATION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Queue $QUEUE_NAME already exists. Updating configuration..."
  gcloud tasks queues update "$QUEUE_NAME" \
    --location "$LOCATION" \
    --max-concurrent-dispatches 1 \
    --project "$PROJECT_ID"
else
  gcloud tasks queues create "$QUEUE_NAME" \
    --location "$LOCATION" \
    --max-concurrent-dispatches 1 \
    --project "$PROJECT_ID"
fi

# 3. Set up IAM roles (Optional/Example)
# echo "Setting up IAM roles..."
# gcloud projects add-iam-policy-binding "$PROJECT_ID" \
#   --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
#   --role="roles/cloudtasks.enqueuer"

echo "✅ GCP Infrastructure deployment complete!"
