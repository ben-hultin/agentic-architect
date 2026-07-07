#!/bin/bash

# Exit on error
set -e

# Load environment variables if .env.local exists
if [ -f dashboard/.env.local ]; then
  export $(grep -v '^#' dashboard/.env.local | xargs)
fi

PROJECT_ID=${NEXT_PUBLIC_FIREBASE_PROJECT_ID:-$(firebase target:list | grep "project: " | sed 's/.*project: //')}

echo "🔥 Deploying Firebase Infrastructure for Project: $PROJECT_ID"

# 1. Initialize Firebase if not already done
if [ ! -f firebase.json ]; then
  echo "Initializing Firebase configuration..."
  firebase init firestore storage --project "$PROJECT_ID"
fi

# 2. Deploy Firestore Rules and Indexes
echo "Deploying Firestore rules and indexes..."
firebase deploy --only firestore --project "$PROJECT_ID"

# 3. Deploy Storage Rules
echo "Deploying Storage rules..."
firebase deploy --only storage --project "$PROJECT_ID"

echo "✅ Firebase Infrastructure deployment complete!"
