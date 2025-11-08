#!/bin/bash

# Facial Recognition App Startup Script
# This script activates the virtual environment and starts the app with Gunicorn

set -e  # Exit on any error

echo "Starting Facial Recognition App..."

# Change to the project directory (adjust path as needed)
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found!"
    echo "Please create it first with: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if Gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "Installing Gunicorn..."
    pip install gunicorn
fi

# Start the application with Gunicorn
echo "Starting Flask app with Gunicorn..."
echo "App will be available at: http://0.0.0.0:8000"
echo "Press Ctrl+C to stop the server"

# Run with 4 workers, binding to all interfaces on port 8000
gunicorn -w 4 -b 0.0.0.0:8000 app:app --access-logfile - --error-logfile -