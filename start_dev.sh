#!/bin/bash

# Facial Recognition App Development Server Script
# This script activates the virtual environment and starts the app in development mode

set -e  # Exit on any error

echo "Starting Facial Recognition App (Development Mode)..."

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

# Start the application in development mode
echo "Starting Flask app in development mode..."
echo "App will be available at: http://0.0.0.0:5000"
echo "Press Ctrl+C to stop the server"

# Run the Flask development server
python app.py