#!/bin/bash

# Simple local test server for Cloud Specialization Track
# Use this to test the dashboard locally before deploying to nginx

echo "🧪 Starting Local Test Server..."
echo "================================"

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python to run the test server."
    exit 1
fi

# Get the current directory
CURRENT_DIR=$(pwd)
echo "📁 Serving from: $CURRENT_DIR"

# Start the server
echo "🌐 Starting local server on http://localhost:8000"
echo ""
echo "📊 Test these URLs:"
echo "   Main Portal: http://localhost:8000/"
echo "   Dashboard: http://localhost:8000/deployment/review_dashboard.html"
echo "   Azure Guide: http://localhost:8000/azure_solution_guide.html"
echo "   CSV Files: http://localhost:8000/deployment/comments_tracking/"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"

# Start Python HTTP server
$PYTHON_CMD -m http.server 8000