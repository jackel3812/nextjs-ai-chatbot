#!/bin/bash

# Start the Python backend in the background
echo "Starting Riley-AI Python backend..."
cd "$(dirname "$0")"
(cd riley_ai && FLASK_ENV=development python app.py) &
PYTHON_PID=$!

# Wait a moment for the backend to start
sleep 2

# Start the Next.js frontend in the background
echo "Starting Riley-AI Next.js frontend..."
NEXT_PUBLIC_BACKEND_URL=http://localhost:5000 pnpm dev &
NEXTJS_PID=$!

# Function to handle exit
function cleanup {
  echo "Stopping services..."
  kill $PYTHON_PID
  kill $NEXTJS_PID
  exit 0
}

# Register the cleanup function for when Ctrl+C is pressed
trap cleanup SIGINT

echo "Riley-AI is running!"
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:5000"
echo "Press Ctrl+C to stop all services"

# Keep the script running
wait
