#!/bin/bash
# Riley-AI startup script for production environments

# Run the helper script to set up the environment
if [ -f /app/hf-setup-helper.sh ]; then
  echo "Running setup helper..."
  bash /app/hf-setup-helper.sh
fi

# Start the Flask backend with Gunicorn
echo "Starting Python backend on port 5000..."
(cd /app && python -m gunicorn --bind 0.0.0.0:5000 --workers 2 riley_ai.app:app) &
BACKEND_PID=$!

# Start the Next.js frontend
echo "Starting Next.js frontend on port 3000..."
cd /app && pnpm start &
FRONTEND_PID=$!

# Function to handle process termination
function cleanup {
  echo "Stopping services..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  exit 0
}

# Register the cleanup function for when Ctrl+C is pressed
trap cleanup SIGINT SIGTERM

echo "Riley-AI is running!"
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:5000"
echo "Press Ctrl+C to stop all services"

# Keep the script running and monitor processes
wait -n $BACKEND_PID $FRONTEND_PID

# If one process exits, stop the other one
cleanup

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
