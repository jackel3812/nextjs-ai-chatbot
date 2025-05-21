#!/bin/bash
# Start the Flask backend
cd /app
(cd /app && python -m gunicorn --bind 0.0.0.0:5000 riley_ai.app:app) &

# Start the Next.js frontend
cd /app && pnpm start

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
