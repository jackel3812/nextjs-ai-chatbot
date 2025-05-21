FROM node:18 AS frontend-builder

WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install

COPY . .
RUN pnpm build

FROM python:3.10-slim

WORKDIR /app

# Install Node.js for running the Next.js app
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs npm && \
    npm install -g pnpm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Python backend requirements and install them
COPY riley_ai/requirements.txt ./riley_ai/requirements.txt
RUN pip install --no-cache-dir -r riley_ai/requirements.txt && \
    pip install --no-cache-dir flask-cors gunicorn

# Copy the frontend build from the first stage
COPY --from=frontend-builder /app/.next ./.next
COPY --from=frontend-builder /app/public ./public
COPY --from=frontend-builder /app/package.json ./package.json
COPY --from=frontend-builder /app/next.config.ts ./next.config.ts

# Copy Python backend files
COPY riley_ai ./riley_ai

# Copy the startup script
COPY app-start.sh ./app-start.sh
RUN chmod +x ./app-start.sh

# Expose both ports (Next.js and Flask)
EXPOSE 3000
EXPOSE 5000

# Start both services
CMD ["./app-start.sh"]
