# Services Setup Guide

## 1. PostgreSQL Database (Neon.tech)
1. Go to https://neon.tech
2. Sign up/Log in
3. Click "Create New Project"
4. Name it "riley-ai"
5. Select the closest region
6. Copy the connection string (it looks like: postgres://user:pass@ep-something.region.aws.neon.tech/neondb)

## 2. Redis (Upstash)
1. Go to https://upstash.com
2. Sign up/Log in
3. Click "Create Database"
4. Name it "riley-ai"
5. Select "Global" region for better performance
6. Copy the Redis connection string (it looks like: redis://default:password@regionname.upstash.io:port)

## 3. Vercel Blob Storage
1. Go to https://vercel.com
2. Sign up/Log in
3. Create a new project (can be empty)
4. Go to Storage > Blob
5. Create a new Blob store
6. Copy the read-write token

## 4. xAI API Key
1. Go to https://console.x.ai
2. Sign up/Log in
3. Navigate to API Keys
4. Create a new API key
5. Copy the key

Please save each credential after obtaining it. We'll use them to set up the Hugging Face deployment.
