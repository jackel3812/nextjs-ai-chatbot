# Setting Up Riley-AI on Hugging Face Spaces

This guide provides step-by-step instructions for deploying Riley-AI to Hugging Face Spaces.

## Prerequisites

Before you begin, make sure you have the following:

1. A [Hugging Face](https://huggingface.co/) account
2. The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start) installed (`pip install huggingface_hub`)
3. Git installed on your local machine
4. A local clone of the Riley-AI repository

## Setup Steps

### 1. Authentication

First, log in to Hugging Face using the CLI:

```bash
huggingface-cli login
```

Follow the prompts to authenticate with your Hugging Face account.

### 2. Create a New Space

Create a new Hugging Face Space for Riley-AI:

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Enter "riley-ai" as the Space name
4. Select "Docker" as the SDK
5. Choose "Public" or "Private" visibility as preferred
6. Click "Create Space"

### 3. Set Environment Variables

Set the following environment variables in your Hugging Face Space:

1. Go to your Space settings
2. Navigate to the "Variables" tab
3. Add the following variables:
   - `NEXTAUTH_SECRET`: A random string for NextAuth.js (generate with `openssl rand -base64 32`)
   - Any other environment variables your application needs

### 4. Deploy Using the Script

The repository includes a deployment script that will handle pushing your code to Hugging Face Spaces:

1. Open the `deploy-to-hf.sh` file and replace `YOUR_HF_USERNAME` with your actual Hugging Face username:
   ```bash
   # Edit this line in deploy-to-hf.sh
   HF_USERNAME="your-username"  # Replace with your Hugging Face username
   ```

2. Make the script executable:
   ```bash
   chmod +x deploy-to-hf.sh
   ```

3. Run the deployment script:
   ```bash
   ./deploy-to-hf.sh
   ```

This script will:
- Add your Hugging Face Space as a Git remote
- Create a deployment branch
- Push your code to the Space
- Trigger the build process on Hugging Face

### 5. Monitor the Build

After pushing your code:

1. Go to your Hugging Face Space page
2. Click on "Settings" and then "CI/CD"
3. Monitor the build logs to ensure everything is working correctly

The build process will:
- Install dependencies for both Python and Next.js
- Build the Next.js frontend
- Start both services using the app-start.sh script

### 6. Access Your Deployment

Once the build is complete, you can access your Riley-AI instance at:
`https://[YOUR-USERNAME]-riley-ai.hf.space`

## Troubleshooting

### Build Failures

If your build fails, check the following:

1. CI/CD logs on Hugging Face for specific error messages
2. Ensure all required environment variables are set
3. Verify that your Dockerfile and app-start.sh script are properly configured
4. Check that the huggingface.json file has your correct username

### Runtime Issues

If the application builds but doesn't work correctly:

1. Check the application logs in your Space
2. Ensure the Flask backend is running on port 5000
3. Verify that the Next.js frontend can communicate with the backend

## Updating Your Deployment

To update your deployment after making changes:

1. Make your changes locally
2. Run the deployment script again:
   ```bash
   ./deploy-to-hf.sh
   ```

## Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces-overview)
- [Docker on Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Next.js Documentation](https://nextjs.org/docs)
