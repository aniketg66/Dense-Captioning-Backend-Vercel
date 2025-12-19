# Vercel Deployment Guide for Flask App

This guide provides detailed instructions for deploying the Flask application to Vercel.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Configuration Files](#configuration-files)
4. [Environment Variables](#environment-variables)
5. [Deployment Steps](#deployment-steps)
6. [Testing Locally](#testing-locally)
7. [Troubleshooting](#troubleshooting)
8. [Important Considerations](#important-considerations)

## Prerequisites

Before deploying, ensure you have:

- ✅ A Vercel account (sign up at [vercel.com](https://vercel.com))
- ✅ Vercel CLI installed (`npm i -g vercel`)
- ✅ Git repository (GitHub, GitLab, or Bitbucket)
- ✅ All required API keys and credentials

## Pre-Deployment Checklist

- [x] ✅ Created `vercel.json` configuration file
- [x] ✅ Created `.vercelignore` to exclude unnecessary files
- [x] ✅ Created `api/index.py` serverless function wrapper
- [x] ✅ Updated `app.py` to use `/tmp` for file storage in Vercel
- [x] ✅ Updated `requirements.txt` with all dependencies
- [x] ✅ Made OpenAI client initialization optional
- [x] ✅ Tested app locally

## Configuration Files

### 1. `vercel.json`

This file configures Vercel to use Python runtime and sets up routing:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ],
  "env": {
    "PYTHON_VERSION": "3.11"
  },
  "functions": {
    "api/index.py": {
      "maxDuration": 60
    }
  }
}
```

**Key Points:**
- Uses Python 3.11 runtime
- Maximum function duration: 60 seconds (Pro plan) or 10 seconds (Hobby plan)
- All routes are proxied to the Flask app

### 2. `.vercelignore`

Excludes unnecessary files from deployment to reduce build size:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments
- Test files
- Large model files (`.pth`, `.pkl`)
- Local uploads and transcriptions

### 3. `api/index.py`

Serverless function wrapper that imports and exports the Flask app:

```python
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import app
```

## Environment Variables

Set these environment variables in Vercel Dashboard → Project Settings → Environment Variables:

### Required Variables

```bash
# OpenAI API (for GPT transcription refinement)
OPENAI_API_KEY=sk-...

# Supabase Configuration
REACT_APP_SUPABASE_URL=https://your-project.supabase.co
REACT_APP_SUPABASE_ANON_KEY=your-anon-key

# Optional: MedSAM Image ID
MEDSAM_IMAGE_ID=your-image-id
```

### How to Set Environment Variables

**Option 1: Via Vercel Dashboard**
1. Go to your project in Vercel Dashboard
2. Navigate to Settings → Environment Variables
3. Add each variable for Production, Preview, and Development environments

**Option 2: Via Vercel CLI**
```bash
vercel env add OPENAI_API_KEY
vercel env add REACT_APP_SUPABASE_URL
vercel env add REACT_APP_SUPABASE_ANON_KEY
```

## Deployment Steps

### Method 1: Deploy via Vercel Dashboard (Recommended)

1. **Push to Git Repository**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Import Project in Vercel**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import your Git repository
   - Vercel will auto-detect Python configuration

3. **Configure Environment Variables**
   - Add all required environment variables (see above)
   - Set them for Production, Preview, and Development

4. **Deploy**
   - Click "Deploy"
   - Wait for build to complete
   - Your app will be live at `https://your-project.vercel.app`

### Method 2: Deploy via Vercel CLI

1. **Install Vercel CLI** (if not already installed)
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```
   - Follow prompts to link project
   - Set environment variables when prompted

4. **Deploy to Production**
   ```bash
   vercel --prod
   ```

## Testing Locally

Before deploying, test the app locally:

### 1. Run Local Tests

```bash
python3 test_app_local.py
```

This will verify:
- ✅ App imports successfully
- ✅ Routes are registered
- ✅ Configuration is correct
- ✅ Vercel compatibility (uses `/tmp` for storage)

### 2. Test with Vercel Environment Simulation

```bash
# Set Vercel environment variable
export VERCEL=1

# Run the app
python3 app.py
```

### 3. Test Serverless Function Locally

```bash
# Install Vercel CLI dev server
vercel dev
```

This runs a local server that simulates Vercel's serverless environment.

## Troubleshooting

### Issue: Build Fails - "Module not found"

**Solution:** Ensure all dependencies are in `requirements.txt`:
```bash
pip freeze > requirements.txt
```

### Issue: Function Timeout

**Causes:**
- Heavy ML model loading (Whisper, MaskFormer)
- Long-running operations

**Solutions:**
1. Use Pro plan (60s timeout vs 10s on Hobby)
2. Optimize model loading (lazy initialization)
3. Use external APIs for heavy operations (already using HuggingFace Space)

### Issue: File System Errors

**Cause:** Vercel has read-only filesystem except `/tmp`

**Solution:** Already handled - app uses `/tmp` directory when `VERCEL` env var is set

### Issue: Import Errors

**Solution:** Check that `api/index.py` correctly adds parent directory to `sys.path`

### Issue: Large Deployment Size

**Causes:**
- Large ML models
- Unnecessary files included

**Solutions:**
1. Check `.vercelignore` includes all unnecessary files
2. Consider using external APIs instead of local models
3. Use Vercel's build optimization

### Issue: Environment Variables Not Working

**Solution:**
1. Verify variables are set in Vercel Dashboard
2. Redeploy after adding variables
3. Check variable names match exactly (case-sensitive)

## Important Considerations

### ⚠️ Limitations

1. **Function Timeout**
   - Hobby plan: 10 seconds
   - Pro plan: 60 seconds
   - Enterprise: Custom limits

2. **Deployment Size**
   - Maximum 50MB uncompressed
   - Large ML models may exceed limit
   - Consider using external APIs (already implemented for segmentation)

3. **Cold Starts**
   - First request may be slow due to model loading
   - Consider using Vercel Pro for better performance

4. **File Storage**
   - `/tmp` directory is ephemeral (cleared between invocations)
   - Use external storage (Supabase) for persistent files

5. **Concurrent Requests**
   - Each request spawns a new serverless function
   - Models are loaded per request (consider caching)

### ✅ Optimizations Already Implemented

1. **External APIs for Heavy Operations**
   - Segmentation via HuggingFace Space API
   - No local SAM/MedSAM models

2. **Lazy Model Loading**
   - HuggingFace client initialized on first use
   - Models loaded only when needed

3. **Efficient File Handling**
   - Uses `/tmp` in Vercel environment
   - Temporary files cleaned up automatically

### 📊 Monitoring

After deployment, monitor:

1. **Function Logs**
   - Vercel Dashboard → Functions → View Logs
   - Check for errors and warnings

2. **Performance Metrics**
   - Function duration
   - Memory usage
   - Error rates

3. **API Usage**
   - OpenAI API calls
   - HuggingFace Space API calls
   - Supabase queries

## Post-Deployment

### 1. Verify Deployment

```bash
# Check deployment status
vercel ls

# View logs
vercel logs
```

### 2. Test Endpoints

Test key endpoints:
- `GET /` - Home page
- `GET /api/get-image` - Image retrieval
- `POST /api/segment` - Segmentation
- `POST /api/transcribe` - Audio transcription

### 3. Set Up Custom Domain (Optional)

1. Go to Vercel Dashboard → Settings → Domains
2. Add your custom domain
3. Configure DNS records as instructed

## Additional Resources

- [Vercel Python Documentation](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [Vercel Environment Variables](https://vercel.com/docs/concepts/projects/environment-variables)
- [Flask on Vercel](https://vercel.com/guides/deploying-flask-with-vercel)

## Support

If you encounter issues:

1. Check Vercel function logs
2. Review error messages in deployment logs
3. Test locally with `vercel dev`
4. Check Vercel status page for service issues

---

**Last Updated:** $(date)
**App Version:** 1.0.0
**Python Version:** 3.11

