# Railway Deployment Guide

## Overview
This Flask application is configured for deployment on Railway. Railway is a container-based platform with no function size limits, making it perfect for ML applications with heavy dependencies.

## Prerequisites
1. Railway account (sign up at [railway.app](https://railway.app))
2. GitHub repository with your code
3. All environment variables ready

## Deployment Steps

### 1. Prepare Your Repository
Ensure your repository has:
- ✅ `app.py` (main Flask application)
- ✅ `requirements.txt` (all dependencies)
- ✅ `Procfile` (start command)
- ✅ `railway.json` (optional configuration)

### 2. Deploy on Railway

#### Option A: Deploy from GitHub (Recommended)
1. Go to [railway.app](https://railway.app) and log in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Authorize Railway to access your GitHub account
5. Select your repository
6. Railway will automatically detect Python and start building

#### Option B: Deploy via Railway CLI
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

### 3. Configure Environment Variables

In Railway Dashboard → Your Project → Variables, add:

**Required:**
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_KEY` - Your Supabase anon/service key

**Optional:**
- `HF_TOKEN` or `HUGGINGFACE_TOKEN` - Your HuggingFace token for accessing MedSAM Space (required if Space is private)
- `OPENAI_API_KEY` - For GPT transcription refinement
- `PORT` - Railway sets this automatically, but you can override

### 4. Configure Build Settings

Railway will auto-detect Python, but you can specify:

**Python Version:**
- Railway uses the latest Python 3.x by default
- To specify version, create `runtime.txt` with: `python-3.11`

**Build Command:**
- Railway auto-detects from `Procfile`
- Or specify in Railway Dashboard → Settings → Build

### 5. Monitor Deployment

**View Logs:**
- Railway Dashboard → Deployments → View Logs
- Or use CLI: `railway logs`

**Check Status:**
- Railway Dashboard shows build and deployment status
- Green = Success, Red = Failed

### 6. Access Your Application

Once deployed:
- Railway provides a public URL (e.g., `your-app.railway.app`)
- You can set a custom domain in Railway Dashboard → Settings → Domains

## Configuration Files

### Procfile
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```
- Uses Gunicorn as WSGI server
- Binds to `$PORT` (Railway sets this)
- 2 workers for better performance
- 120s timeout for long-running ML operations

### railway.json
Optional configuration file for Railway-specific settings.

### requirements.txt
Contains all Python dependencies. Railway installs these during build.

## Key Differences from Vercel

| Feature | Vercel | Railway |
|---------|--------|---------|
| **Function Size Limit** | 2GB (hard limit) | No limit (container-based) |
| **Cold Starts** | Yes (serverless) | No (always running) |
| **File Storage** | `/tmp` (ephemeral) | Persistent filesystem |
| **Deployment** | Serverless functions | Container-based |
| **Scaling** | Automatic | Manual or auto-scaling |

## Advantages of Railway

✅ **No Size Limits**: Can deploy large ML models without issues
✅ **Persistent Storage**: Files persist between requests
✅ **No Cold Starts**: Always running, faster response times
✅ **Better for ML**: Designed for long-running processes
✅ **Simple Configuration**: Auto-detects Python, Flask, etc.

## Troubleshooting

### Build Fails
- Check logs in Railway Dashboard
- Verify `requirements.txt` has all dependencies
- Ensure Python version is compatible

### App Crashes on Start
- Check `Procfile` syntax
- Verify `app:app` matches your Flask app variable
- Check environment variables are set

### Timeout Errors
- Increase timeout in `Procfile`: `--timeout 300`
- Or adjust in Railway Dashboard → Settings

### Memory Issues
- Railway free tier: 512MB RAM
- Upgrade plan for more memory if needed
- Consider lazy loading models

## Environment Variables Reference

```bash
# Supabase (Required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key

# OpenAI (Optional - for GPT refinement)
OPENAI_API_KEY=sk-...

# Railway sets automatically
PORT=3000  # Don't set manually
```

## Post-Deployment

### 1. Test Endpoints
```bash
# Health check
curl https://your-app.railway.app/

# Test API
curl https://your-app.railway.app/api/get-image
```

### 2. Monitor Performance
- Railway Dashboard → Metrics
- Check CPU, Memory, Network usage
- Monitor response times

### 3. Set Up Custom Domain (Optional)
1. Railway Dashboard → Settings → Domains
2. Add your domain
3. Configure DNS as instructed

## Support

- Railway Docs: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- Railway Status: [status.railway.app](https://status.railway.app)

