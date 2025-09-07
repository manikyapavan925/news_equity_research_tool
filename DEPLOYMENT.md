# News Equity Research Tool - Deployment Guide

## Option 1: Streamlit Cloud (Recommended)

For the best experience with Streamlit apps, deploy to Streamlit Cloud:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository: `manikyapavan925/news_equity_research_tool`
4. Set the main file path: `src/app.py`
5. Deploy

## Option 2: Heroku

Deploy to Heroku for more resources:

1. Create a Heroku app
2. Add Python buildpack
3. Deploy from GitHub

## Option 3: Railway

Deploy to Railway for better ML model support:

1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy automatically

## Files Structure

- `src/app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `Procfile` - For Heroku deployment
- `runtime.txt` - Python version specification

## Environment Variables

Set these environment variables in your deployment platform:

- `PYTORCH_NO_CUDA=1`
- `TRANSFORMERS_OFFLINE=0`
- `TORCH_DEVICE=cpu`

## Note on Vercel

Vercel has limitations for ML/AI applications:
- 50MB function size limit
- Memory constraints on Hobby plan
- Cold start issues with large models

For production use, consider the alternatives above.
