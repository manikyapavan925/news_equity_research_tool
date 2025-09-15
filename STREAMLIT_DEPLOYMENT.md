# 🚀 Streamlit Cloud Deployment Guide

This guide walks you through deploying the News Research Assistant to Streamlit Cloud.

## 📋 Prerequisites

- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- Your repository pushed to GitHub

## 🔧 Pre-Deployment Checklist

### ✅ Required Files
Make sure these files are in your repository root:

- `streamlit_app.py` - Main application file
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `packages.txt` - System dependencies (optional)
- `README.md` - Documentation

### ✅ File Structure
```
News_Equity_Research_Tool/
├── streamlit_app.py          # Main app file
├── requirements.txt          # Python packages
├── packages.txt             # System packages
├── README.md               # Documentation
├── .streamlit/
│   └── config.toml         # Streamlit config
└── other files...
```

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository
1. Ensure all changes are committed and pushed to GitHub:
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Choose your repository: `manikyapavan925/news_equity_research_tool`
   - Set branch: `master` (or `main`)
   - Set main file path: `streamlit_app.py`

3. **Configure App Settings**
   - App name: `news-research-assistant`
   - Python version: `3.11` (recommended)
   - Click "Deploy!"

### Step 3: Monitor Deployment

1. **Watch Build Logs**
   - Streamlit Cloud will show real-time build logs
   - Installation of dependencies from `requirements.txt`
   - Installation of system packages from `packages.txt`

2. **Expected Build Time**
   - Initial deployment: 3-5 minutes
   - Subsequent deployments: 1-2 minutes

## ⚙️ Configuration Details

### Requirements.txt
```txt
streamlit>=1.32.0
requests>=2.31.0
beautifulsoup4>=4.12.2
pandas>=1.5.0
numpy>=1.24.0
python-dotenv>=1.0.0
```

### Streamlit Config (.streamlit/config.toml)
```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
runOnSave = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

### System Packages (packages.txt)
```txt
build-essential
python3-dev
libxml2-dev
libxslt-dev
libffi-dev
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. **Build Fails - Dependency Conflicts**
```bash
# Solution: Update requirements.txt with compatible versions
streamlit>=1.32.0  # Use >= instead of ==
requests>=2.31.0
```

#### 2. **App Crashes on Startup**
- Check logs for import errors
- Ensure all imports are available in requirements.txt
- Verify no local file dependencies

#### 3. **Slow Performance**
- Implement caching with `@st.cache_data`
- Optimize data processing functions
- Reduce API calls

#### 4. **Memory Issues**
- Limit article content size
- Clear session state when needed
- Use efficient data structures

## 🔄 Updates & Maintenance

### Automatic Deployments
Streamlit Cloud automatically redeploys when you push to the connected branch:

```bash
# Make changes
git add .
git commit -m "Update feature X"
git push origin master

# App will automatically redeploy
```

### Manual Redeployment
1. Go to your Streamlit Cloud dashboard
2. Find your app
3. Click "Reboot" to restart
4. Click "Redeploy" to rebuild

## 📊 Monitoring & Analytics

### App Metrics
- Monitor usage in Streamlit Cloud dashboard
- Check performance metrics
- Review error logs

### Performance Optimization
```python
# Use caching for expensive operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def extract_article_content(url):
    # Expensive operation
    pass

# Monitor memory usage
import psutil
memory_usage = psutil.virtual_memory().percent
```

## 🔒 Security Best Practices

### Environment Variables
- No API keys needed for this app
- If adding APIs later, use Streamlit secrets:

```toml
# .streamlit/secrets.toml (not committed to git)
api_key = "your_secret_key"
```

### Rate Limiting
- Respect website robots.txt
- Implement delays between requests
- Handle 403/429 errors gracefully

## 🚀 Going Live

### Custom Domain (Optional)
Streamlit Cloud apps get URLs like:
`https://manikyapavan925-news-equity-research-tool-streamlit-app-xyz123.streamlit.app`

### Sharing Your App
1. Share the Streamlit Cloud URL
2. Add to your GitHub README
3. Social media promotion

## 📈 Performance Tips

### Optimization Strategies
1. **Caching**: Cache expensive operations
2. **Session State**: Manage data efficiently
3. **Error Handling**: Graceful failure handling
4. **User Experience**: Progress indicators and feedback

### Resource Limits
Streamlit Cloud (free tier):
- 1 GB RAM
- 2 CPU cores
- Limited to public repositories

## 🎯 Next Steps

After successful deployment:
1. Test all functionality thoroughly
2. Monitor for any issues in the first 24 hours
3. Share with users and gather feedback
4. Plan feature enhancements

## 📞 Support

### Resources
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [GitHub Issues](https://github.com/manikyapavan925/news_equity_research_tool/issues)

### Common Commands
```bash
# Local testing
streamlit run streamlit_app.py

# Check dependencies
pip freeze > requirements.txt

# Test requirements
pip install -r requirements.txt
```

---

**🎉 Congratulations!** Your News Research Assistant should now be live on Streamlit Cloud!
