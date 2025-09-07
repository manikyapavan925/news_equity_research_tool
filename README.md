# 📰 News Research Assistant - Streamlit Cloud Edition

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A powerful financial news analysis tool built for Streamlit Cloud that helps you analyze multiple news articles, extract insights, and perform intelligent Q&A on financial content.

## 🚀 Live Demo

**Streamlit Cloud Deployment:** [Launch App](https://your-app-url.streamlit.app)

## ✨ Features

### 📊 Core Analytics
- **Multi-Article Processing**: Analyze up to 5 news articles simultaneously
- **Advanced Sentiment Analysis**: AI-powered sentiment scoring with confidence metrics
- **Smart Question Answering**: Both keyword and semantic search capabilities
- **Company & Ticker Extraction**: Automatically identify companies and stock symbols
- **Real-time Dashboard**: Interactive metrics and visualizations

### 🎯 Streamlit Cloud Optimizations
- **Caching**: Smart caching for improved performance
- **Error Handling**: Robust error handling for network requests
- **Progress Tracking**: Real-time processing indicators
- **Responsive Design**: Mobile-friendly interface
- **Performance Monitoring**: Optimized for cloud deployment

## 🔧 Quick Start

### Option 1: Use Streamlit Cloud (Recommended)
1. Visit the [live app](https://your-app-url.streamlit.app)
2. Add news article URLs in the sidebar
3. Click "Process Articles" 
4. Start analyzing and asking questions!

### Option 2: Local Development
```bash
# Clone the repository
git clone https://github.com/manikyapavan925/news_equity_research_tool.git
cd news_equity_research_tool

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

## 📱 How to Use

### 1. **Add Articles**
- Enter up to 5 financial news URLs in the sidebar
- URLs are validated automatically
- Click "Process Articles" to analyze

### 2. **Analyze Results**
- View sentiment analysis for each article
- Check the interactive dashboard
- Explore extracted companies and tickers

### 3. **Ask Questions**
- Use the Q&A section to query your articles
- Try both keyword and semantic search
- Get ranked results with relevance scores

### 4. **Quick Analysis**
- Market sentiment overview
- Company extraction
- Export summary reports

## 📈 Example Questions

- "What companies were mentioned in earnings reports?"
- "What are the main financial trends discussed?"
- "Are there any merger and acquisition announcements?"
- "What regulatory changes are mentioned?"
- "Which stocks showed positive sentiment?"

## 🛠️ Technology Stack

### Frontend & UI
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **Custom CSS** - Enhanced styling and responsiveness

### Data Processing
- **BeautifulSoup4** - Web scraping and HTML parsing
- **Requests** - HTTP library for fetching articles
- **Regular Expressions** - Text pattern matching and extraction

### Cloud Deployment
- **Streamlit Cloud** - Hosting and deployment platform
- **GitHub Integration** - Continuous deployment
- **Caching Strategy** - Performance optimization

## 🔧 Configuration

### Streamlit Cloud Settings
The app is pre-configured for Streamlit Cloud with:
- Optimized `requirements.txt`
- Custom `.streamlit/config.toml`
- System packages in `packages.txt`
- Performance caching enabled

### Environment Variables
No API keys required! The app works out-of-the-box on Streamlit Cloud.

## 📊 Performance Features

- **Caching**: Article content cached for 1 hour
- **Async Processing**: Non-blocking article extraction
- **Error Recovery**: Graceful handling of failed requests
- **Memory Optimization**: Efficient data storage
- **Rate Limiting**: Respectful web scraping practices

## 🐛 Troubleshooting

### Common Issues
- **Slow Loading**: Some websites may block automated requests
- **Empty Content**: Ensure URLs are publicly accessible
- **No Results**: Try rephrasing questions or using different keywords

### Performance Tips
- Use specific financial news sources (Yahoo Finance, CNBC, Reuters)
- Wait for processing to complete before asking questions
- Clear articles periodically to free up memory

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

### Development Setup
```bash
# Fork the repository
git clone https://github.com/yourusername/news_equity_research_tool.git

# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and test
streamlit run streamlit_app.py

# Submit a pull request
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Financial news sources for providing accessible content
- Streamlit team for the amazing framework
- Open source community for various libraries used

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/manikyapavan925/news_equity_research_tool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/manikyapavan925/news_equity_research_tool/discussions)
- **Email**: your-email@example.com

---

**Made with ❤️ for the financial community**
