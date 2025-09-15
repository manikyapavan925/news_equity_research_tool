# 🌐 Tavily Web Search Integration Setup

This guide explains how to set up and configure Tavily web search for enhanced, real-time financial data retrieval.

## 🎯 Overview

The News Equity Research Tool now includes intelligent Tavily web search integration with the following features:

- **Smart Fallback System**: LLM response quality is automatically evaluated
- **Real-time Data**: Tavily provides the most current financial information
- **Enhanced Accuracy**: Searches focus on reputable financial sources
- **Seamless Integration**: Falls back to Tavily only when needed

## 🔧 Setup Instructions

### 1. Get Tavily API Key

1. Visit [https://tavily.com](https://tavily.com)
2. Sign up for a free account
3. Navigate to your dashboard
4. Copy your API key

### 2. Configure Environment Variables

Create a `.env` file in your project root:

```bash
# Tavily Web Search API Configuration
TAVILY_API_KEY=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  # ✅ CONFIGURED

# Optional: Other API keys for enhanced functionality
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
```

**✅ STATUS: API KEY SUCCESSFULLY CONFIGURED AND TESTED**

**Important:** Never commit your `.env` file to version control!

### 3. Test the Integration

Run this test to verify your setup:

```python
from app.ai_original import generate_realtime_ai_answer

# Test with a financial question
question = "What is the current Tesla stock target price for 2025?"
response, used_web_search = generate_realtime_ai_answer(
    question, 
    articles=[], 
    enable_web_search=True
)

print(f"Used Tavily: {used_web_search}")
print(response)
```

## 📊 How It Works

### 1. Initial Response Generation
The system first generates a response using available context and basic LLM capabilities.

### 2. Quality Assessment
The response is automatically evaluated for:
- **Length and Detail**: Adequate information depth
- **Financial Terminology**: Relevant financial terms and metrics
- **Recency Indicators**: Current/recent data references  
- **Numerical Data**: Specific figures and statistics
- **Generic Content**: Avoids template responses

### 3. Intelligent Fallback
If the quality score is below 50/100 or has 3+ issues, Tavily search activates:
- Enhanced search queries for financial focus
- Targets reputable financial news sources
- Returns structured, detailed results
- Provides source attribution and timestamps

## 🎯 Targeted Sources

Tavily searches prioritize these financial sources:
- Reuters (reuters.com)
- Bloomberg (bloomberg.com) 
- MoneyControl (moneycontrol.com)
- Economic Times (economictimes.com)
- Business Standard (business-standard.com)
- LiveMint (livemint.com)
- CNBC (cnbc.com)
- Yahoo Finance (yahoo.com)

## 📋 Example Usage

### Basic Question (May use LLM only)
```
Q: "What is a stock market?"
A: Uses basic LLM response (educational/general)
```

### Financial Query (Likely to trigger Tavily)
```
Q: "What is HDFC Bank's target price for 2025?"
A: Triggers Tavily search for latest analyst reports
```

### Recent Event Query (Definitely triggers Tavily)  
```
Q: "Latest quarterly results impact on TCS share price"
A: Uses Tavily for most current financial news
```

## ⚙️ Configuration Options

### Quality Threshold Adjustment

To modify when Tavily fallback triggers, adjust these parameters in `ai_original.py`:

```python
# In evaluate_response_quality function
needs_fallback = quality_score < 50 or len(issues) >= 3

# Lower threshold = more Tavily usage
needs_fallback = quality_score < 70 or len(issues) >= 2

# Higher threshold = less Tavily usage  
needs_fallback = quality_score < 30 or len(issues) >= 4
```

### Search Result Count

Modify maximum Tavily results:

```python
# In search_with_tavily function
tavily_results = search_with_tavily(question, max_results=5)  # Default

# More results for comprehensive analysis
tavily_results = search_with_tavily(question, max_results=10)

# Fewer results for faster responses
tavily_results = search_with_tavily(question, max_results=3)
```

## 🚀 Benefits

1. **Accuracy**: Real-time data from trusted financial sources
2. **Relevance**: Context-aware search queries  
3. **Efficiency**: Only activates when needed
4. **Transparency**: Clear indication when web search is used
5. **Attribution**: Proper source links and timestamps

## 🔍 Troubleshooting

### No Tavily Results
- Check API key configuration
- Verify internet connection
- Review API usage limits

### Poor Quality Assessment
- Check if financial terms are being detected
- Verify recency indicators in responses
- Adjust quality thresholds if needed

### Unexpected Fallback Behavior
- Review quality assessment logs
- Check response content for generic phrases
- Adjust evaluation criteria as needed

## 📈 Performance Tips

1. **API Key Management**: Keep your API key secure and monitor usage
2. **Caching**: Consider implementing result caching for repeated queries
3. **Rate Limiting**: Be mindful of API rate limits during heavy usage
4. **Source Selection**: Customize target domains based on your needs

## 🔒 Security

- Store API keys in environment variables only
- Never hardcode API keys in source code  
- Add `.env` to your `.gitignore` file
- Use separate API keys for development/production

---

**Need Help?** Check the Tavily documentation at [https://docs.tavily.com](https://docs.tavily.com)
