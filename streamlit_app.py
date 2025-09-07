import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urlparse

# Simple news research tool without heavy ML dependencies
st.set_page_config(
    page_title="News Research Assistant", 
    page_icon="📰", 
    layout="wide"
)

st.title("📰 News Research Assistant (Lite)")
st.subheader("Simple Financial News Analysis Tool")

# Sidebar for URL input
st.sidebar.header("📝 Add News Articles")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url:
        urls.append(url)

# Function to extract article content
def extract_article_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Try to find main content
        content = ""
        for selector in ['article', '.article-content', '.post-content', 'main', '.content']:
            element = soup.select_one(selector)
            if element:
                content = element.get_text()
                break
        
        if not content:
            content = soup.get_text()
        
        # Clean content
        content = re.sub(r'\s+', ' ', content).strip()
        return content[:2000]  # Limit content length
        
    except Exception as e:
        return f"Error extracting content: {str(e)}"

# Process URLs button
if st.sidebar.button("📄 Process Articles") and urls:
    articles_data = []
    
    with st.spinner("Processing articles..."):
        for url in urls:
            st.write(f"Processing: {url}")
            content = extract_article_content(url)
            articles_data.append({
                'url': url,
                'content': content
            })
    
    st.session_state.articles = articles_data
    st.success(f"Processed {len(articles_data)} articles!")

# Display articles
if 'articles' in st.session_state:
    st.header("📊 Article Analysis")
    
    for i, article in enumerate(st.session_state.articles):
        with st.expander(f"Article {i+1}: {article['url'][:50]}..."):
            st.write("**Content Preview:**")
            st.write(article['content'][:500] + "...")
            
            # Simple analysis
            content = article['content'].lower()
            
            # Sentiment indicators
            positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'profit', 'success', 'up', 'rise', 'increase']
            negative_words = ['bad', 'poor', 'negative', 'loss', 'decline', 'down', 'fall', 'decrease', 'crisis', 'problem']
            
            pos_count = sum(1 for word in positive_words if word in content)
            neg_count = sum(1 for word in negative_words if word in content)
            
            st.write("**Simple Sentiment Analysis:**")
            if pos_count > neg_count:
                st.success(f"😊 Positive sentiment detected (Score: +{pos_count - neg_count})")
            elif neg_count > pos_count:
                st.error(f"😟 Negative sentiment detected (Score: -{neg_count - pos_count})")
            else:
                st.info("😐 Neutral sentiment detected")
            
            # Extract companies/tickers
            ticker_pattern = r'\b[A-Z]{1,5}\b'
            tickers = re.findall(ticker_pattern, article['content'])
            common_tickers = [t for t in tickers if len(t) <= 4 and t not in ['THE', 'AND', 'FOR', 'ARE', 'BUT']]
            
            if common_tickers:
                st.write("**Potential Stock Tickers:**")
                st.write(", ".join(set(common_tickers[:10])))

# Q&A Section
st.header("💬 Ask Questions")
if 'articles' in st.session_state:
    question = st.text_input("Ask a question about the loaded articles:")
    
    if question and st.button("🔍 Search"):
        # Simple keyword search across articles
        results = []
        question_lower = question.lower()
        
        for i, article in enumerate(st.session_state.articles):
            content_lower = article['content'].lower()
            
            # Simple relevance scoring
            question_words = question_lower.split()
            relevance_score = sum(1 for word in question_words if word in content_lower)
            
            if relevance_score > 0:
                # Find sentences containing question keywords
                sentences = article['content'].split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    if any(word in sentence.lower() for word in question_words):
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    results.append({
                        'article_index': i + 1,
                        'url': article['url'],
                        'sentences': relevant_sentences[:3],  # Top 3 relevant sentences
                        'score': relevance_score
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if results:
            st.success(f"Found {len(results)} relevant article(s)")
            
            for result in results:
                with st.expander(f"Article {result['article_index']} (Relevance: {result['score']})"):
                    st.write(f"**Source:** {result['url']}")
                    st.write("**Relevant excerpts:**")
                    for sentence in result['sentences']:
                        st.write(f"• {sentence}")
        else:
            st.warning("No relevant information found. Try different keywords.")
else:
    st.info("👆 Please load some articles first to ask questions!")

# Quick analysis buttons
st.header("🚀 Quick Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 Market Trends"):
        if 'articles' in st.session_state:
            trend_words = ['market', 'stock', 'price', 'trading', 'investment']
            # Simple trend analysis logic here
            st.info("Analyzing market trends... (Feature coming soon!)")
        else:
            st.warning("Load articles first!")

with col2:
    if st.button("🏢 Company Analysis"):
        if 'articles' in st.session_state:
            # Simple company analysis logic here
            st.info("Analyzing companies... (Feature coming soon!)")
        else:
            st.warning("Load articles first!")

with col3:
    if st.button("📊 Summary Report"):
        if 'articles' in st.session_state:
            st.write("**Quick Summary:**")
            st.write(f"• Total articles processed: {len(st.session_state.articles)}")
            total_words = sum(len(article['content'].split()) for article in st.session_state.articles)
            st.write(f"• Total words analyzed: {total_words:,}")
            st.write(f"• Average article length: {total_words // len(st.session_state.articles):,} words")
        else:
            st.warning("Load articles first!")

# Footer
st.markdown("---")
st.markdown("""
**News Research Assistant (Lite Version)**  
This is a simplified version of the full AI-powered tool. For advanced features including:
- AI-powered sentiment analysis
- Vector-based similarity search
- Advanced company research
- LLM-powered Q&A

Please run the full version locally or deploy on platforms with ML support.
""")

st.markdown("**GitHub:** [View Full Source Code](https://github.com/manikyapavan925/news_equity_research_tool)")
