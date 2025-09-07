import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
from urllib.parse import urlparse
import time

# Streamlit page configuration - optimized for deployment
st.set_page_config(
    page_title="News Research Assistant", 
    page_icon="📰", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/manikyapavan925/news_equity_research_tool',
        'Report a bug': 'https://github.com/manikyapavan925/news_equity_research_tool/issues',
        'About': 'News Research Assistant - A tool for analyzing financial news articles'
    }
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>📰 News Research Assistant</h1>
    <p>Analyze Financial News with AI</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Sidebar for URL input
st.sidebar.header("📝 Add News Articles")
st.sidebar.markdown("Enter up to 5 news article URLs for analysis:")

urls = []
for i in range(5):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}", help=f"Enter news article URL #{i+1}")
    if url.strip():
        urls.append(url.strip())

# Add URL validation
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Filter valid URLs
valid_urls = [url for url in urls if is_valid_url(url)]
if urls and len(valid_urls) != len(urls):
    st.sidebar.warning(f"⚠️ {len(urls) - len(valid_urls)} invalid URL(s) detected and will be skipped.")

# Function to generate article summary
@st.cache_data(ttl=3600)
def generate_article_summary(article, length="Medium"):
    """Generate a summary of the article content"""
    content = article.get('content', '')
    title = article.get('title', 'No Title')
    
    if not content or len(content.strip()) < 100:
        return "⚠️ Article content is too short to generate a meaningful summary."
    
    # Split content into sentences
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return "⚠️ Unable to extract meaningful sentences from the article."
    
    # Determine summary length
    if length == "Short":
        target_sentences = min(3, len(sentences))
    elif length == "Medium":
        target_sentences = min(5, len(sentences))
    else:  # Detailed
        target_sentences = min(8, len(sentences))
    
    # Simple extractive summarization
    # Score sentences based on:
    # 1. Position (first and last sentences often important)
    # 2. Length (medium length sentences preferred)
    # 3. Keyword frequency
    
    # Extract important keywords from title
    title_words = set(re.findall(r'\b\w+\b', title.lower()))
    financial_keywords = {
        'stock', 'market', 'price', 'earnings', 'revenue', 'profit', 'loss', 
        'merger', 'acquisition', 'ipo', 'dividend', 'investment', 'trading',
        'financial', 'economic', 'business', 'company', 'corporation', 'shares',
        'nasdaq', 'nyse', 'dow', 'sp500', 'index', 'fund', 'bond', 'crypto'
    }
    
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0
        sentence_lower = sentence.lower()
        sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
        
        # Position scoring
        if i == 0:  # First sentence
            score += 3
        elif i == len(sentences) - 1:  # Last sentence
            score += 2
        elif i < len(sentences) * 0.3:  # Early sentences
            score += 1
        
        # Length scoring (prefer medium-length sentences)
        word_count = len(sentence.split())
        if 10 <= word_count <= 30:
            score += 2
        elif 5 <= word_count <= 50:
            score += 1
        
        # Keyword scoring
        title_matches = len(title_words.intersection(sentence_words))
        financial_matches = len(financial_keywords.intersection(sentence_words))
        score += title_matches * 2 + financial_matches
        
        # Avoid very short or very long sentences
        if word_count < 5 or word_count > 60:
            score -= 2
            
        scored_sentences.append((sentence, score, i))
    
    # Sort by score and select top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    selected_sentences = scored_sentences[:target_sentences]
    
    # Sort selected sentences by original order
    selected_sentences.sort(key=lambda x: x[2])
    
    # Create summary
    summary_sentences = [s[0] for s in selected_sentences]
    summary = '. '.join(summary_sentences)
    
    # Add some basic cleanup
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    if not summary:
        summary = sentences[0] if sentences else "Unable to generate summary."
    
    return summary

# Enhanced article extraction function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def extract_article_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            element.decompose()
        
        # Try multiple selectors to find main content
        content = ""
        selectors = [
            'article',
            '.article-content', 
            '.post-content', 
            '.entry-content',
            '.content-body',
            '.story-body',
            'main',
            '.content',
            '[role="main"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text()
                break
        
        if not content:
            # Fallback to body content
            body = soup.find('body')
            if body:
                content = body.get_text()
            else:
                content = soup.get_text()
        
        # Clean and process content
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Extract title
        title = ""
        title_element = soup.find('title')
        if title_element:
            title = title_element.get_text().strip()
        
        # Extract meta description
        description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            description = meta_desc.get('content', '').strip()
        
        return {
            'content': content[:5000],  # Limit content length
            'title': title[:200],
            'description': description[:300],
            'word_count': len(content.split()),
            'domain': urlparse(url).netloc
        }
        
    except requests.exceptions.Timeout:
        return {'error': 'Request timeout - article took too long to load'}
    except requests.exceptions.RequestException as e:
        return {'error': f'Network error: {str(e)}'}
    except Exception as e:
        return {'error': f'Error extracting content: {str(e)}'}

# Enhanced sentiment analysis
def analyze_sentiment(text):
    text_lower = text.lower()
    
    positive_words = [
        'good', 'great', 'excellent', 'positive', 'growth', 'profit', 'success', 
        'up', 'rise', 'increase', 'gain', 'strong', 'bullish', 'optimistic',
        'beat', 'outperform', 'surge', 'boost', 'rally', 'upward'
    ]
    
    negative_words = [
        'bad', 'poor', 'negative', 'loss', 'decline', 'down', 'fall', 'decrease', 
        'crisis', 'problem', 'weak', 'bearish', 'pessimistic', 'miss', 'underperform',
        'plunge', 'crash', 'drop', 'concern', 'risk', 'warning'
    ]
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = len(text_lower.split())
    pos_ratio = pos_count / max(total_words, 1)
    neg_ratio = neg_count / max(total_words, 1)
    
    sentiment_score = pos_count - neg_count
    confidence = abs(sentiment_score) / max(pos_count + neg_count, 1)
    
    return {
        'score': sentiment_score,
        'positive_count': pos_count,
        'negative_count': neg_count,
        'confidence': confidence,
        'positive_ratio': pos_ratio,
        'negative_ratio': neg_ratio
    }

# Process URLs button with enhanced features
col1, col2 = st.sidebar.columns(2)
with col1:
    process_button = st.button("📄 Process Articles", disabled=not valid_urls or st.session_state.processing)
with col2:
    clear_button = st.button("🗑️ Clear All", help="Clear all processed articles")

if clear_button:
    st.session_state.articles = []
    st.success("All articles cleared!")
    st.rerun()

if process_button and valid_urls:
    st.session_state.processing = True
    articles_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, url in enumerate(valid_urls):
        status_text.text(f"Processing article {i+1}/{len(valid_urls)}: {url[:50]}...")
        progress_bar.progress((i) / len(valid_urls))
        
        article_data = extract_article_content(url)
        
        if 'error' not in article_data:
            article_data['url'] = url
            article_data['processed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            article_data['sentiment'] = analyze_sentiment(article_data['content'])
            articles_data.append(article_data)
        else:
            st.sidebar.error(f"Failed to process {url}: {article_data['error']}")
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    if articles_data:
        st.session_state.articles = articles_data
        st.success(f"✅ Successfully processed {len(articles_data)} out of {len(valid_urls)} articles!")
    else:
        st.error("❌ No articles could be processed successfully.")
    
    st.session_state.processing = False
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

# Display processing statistics
if valid_urls:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Queue Status:**")
    st.sidebar.markdown(f"• Valid URLs: {len(valid_urls)}")
    st.sidebar.markdown(f"• Processed: {len(st.session_state.articles)}")

# Main content area
if st.session_state.articles:
    # Summary dashboard
    st.header("📊 Analysis Dashboard")
    
    # Metrics
    total_articles = len(st.session_state.articles)
    total_words = sum(article.get('word_count', 0) for article in st.session_state.articles)
    avg_words = total_words // total_articles if total_articles > 0 else 0
    
    # Sentiment overview
    positive_articles = sum(1 for article in st.session_state.articles 
                          if article.get('sentiment', {}).get('score', 0) > 0)
    negative_articles = sum(1 for article in st.session_state.articles 
                          if article.get('sentiment', {}).get('score', 0) < 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Articles", total_articles)
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    with col3:
        st.metric("Avg Words/Article", f"{avg_words:,}")
    with col4:
        st.metric("Positive Sentiment", f"{positive_articles}/{total_articles}")
    
    # Sentiment distribution chart
    if total_articles > 0:
        sentiment_data = pd.DataFrame([
            {'Sentiment': 'Positive', 'Count': positive_articles},
            {'Sentiment': 'Negative', 'Count': negative_articles},
            {'Sentiment': 'Neutral', 'Count': total_articles - positive_articles - negative_articles}
        ])
        st.bar_chart(sentiment_data.set_index('Sentiment'))

    # Individual article analysis
    st.header("� Article Details")
    
    for i, article in enumerate(st.session_state.articles):
        with st.expander(f"📰 Article {i+1}: {article.get('title', 'No Title')[:80]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**🔗 Source:** {article['url']}")
                st.write(f"**🌐 Domain:** {article.get('domain', 'Unknown')}")
                st.write(f"**📅 Processed:** {article.get('processed_at', 'Unknown')}")
                
                if article.get('description'):
                    st.write(f"**📝 Description:** {article['description']}")
                
                st.write("**📄 Content Preview:**")
                st.write(article['content'][:500] + "..." if len(article['content']) > 500 else article['content'])
            
            with col2:
                # Sentiment analysis display
                sentiment = article.get('sentiment', {})
                score = sentiment.get('score', 0)
                confidence = sentiment.get('confidence', 0)
                
                if score > 0:
                    st.success(f"😊 Positive\nScore: +{score}\nConfidence: {confidence:.2%}")
                elif score < 0:
                    st.error(f"😟 Negative\nScore: {score}\nConfidence: {confidence:.2%}")
                else:
                    st.info(f"😐 Neutral\nScore: {score}")
                
                # Word count
                st.metric("Word Count", article.get('word_count', 0))
            
            # Extract potential stock tickers
            ticker_pattern = r'\b[A-Z]{2,5}\b'
            tickers = re.findall(ticker_pattern, article['content'])
            # Filter out common false positives
            exclude_words = {
                'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR',
                'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO',
                'WHO', 'BOY', 'DID', 'WHY', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'CEO', 'CFO', 'CTO',
                'USA', 'EUR', 'USD', 'API', 'CEO', 'IPO', 'ETF', 'SEC', 'LLC', 'INC', 'LTD', 'PLC'
            }
            potential_tickers = [t for t in set(tickers) if t not in exclude_words and len(t) <= 5][:10]
            
            if potential_tickers:
                st.write("**📈 Potential Stock Tickers:**")
                ticker_cols = st.columns(min(len(potential_tickers), 5))
                for idx, ticker in enumerate(potential_tickers[:5]):
                    with ticker_cols[idx]:
                        st.code(ticker)

# Enhanced Q&A Section
st.header("💬 Smart Question Answering")
if st.session_state.articles:
    # Check for quick question from URL params
    query_params = st.query_params
    default_question = query_params.get("question", "")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("❓ Ask a question about the loaded articles:", 
                                value=default_question,
                                placeholder="e.g., What companies were mentioned? What are the main themes?")
    with col2:
        search_type = st.selectbox("Search Type", ["Keyword", "Semantic"], help="Keyword: exact word matching, Semantic: meaning-based")
    
    # Clear query params if question was loaded
    if default_question:
        st.query_params.clear()
    
    if question and st.button("🔍 Search Answer"):
        with st.spinner("Searching for relevant information..."):
            # Debug info
            st.info(f"🔍 Searching {len(st.session_state.articles)} article(s) for: '{question}'")
            
            results = []
            question_lower = question.lower()
            question_words = set(re.findall(r'\b\w+\b', question_lower))
            
            # Remove common stop words for better matching
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'what', 'when', 'where', 'how', 'why', 'who'}
            meaningful_words = question_words - stop_words
            
            st.write(f"🔤 Key search words: {list(meaningful_words)}")
            
            # Handle general questions like "what is this about"
            general_questions = [
                "what is this about", "summarize", "summary", "what is this article about",
                "tell me about this", "what does this say", "main points", "key information"
            ]
            
            is_general_question = any(general in question_lower for general in general_questions)
            
            if is_general_question:
                st.info("🤖 Detected general question - providing article summaries")
                
                for i, article in enumerate(st.session_state.articles):
                    content = article.get('content', '')
                    title = article.get('title', 'No Title')
                    
                    if content and len(content.strip()) > 50:
                        # Generate a quick summary for general questions
                        sentences = re.split(r'[.!?]+', content)
                        good_sentences = [s.strip() for s in sentences if 20 <= len(s.strip()) <= 150]
                        
                        # Take first few sentences as summary
                        summary_sentences = good_sentences[:3] if good_sentences else [content[:200] + "..."]
                        
                        results.append({
                            'article_index': i + 1,
                            'title': title,
                            'url': article['url'],
                            'sentences': summary_sentences,
                            'score': 10,  # High score for general questions
                            'domain': article.get('domain', 'Unknown')
                        })
            
            else:
                # Normal keyword search
                for i, article in enumerate(st.session_state.articles):
                    content = article.get('content', '')
                    title = article.get('title', '')
                    
                    # Show content length for debugging
                    st.write(f"📰 Article {i+1}: {len(content)} characters, Title: {title[:50]}...")
                    
                    if not content or len(content.strip()) < 50:
                        st.warning(f"⚠️ Article {i+1} has very little content ({len(content)} chars)")
                        continue
                    
                    content_lower = content.lower()
                    title_lower = title.lower()
                    
                    # Enhanced relevance scoring
                    if search_type == "Keyword":
                        # Exact word matching
                        content_words = set(re.findall(r'\b\w+\b', content_lower))
                        title_words = set(re.findall(r'\b\w+\b', title_lower))
                        
                        common_content_words = meaningful_words.intersection(content_words)
                        common_title_words = meaningful_words.intersection(title_words)
                        
                        # Title matches get higher weight
                        relevance_score = len(common_content_words) + (len(common_title_words) * 3)
                        
                        # If no exact matches, try partial matching
                        if relevance_score == 0 and meaningful_words:
                            for word in meaningful_words:
                                # Look for partial matches (contains word)
                                if any(word in content_word for content_word in content_words):
                                    relevance_score += 0.5
                                if any(word in title_word for title_word in title_words):
                                    relevance_score += 1.5
                        
                        st.write(f"  🎯 Content matches: {list(common_content_words)}")
                        st.write(f"  📝 Title matches: {list(common_title_words)}")
                        st.write(f"  📊 Score: {relevance_score}")
                        
                    else:
                        # Semantic matching
                        relevance_score = 0
                        for word in meaningful_words:
                            if word in content_lower:
                                relevance_score += content_lower.count(word)
                            if word in title_lower:
                                relevance_score += title_lower.count(word) * 2
                    
                    # Lower the threshold for showing results (include partial matches)
                    if relevance_score >= 0.5:
                        # Find relevant sentences
                        sentences = re.split(r'[.!?]+', content)
                        relevant_sentences = []
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if len(sentence) > 15:
                                sentence_lower = sentence.lower()
                                sentence_score = 0
                                
                                # Check for word matches
                                for word in meaningful_words:
                                    if word in sentence_lower:
                                        sentence_score += 2
                                    # Partial word matching
                                    elif any(word in sentence_word for sentence_word in sentence_lower.split()):
                                        sentence_score += 1
                                        
                                if sentence_score > 0:
                                    relevant_sentences.append((sentence, sentence_score))
                        
                        # Sort sentences by relevance
                        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                        
                        if relevant_sentences:
                            results.append({
                                'article_index': i + 1,
                                'title': title,
                                'url': article['url'],
                                'sentences': [s[0] for s in relevant_sentences[:5]],
                                'score': relevance_score,
                                'domain': article.get('domain', 'Unknown')
                            })
                        elif relevance_score > 0:
                            # If we have matches but no good sentences, show content preview
                            content_preview = content[:400] + "..." if len(content) > 400 else content
                            results.append({
                                'article_index': i + 1,
                                'title': title,
                                'url': article['url'],
                                'sentences': [f"Found relevant content: {content_preview}"],
                                'score': relevance_score,
                                'domain': article.get('domain', 'Unknown')
                            })
            
            # Sort results by relevance
            results.sort(key=lambda x: x['score'], reverse=True)
            
            if results:
                st.success(f"🎯 Found {len(results)} relevant source(s)")
                
                for idx, result in enumerate(results[:5]):  # Show top 5 results
                    with st.expander(f"📄 {result['title'][:60]}... (Relevance: {result['score']})"):
                        st.write(f"**🔗 Source:** {result['url']}")
                        st.write(f"**🌐 Domain:** {result['domain']}")
                        st.write(f"**🎯 Relevant excerpts:**")
                        for sentence in result['sentences']:
                            st.write(f"💡 {sentence}")
            else:
                st.warning("🤔 No relevant information found. Try rephrasing your question or using different keywords.")
    
    # Quick question suggestions
    st.markdown("**💡 Quick Questions:**")
    quick_questions = [
        "What companies were mentioned?",
        "What are the main financial trends?",
        "Are there any merger or acquisition news?",
        "What earnings reports are discussed?",
        "Are there any regulatory changes mentioned?"
    ]
    
    cols = st.columns(len(quick_questions))
    for i, q in enumerate(quick_questions):
        with cols[i]:
            if st.button(q, key=f"quick_q_{i}", help="Click to use this question"):
                st.query_params["question"] = q
                st.rerun()
else:
    st.info("👆 Please load some articles first to start asking questions!")
    
    # Sample articles for demonstration
    st.markdown("**📚 Try these sample financial news URLs:**")
    sample_urls = [
        "https://finance.yahoo.com/news/",
        "https://www.cnbc.com/world/?region=world",
        "https://www.reuters.com/business/finance/"
    ]
    for url in sample_urls:
        st.code(url)

# Article Summarization Section
if st.session_state.articles:
    st.header("📄 Article Summarization")
    
    # Select article to summarize
    article_options = [f"Article {i+1}: {article.get('title', 'No Title')[:60]}..." 
                      for i, article in enumerate(st.session_state.articles)]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_article_idx = st.selectbox(
            "Select an article to summarize:",
            range(len(st.session_state.articles)),
            format_func=lambda x: article_options[x]
        )
    with col2:
        summary_length = st.selectbox("Summary Length", ["Short", "Medium", "Detailed"])
    
    if st.button("📝 Generate Summary", key="summarize_btn"):
        selected_article = st.session_state.articles[selected_article_idx]
        
        with st.spinner("Generating summary..."):
            summary = generate_article_summary(selected_article, summary_length)
            
            st.subheader(f"📄 Summary: {selected_article.get('title', 'No Title')}")
            
            # Article metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📅 Processed", selected_article.get('processed_at', 'Unknown'))
            with col2:
                sentiment = selected_article.get('sentiment', {})
                sentiment_label = "Positive" if sentiment.get('positive_ratio', 0) > sentiment.get('negative_ratio', 0) else "Negative"
                st.metric("😊 Sentiment", sentiment_label, f"{sentiment.get('positive_ratio', 0):.1%}")
            with col3:
                word_count = len(selected_article.get('content', '').split())
                st.metric("📝 Word Count", f"{word_count:,}")
            
            # Display summary
            st.markdown("### 🎯 Key Summary")
            st.info(summary)
            
            # Original article link
            st.markdown(f"**🔗 Original Article:** [{selected_article.get('domain', 'Source')}]({selected_article['url']})")
            
            # Download summary option
            summary_text = f"ARTICLE SUMMARY\n\nTitle: {selected_article.get('title', 'No Title')}\nSource: {selected_article['url']}\nProcessed: {selected_article.get('processed_at', 'Unknown')}\n\nSUMMARY:\n{summary}\n\nGenerated by News Research Assistant"
            
            st.download_button(
                label="💾 Download Summary",
                data=summary_text,
                file_name=f"summary_{selected_article_idx+1}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Quick analysis and export features
st.header("🚀 Quick Analysis Tools")

if st.session_state.articles:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📈 Market Sentiment", help="Analyze overall market sentiment"):
            # Calculate overall sentiment
            total_positive = sum(article.get('sentiment', {}).get('positive_count', 0) for article in st.session_state.articles)
            total_negative = sum(article.get('sentiment', {}).get('negative_count', 0) for article in st.session_state.articles)
            overall_score = total_positive - total_negative
            
            st.write("**📊 Overall Market Sentiment:**")
            if overall_score > 0:
                st.success(f"🟢 Bullish sentiment detected (Score: +{overall_score})")
            elif overall_score < 0:
                st.error(f"🔴 Bearish sentiment detected (Score: {overall_score})")
            else:
                st.info("🟡 Neutral market sentiment")
    
    with col2:
        if st.button("🏢 Extract Companies", help="Find mentioned companies"):
            # Extract company names and tickers
            all_text = " ".join([article['content'] for article in st.session_state.articles])
            
            # Pattern for company names (capitalized words)
            company_pattern = r'\b[A-Z][a-z]+ (?:Inc|Corp|Corporation|Company|Ltd|Limited|Group|Holdings)\b'
            companies = re.findall(company_pattern, all_text)
            
            # Pattern for stock tickers
            ticker_pattern = r'\b[A-Z]{2,5}\b'
            tickers = re.findall(ticker_pattern, all_text)
            
            st.write("**🏢 Companies Mentioned:**")
            if companies:
                unique_companies = list(set(companies))[:10]
                for company in unique_companies:
                    st.write(f"• {company}")
            
            st.write("**📊 Potential Tickers:**")
            if tickers:
                exclude_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'CEO', 'CFO', 'USA', 'EUR', 'USD'}
                unique_tickers = [t for t in set(tickers) if t not in exclude_words][:15]
                ticker_text = " | ".join(unique_tickers)
                st.code(ticker_text)
    
    with col3:
        if st.button("📊 Export Summary", help="Generate summary report"):
            # Create summary report
            report_data = {
                'Total Articles': len(st.session_state.articles),
                'Total Words': sum(article.get('word_count', 0) for article in st.session_state.articles),
                'Positive Articles': sum(1 for article in st.session_state.articles if article.get('sentiment', {}).get('score', 0) > 0),
                'Negative Articles': sum(1 for article in st.session_state.articles if article.get('sentiment', {}).get('score', 0) < 0),
                'Sources': len(set(article.get('domain', '') for article in st.session_state.articles))
            }
            
            st.write("**📋 Analysis Summary:**")
            for key, value in report_data.items():
                st.metric(key, value)
    
    with col4:
        if st.button("🔄 Refresh Analysis", help="Re-analyze all articles"):
            for article in st.session_state.articles:
                article['sentiment'] = analyze_sentiment(article['content'])
            st.success("✅ Analysis refreshed!")
            st.rerun()

else:
    st.info("📝 Load some articles first to access analysis tools!")

# Footer with enhanced information
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🎯 News Research Assistant - Streamlit Cloud Edition</h3>
    <p><strong>Features Available:</strong></p>
    <ul style="text-align: left; display: inline-block;">
        <li>✅ Real-time news article processing</li>
        <li>✅ Advanced sentiment analysis</li>
        <li>✅ Smart question answering</li>
        <li>✅ Company and ticker extraction</li>
        <li>✅ Interactive dashboards</li>
        <li>✅ Multi-source aggregation</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Add some helpful tips in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips for Better Results")
st.sidebar.markdown("""
- **Use financial news URLs** for best results
- **Wait for processing** to complete before asking questions
- **Try specific questions** like "What companies reported earnings?"
- **Use both keyword and semantic search** for different insights
- **Check multiple sources** for comprehensive analysis
""")

st.sidebar.markdown("### 🔧 Troubleshooting")
st.sidebar.markdown("""
- **Slow loading?** Some sites block automated requests
- **No content?** Check if URL is accessible and contains text
- **Empty results?** Try rephrasing your questions
- **Need help?** Check our GitHub repository
""")
