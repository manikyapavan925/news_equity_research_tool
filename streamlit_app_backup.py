import streamlit as st
import requests
from bs4 import BeautifulSoup
from app.ai_original import generate_realtime_ai_answer
from app.utils import clean_text_content
import re
import os
import pandas as pd
from urllib.parse import urlparse, quote
import time
import random
import json
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global variable for summarizer
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
    return summarizer

# Streamlit page configuration - optimized for deployment
st.set_page_config(
    page_title="News Research Assistant", 
    page_icon="📰", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
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

# Add website compatibility info
with st.sidebar.expander("📋 Website Compatibility", expanded=False):
    st.markdown("""
    **✅ Usually Work Well:**
    - Reuters, BBC, Guardian
    - Financial Times, Bloomberg
    - Yahoo Finance, CNN
    - TechCrunch, Ars Technica

    **❌ Often Blocked (403 Error):**
    - Invezz.com, Investing.com
    - MarketWatch, some Seeking Alpha

    **💡 If URL fails:**
    - Use AI-Powered search instead
    - Try alternative news sources
    - Copy-paste content manually
    """)

# Create tabs for better organization
tab1, tab2 = st.tabs(["🔗 Add URL", "⚙️ Process & Analyze"])

with tab1:
    st.header("🔗 Add News Article URL")

    # Single URL input instead of 5
    url = st.text_input("Enter news article URL:", key="single_url",
                       placeholder="https://www.example.com/news-article",
                       help="Enter a single news article URL for analysis")

    # URL validation
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    # Show URL validation status
    if url.strip():
        if is_valid_url(url.strip()):
            st.success("✅ Valid URL format")
        else:
            st.error("❌ Invalid URL format")

    # Add URL to session state when valid
    if url.strip() and is_valid_url(url.strip()):
        if 'current_url' not in st.session_state:
            st.session_state.current_url = ""
        st.session_state.current_url = url.strip()

        st.info("💡 URL saved! Switch to 'Process & Analyze' tab to process this article.")

# Function to load LLM for AI-powered answering
@st.cache_resource
def load_simple_llm():
    """Load a lightweight LLM for question answering"""
    try:
        # Use a simple text generation approach for better reliability on Streamlit Cloud
        return "simple_qa"  # Placeholder for now, will implement actual LLM loading
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

# Enhanced LLM Integration
def load_advanced_llm():
    """Load a more capable LLM for question answering"""
    try:
        # Try to use a more capable model
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        # Try Google's Flan-T5 first (better for Q&A)
        try:
            model_name = "google/flan-t5-base"  # Using base instead of large for better compatibility
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            qa_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=400,
                temperature=0.7,
                do_sample=True
            )
            
            return qa_pipeline, "google/flan-t5-base"
            
        except Exception as e:
            print(f"Failed to load Flan-T5: {e}")
            
            # Fallback to DistilBERT for Q&A
            try:
                model_name = "distilbert-base-cased-distilled-squad"
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    tokenizer=model_name
                )
                return qa_pipeline, "distilbert-base-cased-distilled-squad"
                
            except Exception as e:
                print(f"Failed to load DistilBERT: {e}")
                
                # Final fallback to T5-small
                model_name = "t5-small"
                qa_pipeline = pipeline(
                    "text2text-generation",
                    model=model_name,
                    max_length=300
                )
                return qa_pipeline, "t5-small"
                
    except Exception as e:
        print(f"All LLM loading failed: {e}")
        return None, None

# Initialize the advanced LLM
@st.cache_resource
def get_advanced_llm():
    """Get cached advanced LLM"""
    return load_advanced_llm()

def analyze_question_type(question):
    """Analyze the question to understand what type of information is being requested"""
    question_lower = question.lower()
    
    # Question type patterns
    question_types = {
        'plans': ['plan', 'plans', 'strategy', 'roadmap', 'future', 'initiative', 'upcoming'],
        'financial': ['price', 'target price', 'revenue', 'profit', 'earnings', 'financial', 'valuation'],
        'ai_tech': ['ai', 'artificial intelligence', 'technology', 'innovation', 'digital'],
        'performance': ['performance', 'growth', 'decline', 'increase', 'decrease', 'change'],
        'investment': ['investment', 'invest', 'funding', 'capital', 'spending'],
        'market': ['market', 'competition', 'industry', 'sector'],
        'general': ['what', 'why', 'how', 'when', 'where']
    }
    
    detected_types = []
    for q_type, keywords in question_types.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_types.append(q_type)
    
    return detected_types if detected_types else ['general']

def create_smart_prompt(question, context, question_types):
    """Create an intelligent prompt based on question type and context relevance"""
    
    # Check if context is relevant to the question
    question_keywords = set(question.lower().split())
    context_words = set(context.lower().split())
    
    # Remove common stop words for better matching
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    meaningful_q_words = question_keywords - stop_words
    
    # Check for keyword overlap
    keyword_overlap = len(meaningful_q_words.intersection(context_words)) / max(len(meaningful_q_words), 1)
    
    # For AI/tech questions with low relevance, provide structured response
    if 'ai_tech' in question_types or 'plans' in question_types:
        if keyword_overlap < 0.3:  # Low relevance
            return f"""Analyze this article and answer the user's question. Be specific about what information is available.

User Question: {question}

Article Content: {context}

Instructions: If the article doesn't discuss AI strategies, technology plans, or future initiatives, start your answer with "This article does not contain information about [the topic]" and then explain what the article actually discusses.

Answer:"""
        else:
            return f"Based on this technology article, answer: {question}\n\nArticle: {context}\n\nDetailed answer:"
    
    elif 'financial' in question_types:
        return f"Analyze this financial article and answer: {question}\n\nArticle: {context}\n\nFinancial analysis:"
    
    elif keyword_overlap > 0.4:  # Good relevance
        return f"Answer this question based on the article: {question}\n\nArticle: {context}\n\nAnswer:"
    
    else:  # Low relevance
        return f"""Answer this question based on the article. If the information isn't available, explain what the article actually covers.

Question: {question}

Article: {context}

Response:"""

def generate_llm_response(question, context, model_pipeline, model_name):
    """Generate response using the loaded LLM with intelligent prompting"""
    try:
        if not model_pipeline:
            return "⚠️ LLM not available. Using fallback analysis."
        
        # Prepare context (limit to avoid token limits)
        clean_context = clean_text_content(context)
        if len(clean_context) > 2200:  # Leave room for prompt
            clean_context = clean_context[:2200] + "..."
        
        # Analyze question type for smarter prompting
        question_types = analyze_question_type(question)
        
        if "flan-t5" in model_name.lower() or "t5" in model_name.lower():
            # Create intelligent prompt based on question analysis
            prompt = create_smart_prompt(question, clean_context, question_types)
            
            # Add stopping criteria and better generation parameters
            response = model_pipeline(
                prompt, 
                max_length=300,  # Reduced to prevent repetition
                num_return_sequences=1, 
                do_sample=True, 
                temperature=0.1,  # Lower temperature for more focused responses
                top_p=0.9,  # Nucleus sampling to avoid repetition
                repetition_penalty=1.5,  # Penalty for repeating text
                early_stopping=True,
                pad_token_id=model_pipeline.tokenizer.eos_token_id
            )
            answer = response[0]['generated_text'].strip()
            
            # Check for repetitive content and fix it
            if is_repetitive_response(answer):
                # If response is repetitive, provide a clean fallback
                main_topic = extract_main_topic(clean_context)
                if 'ai_tech' in question_types or 'plans' in question_types:
                    clean_answer = f"""**📋 Analysis for: {question}**

**Current Article:** This article discusses {main_topic} but does not contain information about AI strategies or technology plans for 2026.

**💡 For Microsoft AI strategies 2026, check:**
- Microsoft's official AI announcements and roadmaps
- Annual investor presentations and earnings calls  
- Microsoft Build conference presentations
- Azure AI service updates and documentation

**🔗 Context:** The current article focuses on {main_topic} rather than future AI initiatives."""
                else:
                    clean_answer = f"""**📋 Analysis:** The article about {main_topic} doesn't contain specific information to answer your question about '{question}'.

**💡 Suggestion:** Look for sources that specifically cover this topic for detailed information."""
                
                return clean_answer
            
            # Enhanced checks for poor responses
            if (len(answer) < 30 or 
                answer.lower() in ['data', 'jobs data', 'stock', 'microsoft', 'ai', 'not available', 'tariff concerns', 'weak', 'decline'] or
                answer.strip() == clean_context.strip()[:len(answer)] or  # Check if just repeating input
                len(set(answer.split())) < 5):  # Check for very limited vocabulary
                
                print(f"Poor LLM response detected: '{answer[:50]}...', providing intelligent fallback...")
                # Provide intelligent fallback based on question type
                main_topic = extract_main_topic(clean_context)
                
                if 'ai_tech' in question_types or 'plans' in question_types:
                    enhanced_answer = f"""**📋 Expert Analysis for: {question}**

**Article Assessment:** This article primarily discusses {main_topic} and does not contain information about Microsoft's AI strategies or technology plans for 2026.

**🔍 What the article covers instead:**
{clean_context[:200]}...

**💡 To find Microsoft's AI strategies for 2026:**
- **Official Sources:** Microsoft.com investor relations and AI announcements
- **Technology Events:** Microsoft Build, Ignite conferences
- **Strategic Documents:** Annual reports and strategic planning documents
- **Industry Analysis:** Technology publications covering Microsoft's AI roadmap
- **Research Reports:** Investment analyst reports on Microsoft's AI initiatives

**🎯 Why this information isn't in stock articles:**
Stock-focused articles typically discuss market performance and financial factors rather than detailed technology strategies and future planning."""
                    return enhanced_answer
                    
                elif 'financial' in question_types:
                    enhanced_answer = f"""**📋 Financial Analysis for: {question}**

**Article Content:** This article discusses {main_topic} but doesn't provide the specific financial information requested.

**💡 For detailed financial analysis, check:**
- Financial analyst reports and price targets
- Earnings call transcripts and investor presentations
- SEC filings and quarterly reports
- Financial news services and investment research platforms"""
                    return enhanced_answer
                
                else:
                    enhanced_answer = f"""**Analysis Result for: {question}**

**Current Article Content:** This article focuses on {main_topic}.

**Analysis:** The article doesn't contain specific information to directly answer your question about '{question}'.

**Recommendation:** For comprehensive information about this topic, consider sources that specifically address this subject area.

**Current Article Summary:** {clean_context[:150]}..."""
                    return enhanced_answer
                
                if 'ai_tech' in question_types or 'plans' in question_types:
                    enhanced_answer = f"""**Analysis for: {question}**

**Current Article Focus:** This article primarily discusses {main_topic}.

**Why this question can't be answered from this article:**
The article doesn't contain information about AI strategies, technology plans, or future initiatives.

**Where to find this information:**
- Microsoft's investor relations website and quarterly earnings calls
- Technology conferences like Microsoft Build or Ignite
- Annual reports (10-K filings) and strategic announcements
- Tech industry publications covering Microsoft's AI initiatives

**Related context from article:**
Based on the current article's focus on {main_topic}, you might also be interested in how market conditions affect technology investments and strategic planning."""
                else:
                    enhanced_answer = f"""**Analysis for: {question}**

**Current Article Content:** {main_topic}

**Analysis:** The provided article doesn't contain specific information to directly answer your question about '{question}'.

**Recommendation:** For comprehensive information about your question, consider sources that specifically cover this topic area.

**Context:** The article provides insights into {main_topic}, which may be relevant background information."""
                
                return enhanced_answer
            
            # Check if answer indicates unavailability but enhance it
            if any(phrase in answer.lower() for phrase in ['does not contain', 'not discuss', 'does not provide', 'not mention', 'no information']):
                return f"**Expert Analysis:** {answer}"
            
            return f"**Answer:** {answer}"
            
        elif "distilbert" in model_name.lower() or "squad" in model_name.lower():
            # For BERT-style Q&A models
            response = model_pipeline(question=question, context=clean_context)
            answer = response['answer']
            confidence = response['score']
            
            if confidence > 0.4:  # Higher confidence threshold
                return f"**Answer:** {answer} (Confidence: {confidence:.2f})"
            else:
                main_topic = extract_main_topic(clean_context)
                return f"**Analysis Result:** The article doesn't contain sufficient information to answer '{question}' with high confidence. The content primarily focuses on {main_topic}."
            
        else:
            # Generic text generation with improved prompt
            prompt = create_smart_prompt(question, clean_context, question_types)
            response = model_pipeline(prompt, max_length=350)
            answer = response[0]['generated_text'].split("Answer:")[-1].strip()
            return f"**Answer:** {answer}" if answer else "**Analysis Result:** The provided article doesn't contain information relevant to your question."
            
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return f"⚠️ Error generating response: {str(e)}"

def extract_main_topic(content):
    """Extract the main topic from content for better error messages"""
    content_lower = content.lower()
    if 'stock' in content_lower and 'decline' in content_lower:
        return "stock market performance and decline factors"
    elif 'financial' in content_lower or 'earnings' in content_lower:
        return "financial performance and earnings"
    elif 'investment' in content_lower:
        return "investment analysis"
    elif 'market' in content_lower:
        return "market conditions and trends"
    else:
        # Get first few meaningful words
        words = content.split()[:10]
        return f"topics including {' '.join(words[:5])}..."

def is_repetitive_response(text):
    """Detect if the response is repetitive or stuck in a loop"""
    if not text or len(text) < 50:
        return False
    
    # Check for exact phrase repetition
    sentences = text.split('.')
    if len(sentences) > 3:
        # Check if the same sentence appears multiple times
        sentence_counts = {}
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only check meaningful sentences
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
                if sentence_counts[sentence] > 2:  # Same sentence repeated 3+ times
                    return True
    
    # Check for word repetition patterns
    words = text.split()
    if len(words) > 20:
        # Check if we have excessive repetition of the same phrases
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, it's likely repetitive
        max_word_count = max(word_counts.values()) if word_counts else 0
        if max_word_count > len(words) * 0.3:
            return True
    
    # Check for specific repetitive patterns
    repetitive_patterns = [
        "The purpose of the article is to provide",
        "The purpose of this article is to",
        "information about AI plans, technology strategies"
    ]
    
    for pattern in repetitive_patterns:
        if text.count(pattern) > 2:
            return True
    
    return False

def is_inadequate_response(response_text, question):
    """Detect if the LLM response is too generic or inadequate"""
    
    if not response_text or len(response_text.strip()) < 50:
        return True
    
    response_lower = response_text.lower()
    question_lower = question.lower()
    
    # Patterns indicating generic/inadequate responses
    generic_patterns = [
        r'is a.*company.*based in',  # "X is a company based in Y"
        r'is a.*manufacturer.*based',  # "X is a manufacturer based in Y"  
        r'is.*indian.*company',      # "X is an Indian company"
        r'is.*automotive.*company',  # "X is an automotive company"
        r'is known for.*manufacturing',  # "X is known for manufacturing"
        r'was founded in.*year',     # Generic founding information
        r'has been.*since',          # Generic historical statements
        r'produces.*vehicles',       # Generic production statements
    ]
    
    # Check for overly generic responses
    is_generic = any(re.search(pattern, response_lower) for pattern in generic_patterns)
    
    # Check if response doesn't address the specific question
    question_keywords = set(re.findall(r'\b\w+\b', question_lower))
    question_keywords.discard('what')
    question_keywords.discard('are')
    question_keywords.discard('the')
    question_keywords.discard('is')
    
    # Remove common stop words
    stop_words = {'what', 'are', 'the', 'is', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    meaningful_question_words = question_keywords - stop_words
    
    # Check how many question keywords appear in the response
    response_words = set(re.findall(r'\b\w+\b', response_lower))
    keyword_overlap = len(meaningful_question_words.intersection(response_words))
    keyword_coverage = keyword_overlap / max(len(meaningful_question_words), 1)
    
    # Response is inadequate if:
    # 1. It's generic, OR
    # 2. It has very low keyword coverage (< 30%), OR
    # 3. It's too short for a detailed question
    return (is_generic or 
            keyword_coverage < 0.3 or 
            (len(question.split()) > 5 and len(response_text.split()) < 20))

def fetch_article_content(url, max_length=2000):
    """Fetch and extract main text content from a web article, filtering out dashboard and unrelated info"""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script, style, nav, header, footer, aside
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        # Try to find main content areas (add more selectors for financial news)
        content_selectors = [
            'article', '.article-content', '.post-content', '.entry-content',
            '.content', '.main-content', 'main', '.article-body', '.post-body',
            '.contentSection', '.story-content', '.main-area', '.content-wrapper', '.news-content', '.news-body'
        ]
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = elements[0].get_text(separator=' ', strip=True)
                break
        # If no specific content area found, get body text
        if not content:
            content = soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text(separator=' ', strip=True)
        # Clean up the text
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        # Remove lines with dashboard, login, register, watchlist, etc.
        ignore_keywords = ['dashboard', 'login', 'register', 'watchlist', 'trade', 'follow', 'add to', 'vol', 'risk', 'mint', 'bse', 'nse']
        filtered_lines = [line for line in lines if not any(k in line.lower() for k in ignore_keywords) and len(line) > 30]
        content = ' '.join(filtered_lines)
        # Robust fallback: If content is still too short, extract all <p>, <h1>-<h6>, <div> tags
        if len(content.strip()) < 50:
            tag_texts = []
            for tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']:
                for element in soup.find_all(tag):
                    txt = element.get_text(separator=' ', strip=True)
                    if txt and len(txt) > 20:
                        tag_texts.append(txt)
            # Remove duplicates and very short lines
            tag_texts = list(dict.fromkeys([t for t in tag_texts if len(t) > 20]))
            content = ' '.join(tag_texts)
        # Final fallback: If still too short, use entire page text
        if len(content.strip()) < 50:
            page_text = soup.get_text(separator=' ', strip=True)
            page_lines = [line.strip() for line in page_text.split('\n') if len(line.strip()) > 20]
            page_lines = list(dict.fromkeys(page_lines))
            content = ' '.join(page_lines)
        # Limit content length
        if len(content) > max_length:
            content = content[:max_length] + "..."
        # Get title
        title = soup.title.get_text(strip=True) if soup.title else 'Extracted Article'
        return {
            'url': url,
            'title': title,
            'content': content,
            'success': True,
            'domain': url.split('/')[2] if '://' in url else 'unknown'
        }
    except Exception as e:
        return {
            'url': url,
            'title': 'Failed to Load',
            'content': f"Unable to fetch content from this URL: {str(e)}",
            'success': False,
            'domain': 'unknown'
        }

def clean_article_content(text):
    """Remove broken lines, excessive punctuation, and non-informative fragments"""
    # Remove repeated dots, dashes, quotes, and isolated punctuation
    text = re.sub(r'(\.{2,}|-{2,}|"{2,}|\'{2,}|\s{2,})', ' ', text)
    # Remove lines with mostly punctuation or very short
    lines = text.splitlines()
    clean_lines = [line for line in lines if len(re.sub(r'[^A-Za-z0-9]', '', line)) > 10]
    text = ' '.join(clean_lines)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def filter_sentences(sentences):
    """Filter out sentences that are noisy, broken, or not meaningful"""
    filtered = []
    for s in sentences:
        s = s.strip()
        # Remove if too short, mostly punctuation, or single letters
        if len(s) < 20:
            continue
        if len(re.sub(r'[^A-Za-z0-9]', '', s)) < 10:
            continue
        if re.match(r"^[\W_]+$", s):
            continue
        # Remove if not enough words
        if len(s.split()) < 4:
            continue
        filtered.append(s)
    return filtered

def generate_article_summary(article, length="Medium"):
    """Generate a structured, detailed summary using BART and custom prompt, with fallback for short articles"""
    content = article.get('content', '')
    title = article.get('title', 'No Title')
    content = clean_article_content(content)
    if not content or len(content.strip()) < 50:
        # Fallback: always show extracted content, even if minimal
        raw_content = article.get('content', '')
        if raw_content:
            return (
                f"⚠️ Article content is too short for a full summary. "
                f"Below is all the extracted content from the article.\n\n"
                f"**Extracted Content:**\n{raw_content}\n\n"
                f"If this is not enough, try a different URL or check the source for more details."
            )
        else:
            return (
                "⚠️ No meaningful content could be extracted from this article. "
                "Try a different URL or check the source for more details."
            )
    # Custom prompt for structured output
    prompt = (
        f"Summarize the following article in a detailed, structured format with sections: "
        f"Share Price, Valuation, Performance, Volume, Shareholders, Profitability, Business Scope, IPO Buzz, Broader Context, Final Thoughts. "
        f"Include key metrics, insights, and context.\n\nArticle:\n{content}"
    )
    max_length = 350 if length == "Short" else 512 if length == "Medium" else 1024
    min_length = 120 if length == "Short" else 180 if length == "Medium" else 300
    try:
        summarizer_instance = get_summarizer()
        summary_list = summarizer_instance(prompt, max_length=max_length, min_length=min_length, do_sample=False)
        summary = summary_list[0]['summary_text']
        # Post-process summary to remove noise
        summary_sentences = re.split(r'[.!?]+', summary)
        summary_sentences = filter_sentences(summary_sentences)
        summary = '. '.join(summary_sentences)
        return f"**Summary:** {summary}"
    except Exception as e:
        return f"⚠️ Summarization failed: {str(e)}"

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
        
        article_data = fetch_article_content(url)
        
        if isinstance(article_data, dict):
            article_data['url'] = url
            article_data['processed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            article_data['sentiment'] = analyze_sentiment(article_data['content'])
            articles_data.append(article_data)
        else:
            st.warning(f"Error loading article from {url}: {article_data}")
    
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
with tab2:
    st.header("⚙️ Process & Analyze")

    # Check if we have a URL to process
    if 'current_url' not in st.session_state or not st.session_state.current_url:
        st.info("💡 Please enter a URL in the 'Add URL' tab first, then come back here to process it.")
        st.stop()

    current_url = st.session_state.current_url
    st.write(f"**🔗 Processing URL:** {current_url}")

    # Process button
    if st.button("🚀 Process Article", type="primary"):
        with st.spinner("🔄 Processing article..."):
            try:
                # Fetch and process the article
                article_data = fetch_article_content(current_url)

                if article_data:
                    # Store in session state
                    st.session_state.articles = [article_data]
                    st.session_state.processing = True
                    st.success("✅ Article processed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to process the article. Please check the URL and try again.")

            except Exception as e:
                st.error(f"❌ Error processing article: {str(e)}")

    # Show processing status
    if 'processing' in st.session_state and st.session_state.processing:
        st.success("✅ Article is ready for analysis!")

    # Main content area (only show if we have processed articles)
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
        st.header("📰 Article Details")

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
                search_type = st.selectbox("Search Type", ["Semantic", "AI-Powered"], help="Semantic: meaning-based search, AI-Powered: LLM reasoning with web search")

            # Initialize variables for all search types
            enable_web_search = True  # Always enable web search fallback for AI-Powered
            use_context = True  # Always use context when available

            # Add advanced AI toggle
            if search_type == "AI-Powered":
                st.info("🤖 AI-Powered mode uses advanced LLM reasoning to provide intelligent answers. If the AI doesn't have sufficient information, it automatically searches the web for current data.")

            # Clear query params if question was loaded
            if default_question:
                st.query_params.clear()

            # Add automatic search trigger for quick questions
            auto_search = bool(default_question)  # Auto-search if question came from quick button

            # Initialize randomization seed for this session if not exists
            if 'question_randomizer' not in st.session_state:
                st.session_state.question_randomizer = random.randint(1, 1000)

            # Use session-based random seed for consistent but varied questions
            random.seed(st.session_state.question_randomizer + int(time.time() // 300))  # Changes every 5 minutes

            # Smart AI-Generated Questions based on article content
            if st.session_state.articles:
                st.markdown("**🧠 Smart Questions (AI-Generated):**")

                # Generate intelligent questions based on actual article content
                sample_article = st.session_state.articles[0]
                article_content = sample_article.get('content', '').lower()
                article_title = sample_article.get('title', '')

                smart_questions = []

                # AI/Technology related questions with variations
                if any(word in article_content for word in ['ai', 'artificial intelligence', 'technology', 'innovation']):
                    tech_questions = [
                        "What are the AI and technology developments mentioned?",
                        "How is technology being implemented or discussed?",
                        "What innovation strategies are being pursued?",
                        "What technological advantages are highlighted?"
                    ]
                    smart_questions.append(random.choice(tech_questions))

                # Financial performance questions with variations
                if any(word in article_content for word in ['revenue', 'profit', 'earnings', 'financial']):
                    financial_questions = [
                        "What are the key financial highlights?",
                        "How is the company performing financially?",
                        "What are the revenue and profit trends?",
                        "What financial metrics are most important here?"
                    ]
                    smart_questions.append(random.choice(financial_questions))

                # Market/Stock performance questions with variations
                if any(word in article_content for word in ['stock', 'share', 'market', 'price']):
                    market_questions = [
                        "How is the stock/market performance?",
                        "What factors are affecting the stock price?",
                        "What market trends are discussed?",
                        "How is investor sentiment reflected?"
                    ]
                    smart_questions.append(random.choice(market_questions))

                # Strategic/Business questions with variations
                if any(word in article_content for word in ['strategy', 'plan', 'initiative', 'expansion']):
                    strategy_questions = [
                        "What business strategies are discussed?",
                        "What are the key strategic initiatives?",
                        "How is the company planning to grow?",
                        "What strategic changes are being made?"
                    ]
                    smart_questions.append(random.choice(strategy_questions))

                # Future outlook questions with variations
                if any(word in article_content for word in ['outlook', 'forecast', 'future', 'guidance', '2024', '2025', '2026']):
                    future_questions = [
                "What is the future outlook and predictions?",
                "What are the growth expectations?",
                "How does management view the future?",
                "What guidance has been provided?"
            ]
            smart_questions.append(random.choice(future_questions))
        
        # Competition/Industry questions with variations
        if any(word in article_content for word in ['competitor', 'industry', 'market share']):
            competition_questions = [
                "What competitive dynamics are mentioned?",
                "How does this compare to competitors?",
                "What industry trends are highlighted?",
                "What is the competitive advantage discussed?"
            ]
            smart_questions.append(random.choice(competition_questions))
        
        # Risk/Challenge questions with variations
        if any(word in article_content for word in ['risk', 'challenge', 'concern', 'issue']):
            risk_questions = [
                "What risks or challenges are discussed?",
                "What concerns are being addressed?",
                "What obstacles does the company face?",
                "How are potential risks being managed?"
            ]
            smart_questions.append(random.choice(risk_questions))
        
        # Default questions if nothing specific found - also randomized
        if not smart_questions:
            company_name = article_title.split()[0] if article_title else 'this company'
            default_questions = [
                f"What are the main points about {company_name}?",
                "What are the key takeaways from this article?",
                "What factors are affecting the business?",
                "What is the most important information here?",
                "What should investors know about this?",
                "What are the critical insights from this news?"
            ]
            smart_questions = random.sample(default_questions, min(3, len(default_questions)))
        
        # Randomize order and limit to 3-4 questions to avoid clutter
        if len(smart_questions) > 4:
            smart_questions = random.sample(smart_questions, 4)
        else:
            random.shuffle(smart_questions)
        
        # Display questions with refresh button
        question_col, refresh_col = st.columns([6, 1])
        with refresh_col:
            if st.button("🔄", help="Refresh questions for different suggestions"):
                st.session_state.question_randomizer = random.randint(1, 1000)
                st.rerun()
        
        with question_col:
            cols = st.columns(len(smart_questions))
            for i, q in enumerate(smart_questions):
                with cols[i]:
                    if st.button(q, key=f"smart_q_{i}", help="AI-generated question based on article content"):
                        st.query_params["question"] = q
                        st.rerun()
    
            if (question and st.button("🔍 Search Answer")) or auto_search:
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
                        # Regular semantic search for specific questions
                        for i, article in enumerate(st.session_state.articles):
                            content = article.get('content', '')
                            title = article.get('title', 'No Title')
                            content_lower = content.lower()
                            title_lower = title.lower()

                            # Enhanced relevance scoring - Semantic matching
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

                    # Only show results summary for Semantic searches, not AI-Powered
                    if search_type == "Semantic":
                        if results:
                            st.success(f"🎯 Found {len(results)} relevant source(s)")

                            for idx, result in enumerate(results[:5]):  # Show top 5 results
                                with st.expander(f"� {result['title'][:60]}... (Relevance: {result['score']})"):
                                    st.write(f"**🔗 Source:** {result['url']}")
                                    st.write(f"**🌐 Domain:** {result['domain']}")
                                    st.write(f"**🎯 Relevant excerpts:**")
                                    for sentence in result['sentences']:
                                        st.write(f"💡 {sentence}")
                        else:
                            st.warning("🤔 No relevant information found. Try rephrasing your question or using different keywords.")
                    elif search_type == "AI-Powered":
                        st.info("🤖 Using AI-Powered analysis to generate intelligent answers...")
                        
                        # Use AI with web search for better answers
                        use_article_context = True
                        enable_web_search = True
                        
                        # Get enhanced real-time AI response with article context
                        ai_response, used_web_search = generate_realtime_ai_answer(question, st.session_state.articles, use_context=use_article_context, enable_web_search=enable_web_search)
                        
                        # Display enhanced response with Tavily indicators
                        with st.expander("💡 AI Response", expanded=True):
                            # Add Tavily enhancement badge if web search was used
                            if used_web_search:
                                if 'Latest Market Intelligence' in ai_response:
                                    st.markdown("🌐 **Enhanced with Tavily Real-time Data** | 🎯 **Latest Financial Intelligence**")
                                else:
                                    st.markdown("🔍 **Enhanced with Web Search**")
                            
                            st.markdown(ai_response)
                        
                        # Show enhanced status message
                        context_type = "Article-Aware" if use_article_context else "Real-time Enhanced"
                        
                        if used_web_search:
                            if 'Latest Market Intelligence' in ai_response:
                                search_method = "🌐 Tavily Intelligence"
                            else:
                                search_method = "🔍 Web Search"
                        else:
                            search_method = "🤖 AI Analysis"
                            
                        st.success(f"{search_method} Complete - Mode: {context_type}")
                    else:
                        # For other search types, show semantic search results
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
                
            else:
                # Handle other search types (Keyword, Semantic)
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
                    
                    # Enhanced relevance scoring - Semantic matching
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
            
            # Only show results summary for Semantic searches, not AI-Powered
            if search_type == "Semantic":
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
- **Enable web search** in AI-Powered mode for current information
""")

st.sidebar.markdown("### 🔧 Troubleshooting")
st.sidebar.markdown("""
- **Slow loading?** Some sites block automated requests
- **No content?** Check if URL is accessible and contains text
- **Empty results?** Try rephrasing your questions
- **Generic AI answers?** Enable web search for current data
""")
