"""
EquityGPT - Unique Financial Intelligence Platform
A specialized web-based AI assistant for real-time financial news analysis
that goes beyond generic chatbots like ChatGPT, Claude, or Gemini.
"""

import os
import re
import json
import time
import warnings
from io import BytesIO
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from app.ai_original import generate_realtime_ai_answer
from app.utils import clean_text_content
# Force reload utils and web modules to get latest version
import importlib
import app.utils
import app.web
importlib.reload(app.utils)
importlib.reload(app.web)
from app.utils import clean_text_content
from app.predictive_analysis import (
    predict_price_movement,
    assess_investment_risk,
    generate_deep_insights,
    predict_target_price,
    analyze_sentiment_trend,
)

# Load environment variables (e.g., TAVILY_API_KEY)
load_dotenv()

# ========================================

st.set_page_config(
    page_title="EquityGPT - Financial Intelligence AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# THEME INITIALIZATION
# ========================================

# Initialize theme in session state if not present
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # Default to light theme

# ========================================
# CUSTOM CSS - DYNAMIC THEME SUPPORT
# ========================================

# Define theme-specific colors
if st.session_state.theme == 'light':
    theme_css = """
    <style>
        /* Light Theme */
        .stApp {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        
        .main .block-container {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        
        /* Main theme colors */
        :root {
            --primary-color: #1f77b4;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --bg-color: #ffffff;
            --text-color: #262730;
            --card-bg: #f8f9fa;
            --chat-bg: rgba(240, 242, 246, 0.8);
            --border-color: #e0e0e0;
        }
        
        /* Global text color for all elements */
        body, p, span, div, h1, h2, h3, h4, h5, h6, label, li, a {
            color: #262730 !important;
        }
        
        /* Streamlit specific text elements */
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #262730 !important;
        }
        
        .stText, .stCaption {
            color: #262730 !important;
        }
        
        /* Headers */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #262730 !important;
        }
        
        /* Chat message styling */
        .stChatMessage {
            background-color: var(--chat-bg) !important;
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            color: #262730 !important;
        }
        
        .stChatMessage p, .stChatMessage span, .stChatMessage div {
            color: #262730 !important;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white !important;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Streamlit metrics */
        [data-testid="stMetricValue"] {
            color: #262730 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #666666 !important;
        }
        
        /* Article preview cards */
        .article-card {
            border-left: 4px solid #1f77b4;
            padding: 15px;
            background-color: var(--card-bg);
            border-radius: 8px;
            margin: 10px 0;
            color: #262730 !important;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .badge-success {
            background-color: #d4edda;
            color: #155724 !important;
        }
        
        .badge-warning {
            background-color: #fff3cd;
            color: #856404 !important;
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] label {
            color: #262730 !important;
        }
        
        /* Button styling */
        .stButton > button {
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--card-bg) !important;
            color: #262730 !important;
        }
        
        .streamlit-expanderContent {
            background-color: var(--card-bg) !important;
            color: #262730 !important;
        }
        
        /* Input fields */
        .stTextInput input, .stTextArea textarea {
            background-color: #ffffff !important;
            color: #262730 !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .stTextInput label, .stTextArea label {
            color: #262730 !important;
        }
        
        /* Radio buttons and checkboxes */
        .stRadio label, .stCheckbox label {
            color: #262730 !important;
        }
        
        /* Selectbox */
        .stSelectbox label {
            color: #262730 !important;
        }
        
        /* Info/warning/success boxes */
        .stAlert {
            color: #262730 !important;
        }
        
        /* Divider */
        hr {
            border-color: #e0e0e0 !important;
        }
        
        /* Code blocks */
        code, pre {
            background-color: #f8f9fa !important;
            color: #262730 !important;
        }
        
        /* Dataframe */
        .stDataFrame {
            color: #262730 !important;
        }
    </style>
    """
else:
    theme_css = """
    <style>
        /* Dark Theme */
        .stApp {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        
        .main .block-container {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        
        /* Main theme colors */
        :root {
            --primary-color: #4da6ff;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --bg-color: #0e1117;
            --text-color: #fafafa;
            --card-bg: #1e222a;
            --chat-bg: rgba(30, 34, 42, 0.8);
            --border-color: #2e3440;
        }
        
        /* Global text color for all elements */
        body, p, span, div, h1, h2, h3, h4, h5, h6, label, li, a {
            color: #fafafa !important;
        }
        
        /* Streamlit specific text elements */
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #fafafa !important;
        }
        
        .stText, .stCaption {
            color: #fafafa !important;
        }
        
        /* Headers */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #fafafa !important;
        }
        
        /* Chat message styling */
        .stChatMessage {
            background-color: var(--chat-bg) !important;
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            color: #fafafa !important;
        }
        
        .stChatMessage p, .stChatMessage span, .stChatMessage div {
            color: #fafafa !important;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
            padding: 20px;
            border-radius: 10px;
            color: white !important;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        /* Streamlit metrics */
        [data-testid="stMetricValue"] {
            color: #fafafa !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #b0b0b0 !important;
        }
        
        /* Article preview cards */
        .article-card {
            border-left: 4px solid #4da6ff;
            padding: 15px;
            background-color: var(--card-bg);
            border-radius: 8px;
            margin: 10px 0;
            color: #fafafa !important;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .badge-success {
            background-color: #1e4620;
            color: #7fd687 !important;
        }
        
        .badge-warning {
            background-color: #4a3600;
            color: #ffd966 !important;
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
            color: white !important;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1e222a !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] label {
            color: #fafafa !important;
        }
        
        /* Button styling */
        .stButton > button {
            transition: all 0.3s ease;
            background-color: #262730;
            color: #fafafa !important;
            border: 1px solid #2e3440;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.4);
            background-color: #31363f;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--card-bg) !important;
            color: #fafafa !important;
        }
        
        .streamlit-expanderContent {
            background-color: var(--card-bg) !important;
            color: #fafafa !important;
        }
        
        /* Input fields */
        .stTextInput input, .stTextArea textarea {
            background-color: #1e222a !important;
            color: #fafafa !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .stTextInput label, .stTextArea label {
            color: #fafafa !important;
        }
        
        /* Radio buttons and checkboxes */
        .stRadio label, .stCheckbox label {
            color: #fafafa !important;
        }
        
        /* Selectbox */
        .stSelectbox label {
            color: #fafafa !important;
        }
        
        .stSelectbox div[data-baseweb="select"] {
            background-color: #1e222a !important;
            color: #fafafa !important;
        }
        
        /* Info/warning/success boxes */
        .stAlert {
            background-color: #1e222a !important;
            color: #fafafa !important;
        }
        
        /* Divider */
        hr {
            border-color: #2e3440 !important;
        }
        
        /* Code blocks */
        code, pre {
            background-color: #1e222a !important;
            color: #fafafa !important;
        }
        
        /* Dataframe */
        .stDataFrame {
            color: #fafafa !important;
        }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# ========================================
# UTILITY FUNCTIONS
# ========================================

def extract_urls_from_text(text: str) -> Tuple[List[str], str]:
    """
    Extract URLs from user input and return cleaned question.
    
    Returns:
        (urls, cleaned_question)
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    # Remove URLs from question
    cleaned_question = re.sub(url_pattern, '[ARTICLE]', text).strip()
    cleaned_question = re.sub(r'\s+', ' ', cleaned_question)
    
    return urls, cleaned_question


def extract_article(url: str, timeout: int = 10) -> Optional[Dict]:
    """
    Extract article content from URL using robust extraction.
    
    Returns:
        dict with title, text, published_date, source
    """
    try:
        # 0) Prefer robust extractor first (earlier working behavior)
        try:
            from app.web import fetch_article_content as _robust_fetch_article
        except Exception:
            _robust_fetch_article = None

        if _robust_fetch_article is not None:
            try:
                ra = _robust_fetch_article(url, max_length=6000)
                if ra and ra.get('success') and ra.get('content') and len(ra['content'].strip()) > 120:
                    source = ra.get('domain') or urlparse(url).netloc.replace('www.', '')
                    title_val = ra.get('title') or 'Untitled Article'
                    text_val = clean_text_content(ra.get('content', ''))
                    return {
                        'url': url,
                        'title': title_val,
                        'text': text_val,
                        'source': source,
                        'extracted_at': datetime.now().isoformat(),
                        'word_count': len(text_val.split())
                    }
            except Exception:
                pass

        # Direct extraction with smart filtering
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
        }
        
        # 0b) Times of India: try reader-mode variation first
        parsed = urlparse(url)
        domain_l = parsed.netloc.lower()
        try_urls = []
        if 'timesofindia' in domain_l or 'indiatimes' in domain_l:
            # append ?from=mdr if not present
            if parsed.query:
                alt = url + ('&from=mdr' if 'from=mdr' not in parsed.query else '')
            else:
                alt = url + '?from=mdr'
            try_urls = [alt, url]
        else:
            try_urls = [url]

        response = None
        last_err = None
        for u in try_urls:
            try:
                r = requests.get(u, headers=headers, timeout=timeout)
                r.raise_for_status()
                response = r
                if u != url:
                    st.info("🧪 Used reader-mode variant for extraction")
                break
            except Exception as e:
                last_err = e
                continue
        if response is None:
            if last_err:
                raise last_err
            else:
                raise RuntimeError("Failed to fetch URL")
        response.raise_for_status()
        
        # Use response.text instead of response.content for proper encoding handling
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract canonical URL and title candidates FIRST
        canonical = None
        can_tag = soup.find('link', rel='canonical')
        if can_tag and can_tag.get('href'):
            canonical = can_tag['href']
        og_url = soup.find('meta', property='og:url')
        if not canonical and og_url and og_url.get('content'):
            canonical = og_url['content']

        title = None
        title_elem = soup.find('h1')
        if title_elem:
            title = title_elem.get_text(strip=True)
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title['content'].strip()
        if not title:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
        title = title or "Untitled Article"

        # Build keyword sets from title to anchor relevance
        def tokenize(text: str) -> List[str]:
            return re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text.lower())

        stop = {
            'the','and','for','with','from','into','that','this','those','these','over','under','about','after','before',
            'of','in','on','to','by','as','at','it','its','is','are','was','were','be','has','have','had','a','an'
        }
        title_tokens = [t for t in tokenize(title) if t not in stop]
        # Prefer proper nouns and domain terms we often care about
        likely_terms = set([t for t in title_tokens if len(t) >= 4])
        # Common negative/noise keywords we want to avoid
        negative_terms = {
            'safari','iphone','ipad','reader','reader mode','summarize','screenshot','button','click','steps','how to',
            'subscribe','sign in','register','login','download','advertisement','promoted','sponsored','tutorial','guide'
        }
        # Site chrome/common noise phrases (ET/TOI etc.)
        negative_terms |= {
            'economic times','follow channel','whatsapp','google logo','read more','newsletter','most popular',
            'live tv','video','comments','share this','trending now','toi','times of india','toi plus'
        }
        # A small helper to score relevance to title and penalize negatives
        def score_text(text: str) -> float:
            tl = text.lower()
            pos = 0
            for term in likely_terms:
                if term in tl:
                    pos += 1
            neg = 0
            for term in negative_terms:
                if term in tl:
                    neg += 1
            # Weight positives more than negatives but penalize strongly
            return pos * 2.0 - neg * 3.0
        
        # Remove scripts and styles only
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # Try JSON-LD first (cleanest method) but validate against page title and canonical URL
        article_text = None
        try:
            jsonld_scripts = soup.find_all('script', type='application/ld+json')
            for script in jsonld_scripts:
                try:
                    data = json.loads(script.string or "{}")
                except Exception:
                    continue

                candidates = []
                if isinstance(data, dict):
                    candidates = [data]
                elif isinstance(data, list):
                    candidates = [d for d in data if isinstance(d, dict)]

                for d in candidates:
                    if d.get('@type') in ['NewsArticle', 'Article']:
                        ld_headline = d.get('headline') or d.get('name') or ''
                        ld_body = d.get('articleBody') or ''
                        ld_url = d.get('url') or d.get('mainEntityOfPage') or ''
                        if not ld_body or len(ld_body) < 300:
                            continue
                        # Validate: headline similarity and body relevance to title
                        h_tokens = set([t for t in tokenize(ld_headline) if t not in stop])
                        overlap = len(h_tokens & likely_terms)
                        body_score = score_text(ld_body)
                        neg_hits = sum(1 for term in negative_terms if term in ld_body.lower())
                        # Extra URL match boost if canonical matches
                        url_match = 0
                        if canonical and isinstance(ld_url, str) and ld_url:
                            try:
                                url_match = 2 if urlparse(ld_url).path == urlparse(canonical).path else (1 if urlparse(ld_url).netloc == urlparse(canonical).netloc else 0)
                            except Exception:
                                url_match = 0

                        if overlap >= 1 or body_score + url_match >= 1:
                            if neg_hits >= 2 and body_score <= 0:
                                # Likely unrelated tutorial embedded in JSON-LD
                                st.warning("⏭️ Skipped JSON-LD candidate due to off-topic signals")
                                continue
                            candidate_text = clean_text_content(ld_body)
                            st.info(f"✅ JSON-LD candidate accepted (overlap={overlap}, score={body_score:.1f}, url_match={url_match})")
                            article_text = candidate_text
                            break
                if article_text:
                    break
        except Exception:
            pass
        
        # If JSON-LD didn't work, try Times of India specific structure
        if not article_text or len(article_text) < 300:
            # Look for the main article container - score by title relevance
            # Gather candidate containers
            candidates = []
            # Common article containers
            for name, selector in [
                ('arttextxml', ('div', {'class': 'arttextxml'})),
                ('artText', ('div', {'class': 'artText'})),
                ('article', ('article', {})),
                ('main', ('main', {})),
                ('itemprop_articleBody_div', ('div', {'itemprop': 'articleBody'})),
                ('itemprop_articleBody_section', ('section', {'itemprop': 'articleBody'})),
            ]:
                sel = soup.find(*selector)
                if sel:
                    candidates.append((name, sel))

            # Heuristic: any div/section with id/class containing article/story/content
            def add_candidates_by_attr(tag_name):
                for node in soup.find_all(tag_name):
                    cid = (node.get('id') or '').lower()
                    cls = ' '.join(node.get('class') or []).lower()
                    if any(k in cid for k in ['article', 'story', 'content']) or any(k in cls for k in ['article', 'story', 'content', 'body']):
                        # Require some <p> children to avoid nav boxes
                        if len(node.find_all('p')) >= 3:
                            candidates.append((f'{tag_name}#{cid or cls[:20]}', node))
            add_candidates_by_attr('div')
            add_candidates_by_attr('section')

            best = None
            best_score = float('-inf')
            for name, container in candidates:
                paragraphs = container.find_all('p', recursive=True)
                raw = ' '.join(p.get_text(strip=True) for p in paragraphs)
                c_score = score_text(raw)
                st.info(f"🎯 Candidate {name}: score={c_score:.1f}, paragraphs={len(paragraphs)}")
                if c_score > best_score and len(raw) > 300:
                    best = (name, container)
                    best_score = c_score

            if best:
                name, container = best
                paragraphs = container.find_all('p', recursive=True)
                st.info(f"📝 Selected container '{name}' with {len(paragraphs)} <p> tags (score={best_score:.1f})")
                if paragraphs and len(paragraphs) >= 2:
                    meaningful = []
                    kept = 0
                    skipped = 0
                    for idx, p in enumerate(paragraphs):
                        text = p.get_text(strip=True)
                        if idx < 5:
                            st.text(f"Para {idx}: {text[:100]}...")
                        if len(text) < 40:
                            skipped += 1
                            continue
                        tl = text.lower()
                        neg_hit = any(term in tl for term in negative_terms)
                        pos_hit = any(term in tl for term in likely_terms) if likely_terms else True
                        # Allow first 2 paragraphs even without positive hits (lead-in), but never allow negative-only
                        if neg_hit and not pos_hit:
                            skipped += 1
                            continue
                        if not pos_hit and idx > 2:
                            skipped += 1
                            continue
                        meaningful.append(text)
                        kept += 1
                        if len(' '.join(meaningful)) > 2500:
                            break
                    if len(meaningful) >= 2:
                        article_text = ' '.join(meaningful)
                        st.success(f"✅ Extracted via container: {len(article_text)} characters (kept {kept}, skipped {skipped})")

        # Domain-specific fallback for Times of India: sometimes body is minimal/static
        domain = urlparse(url).netloc.replace('www.', '').lower()
        if (not article_text or len(article_text) < 200) and ('timesofindia' in domain or 'indiatimes' in domain):
            st.warning("🔧 Applying Times of India fallback extraction")
            candidates = []
            # Look for large text blocks composed of <p> or text-like div/span
            for container in soup.find_all(['article', 'section', 'div'], limit=30):
                paras = container.find_all(['p', 'div', 'span'])
                filtered = []
                for node in paras:
                    cls = ' '.join(node.get('class') or []).lower()
                    tag = node.name
                    text = node.get_text(strip=True)
                    if len(text) < 60:
                        continue
                    if any(k in cls for k in ['author','share','subscribe','breadcrumb','related','comment','social','widget','footer','header','nav']):
                        continue
                    if tag in ['div', 'span'] and not any(k in cls for k in ['normal','content','article','story','text']):
                        continue
                    filtered.append(text)
                full = ' '.join(filtered)
                if len(full) > 300:
                    score = score_text(full)
                    candidates.append((score, len(filtered), container, full))
            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1], len(x[3])), reverse=True)
                best_score, count, cont, full = candidates[0]
                st.info(f"🧭 TOI fallback picked block with score={best_score:.1f}, paras={count}, length={len(full)}")
                article_text = clean_text_content(full[:3000])
        
        # Last resort: get all paragraphs with relevance filtering
        if not article_text or len(article_text) < 200:
            all_paragraphs = soup.find_all('p')
            meaningful = []
            
            st.info(f"🔍 Found {len(all_paragraphs)} total paragraphs, filtering by relevance...")
            
            for p in all_paragraphs:
                text = p.get_text(strip=True)
                
                # Basic length check
                if len(text) < 40:
                    continue
                
                tl = text.lower()
                if any(term in tl for term in negative_terms):
                    continue
                
                # Check if mostly text (not just punctuation/numbers)
                alpha_count = sum(c.isalpha() for c in text)
                if alpha_count / len(text) < 0.5:
                    continue
                # Prefer paragraphs that match title terms
                if likely_terms:
                    if any(term in tl for term in likely_terms):
                        meaningful.append(text)
                else:
                    meaningful.append(text)
                
                if len(' '.join(meaningful)) > 2500:
                    break
            
            if meaningful:
                article_text = ' '.join(meaningful[:20])
                st.info(f"✅ Extracted via fallback: {len(article_text)} characters from {len(meaningful)} paragraphs")

        # Meta description fallback for TOI/indiatimes if still too short
        if (not article_text or len(article_text) < 150) and ('timesofindia' in domain or 'indiatimes' in domain):
            meta_desc = None
            md = soup.find('meta', attrs={'name': 'description'})
            if md and md.get('content'):
                meta_desc = md['content'].strip()
            if not meta_desc:
                og_desc = soup.find('meta', property='og:description')
                if og_desc and og_desc.get('content'):
                    meta_desc = og_desc['content'].strip()
            if meta_desc and len(meta_desc) > 80:
                st.info("🪄 Using meta description as minimal content due to site restrictions")
                article_text = meta_desc
        
        # Clean the text
        if article_text:
            # DEBUG: Test if cleaning corrupts the text
            text_before_clean = article_text[:100]
            garbled_before = sum(1 for c in text_before_clean if ord(c) > 1000)
            
            article_text = clean_text_content(article_text)
            
            text_after_clean = article_text[:100]
            garbled_after = sum(1 for c in text_after_clean if ord(c) > 1000)
            
            if garbled_after > garbled_before:
                st.error(f"🔴 CLEANING CORRUPTS TEXT! Before: {garbled_before} garbled, After: {garbled_after} garbled")
                st.code(f"Before: {text_before_clean[:50]}")
                st.code(f"After: {text_after_clean[:50]}")

        
        # Final validation (relax threshold for Times of India)
        min_len = 120 if ('timesofindia' in (urlparse(url).netloc.lower()) or 'indiatimes' in (urlparse(url).netloc.lower())) else 150
        if not article_text or len(article_text) < min_len:
            st.error(f"❌ Insufficient content extracted: {len(article_text) if article_text else 0} characters")
            st.warning("💡 Try a different news source (Reuters, Bloomberg, Financial Times work well)")
            
            # Debug: show what we found
            if article_text:
                st.text_area("Debug - Extracted text:", article_text[:500], height=150)
            
            return None
        
        # Extract source
        source = urlparse(url).netloc.replace('www.', '')
        
        return {
            'url': url,
            'title': title,
            'text': article_text,
            'source': source,
            'extracted_at': datetime.now().isoformat(),
            'word_count': len(article_text.split())
        }
    
    except Exception as e:
        st.error(f"❌ Extraction error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def calculate_sentiment_score(text: str) -> Dict:
    """
    Calculate sentiment score for text.
    Returns score between -1 (negative) and 1 (positive).
    """
    try:
        from transformers import pipeline
        
        # Use cached sentiment analyzer
        if 'sentiment_analyzer' not in st.session_state:
            st.session_state.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        
        analyzer = st.session_state.sentiment_analyzer
        
        # Analyze in chunks (models have token limits)
        chunk_size = 500
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        results = []
        for chunk in chunks[:5]:  # Limit to first 5 chunks
            result = analyzer(chunk[:512])[0]  # Token limit
            results.append(result)
        
        # Average scores
        positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
        avg_score = positive_count / len(results) if results else 0.5
        
        # Convert to -1 to 1 scale
        sentiment_score = (avg_score * 2) - 1
        
        return {
            'score': sentiment_score,
            'label': 'Positive' if sentiment_score > 0.2 else 'Negative' if sentiment_score < -0.2 else 'Neutral',
            'confidence': abs(sentiment_score)
        }
    
    except Exception as e:
        return {
            'score': 0,
            'label': 'Unknown',
            'confidence': 0,
            'error': str(e)
        }


def extract_companies_and_tickers(text: str) -> List[Dict]:
    """
    Extract company names and ticker symbols from text.
    """
    companies = []
    
    # Common ticker pattern
    ticker_pattern = r'\b[A-Z]{2,5}\b(?=\s|\.|\,|\))'
    tickers = re.findall(ticker_pattern, text)
    
    # Filter out common words
    excluded_words = {'NYSE', 'NASDAQ', 'USD', 'CEO', 'CFO', 'IPO', 'ETF', 'US', 'UK'}
    tickers = [t for t in tickers if t not in excluded_words]
    
    # Company name patterns (simplified)
    company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Co|Group|Holdings)\.?))\b'
    company_names = re.findall(company_pattern, text)
    
    # Combine results
    for ticker in set(tickers[:10]):  # Limit to 10
        companies.append({'ticker': ticker, 'name': None})
    
    for name in set(company_names[:10]):
        companies.append({'ticker': None, 'name': name})
    
    return companies


def create_sentiment_gauge(sentiment_score: float) -> go.Figure:
    """
    Create a beautiful gauge chart for sentiment.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score", 'font': {'size': 24}},
        delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.3], 'color': '#ffcccc'},
                {'range': [-0.3, 0.3], 'color': '#ffffcc'},
                {'range': [0.3, 1], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_article_comparison_chart(articles: List[Dict]) -> go.Figure:
    """
    Create comparison chart for multiple articles.
    """
    if not articles:
        return None
    
    sources = [a.get('source', 'Unknown') for a in articles]
    sentiments = [calculate_sentiment_score(a.get('text', ''))['score'] for a in articles]
    word_counts = [a.get('word_count', 0) for a in articles]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sources,
        y=sentiments,
        name='Sentiment Score',
        marker_color=['#2ecc71' if s > 0 else '#e74c3c' for s in sentiments],
        text=[f"{s:.2f}" for s in sentiments],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Article Sentiment Comparison",
        xaxis_title="Source",
        yaxis_title="Sentiment Score",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_range=[-1, 1],
        showlegend=False
    )
    
    return fig


def export_to_excel(articles: List[Dict], analysis_history: List[Dict]) -> BytesIO:
    """
    Export analysis to Excel file.
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Articles sheet
        if articles:
            df_articles = pd.DataFrame(articles)
            df_articles.to_excel(writer, sheet_name='Articles', index=False)
        
        # Analysis history sheet
        if analysis_history:
            df_history = pd.DataFrame(analysis_history)
            df_history.to_excel(writer, sheet_name='Analysis History', index=False)
    
    output.seek(0)
    return output


# ========================================
# SESSION STATE INITIALIZATION
# ========================================

def initialize_session_state():
    """Initialize all session state variables."""
    # Check for clear cache query parameter
    try:
        query_params = st.query_params
        if query_params.get('clear_cache') == 'true':
            st.session_state.clear()
            st.success("✅ Cache cleared! Remove ?clear_cache=true from URL and refresh.")
    except Exception:
        pass
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 **Welcome to EquityGPT!**\n\nI'm your specialized financial intelligence assistant. Unlike generic AI chatbots, I provide:\n\n✅ Real-time article analysis with visual insights\n✅ Multi-source cross-verification\n✅ Sentiment tracking and company extraction\n✅ Export to PDF/Excel reports\n\n**Just paste article URLs and ask questions!**\n\n💡 *Example: \"Analyze sentiment of https://reuters.com/tesla-earnings\"*"
            }
        ]
    
    if 'articles_cache' not in st.session_state:
        st.session_state.articles_cache = {}
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if 'sentiment_timeline' not in st.session_state:
        st.session_state.sentiment_timeline = []
    
    # Conversation memory for context-aware responses
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {
            'last_company': None,
            'last_topic': None,
            'last_question': None,
            'last_answer': None,
            'entities_discussed': [],  # Track all companies mentioned
            'topics_by_entity': {},  # Map: company -> [topics]
            'conversation_history': []  # Full context chain
        }

initialize_session_state()

# ========================================
# MAIN HEADER
# ========================================

st.markdown("""
<div class="main-header">
    <h1>📊 EquityGPT</h1>
    <p>Your Unique Financial Intelligence Platform</p>
    <p style="font-size: 14px; opacity: 0.9;">
        Beyond ChatGPT • Real-time Analysis • Visual Intelligence • Expert Reports
    </p>
</div>
""", unsafe_allow_html=True)

# ========================================
# SIDEBAR - ANALYTICS & CONTROLS
# ========================================

with st.sidebar:
    # Theme Toggle at the top
    st.subheader("🎨 Theme Settings")
    
    # Initialize theme in session state if not present
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    
    # Theme selector
    theme_choice = st.radio(
        "Choose Theme:",
        options=['Light', 'Dark'],
        index=0 if st.session_state.theme == 'light' else 1,
        horizontal=True,
        key='theme_selector'
    )
    
    # Update theme in session state
    new_theme = theme_choice.lower()
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()
    
    st.divider()
    
    st.header("📊 Analytics Dashboard")
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Articles", len(st.session_state.articles_cache))
    with col2:
        st.metric("💬 Analyses", len(st.session_state.analysis_history))
    
    st.divider()
    
    # Current articles in context
    st.subheader("📚 Loaded Articles")
    
    if st.session_state.articles_cache:
        for idx, (url, article) in enumerate(st.session_state.articles_cache.items(), 1):
            with st.expander(f"📄 {idx}. {article.get('source', 'Unknown')}"):
                st.caption(f"**{article.get('title', 'Untitled')[:60]}...**")
                st.caption(f"🔗 {url[:50]}...")
                st.caption(f"📝 {article.get('word_count', 0)} words")
                
                # Mini sentiment indicator
                sentiment = calculate_sentiment_score(article.get('text', ''))
                sentiment_color = '🟢' if sentiment['score'] > 0.2 else '🔴' if sentiment['score'] < -0.2 else '🟡'
                st.caption(f"{sentiment_color} Sentiment: {sentiment['label']}")
    else:
        st.info("No articles loaded yet.\n\nPaste URLs in your chat messages!")
    
    st.divider()
    
    # Export options
    st.subheader("📥 Export Options")
    
    if st.session_state.articles_cache or st.session_state.analysis_history:
        if st.button("📊 Download Excel Report", use_container_width=True):
            excel_data = export_to_excel(
                list(st.session_state.articles_cache.values()),
                st.session_state.analysis_history
            )
            st.download_button(
                label="💾 Save Excel File",
                data=excel_data,
                file_name=f"equitygpt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.caption("⚠️ No data to export yet")
    
    st.divider()
    
    # Cache management
    st.subheader("🔧 Cache Management")
    
    if st.session_state.articles_cache:
        if st.button("🗑️ Clear All Cached Articles", use_container_width=True, type="secondary"):
            st.session_state.articles_cache = {}
            st.session_state.messages = []
            st.session_state.analysis_history = []
            st.session_state.sentiment_timeline = []
            st.success("✅ Cache cleared! Refresh the page.")
            st.rerun()
    else:
        st.caption("No cached articles to clear")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    
    enable_web_search = st.checkbox("🌐 Real-time Web Search", value=True, 
                                     help="Enable Tavily search for enhanced answers")
    
    show_visualizations = st.checkbox("📊 Show Visual Analytics", value=True,
                                       help="Display charts and graphs")
    
    auto_sentiment = st.checkbox("🎯 Auto Sentiment Analysis", value=True,
                                  help="Automatically analyze sentiment for new articles")
    
    st.divider()
    
    # Clear data
    if st.button("🗑️ New Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
        st.session_state.articles_cache = {}
        st.session_state.analysis_history = []
        st.session_state.sentiment_timeline = []
        st.rerun()

# ========================================
# MAIN CONTENT AREA
# ========================================

# Visual Analytics Section (Top)
if show_visualizations and st.session_state.articles_cache:
    st.subheader("📈 Visual Intelligence Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average sentiment gauge
        all_texts = [a['text'] for a in st.session_state.articles_cache.values() if a.get('text')]
        if all_texts:
            avg_sentiment = sum(calculate_sentiment_score(t)['score'] for t in all_texts) / len(all_texts)
            fig = create_sentiment_gauge(avg_sentiment)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Article comparison
        articles_list = list(st.session_state.articles_cache.values())
        if len(articles_list) > 1:
            comparison_fig = create_article_comparison_chart(articles_list)
            if comparison_fig:
                st.plotly_chart(comparison_fig, use_container_width=True)
        else:
            st.info("📊 Add more articles to see comparison chart")
    
    with col3:
        # Company mentions
        all_companies = []
        for article in st.session_state.articles_cache.values():
            companies = extract_companies_and_tickers(article.get('text', ''))
            all_companies.extend(companies)
        
        if all_companies:
            st.markdown("**🏢 Mentioned Companies/Tickers:**")
            unique_items = set()
            for comp in all_companies[:10]:
                if comp['ticker']:
                    unique_items.add(comp['ticker'])
                if comp['name']:
                    unique_items.add(comp['name'])
            
            for item in list(unique_items)[:8]:
                st.markdown(f"• `{item}`")
        else:
            st.info("🏢 No companies detected yet")
    
    st.divider()

# ========================================
# PREDICTIVE ANALYSIS DASHBOARD
# ========================================

if show_visualizations and st.session_state.articles_cache:
    st.subheader("🔮 Predictive Analysis & Deep Insights")
    
    articles_list = list(st.session_state.articles_cache.values())
    
    # Calculate average sentiment
    all_texts = [a['text'] for a in articles_list if a.get('text')]
    avg_sentiment = sum(calculate_sentiment_score(t)['score'] for t in all_texts) / len(all_texts) if all_texts else 0
    
    # Create tabs for different analyses
    pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs([
        "📈 Price Prediction", 
        "⚠️ Risk Assessment", 
        "🎯 Deep Insights",
        "📊 Trend Analysis"
    ])
    
    with pred_tab1:
        # Price Movement Prediction
        price_pred = predict_price_movement(articles_list, avg_sentiment)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Price Movement Forecast")
            
            if price_pred['direction'] == 'upward':
                st.success(f"📈 **Predicted Direction:** Upward Movement")
                st.metric("Probability", f"{price_pred['probability']*100:.1f}%", 
                         delta=f"{price_pred['bullish_signals']} bullish signals")
            elif price_pred['direction'] == 'downward':
                st.error(f"📉 **Predicted Direction:** Downward Movement")
                st.metric("Probability", f"{price_pred['probability']*100:.1f}%", 
                         delta=f"-{price_pred['bearish_signals']} bearish signals", delta_color="inverse")
            else:
                st.info(f"➡️ **Predicted Direction:** Neutral/Sideways")
                st.metric("Confidence", "50%")
            
            st.markdown(f"**Analysis:** {price_pred['reasoning']}")
            st.caption(f"Confidence Level: {price_pred['confidence_level'].upper()}")
        
        with col2:
            # Price target prediction
            target_pred = predict_target_price(articles_list)
            
            if target_pred['predicted_target']:
                st.metric("Consensus Target", f"${target_pred['predicted_target']}")
                st.metric("Target Range", 
                         f"${target_pred['range_low']} - ${target_pred['range_high']}")
                if target_pred['upside_percent']:
                    st.metric("Upside Potential", 
                             f"{target_pred['upside_percent']}%",
                             delta=f"{target_pred['num_analysts']} analysts")
            else:
                st.info("No price targets found in articles")
    
    with pred_tab2:
        # Risk Assessment
        risk_assessment = assess_investment_risk(articles_list, avg_sentiment)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Risk gauge
            risk_score = risk_assessment['risk_score']
            risk_color = 'red' if risk_score > 0.6 else 'yellow' if risk_score > 0.3 else 'green'
            
            st.markdown(f"### Risk Level: **{risk_assessment['risk_level']}**")
            st.progress(risk_score)
            st.metric("Risk Score", f"{risk_score*100:.0f}/100")
        
        with col2:
            st.markdown("### Risk Factors Identified")
            
            if risk_assessment['risk_factors']:
                for idx, factor in enumerate(risk_assessment['risk_factors'], 1):
                    if 'high risk' in factor.lower():
                        st.error(f"{idx}. {factor}")
                    elif 'medium risk' in factor.lower():
                        st.warning(f"{idx}. {factor}")
                    else:
                        st.info(f"{idx}. {factor}")
            else:
                st.success("✅ No significant risk factors detected")
            
            st.markdown(f"**Recommendation:** {risk_assessment['recommendation']}")
    
    with pred_tab3:
        # Deep Insights
        insights = generate_deep_insights(articles_list, "")
        
        st.markdown("### 🧠 AI-Powered Deep Analysis")
        st.info(insights['summary'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚀 Opportunities")
            if insights['opportunities']:
                for opp in insights['opportunities'][:3]:
                    st.success(f"• {opp[:150]}...")
            else:
                st.caption("No specific opportunities identified")
        
        with col2:
            st.markdown("#### ⚠️ Risks to Monitor")
            if insights['risks']:
                for risk in insights['risks'][:3]:
                    st.warning(f"• {risk[:150]}...")
            else:
                st.caption("No specific risks identified")
        
        st.markdown("#### 💡 Key Predictions")
        for pred in insights['predictions']:
            st.markdown(f"• {pred}")
        
        st.markdown("#### 📋 Recommendations")
        for rec in insights['recommendations']:
            st.markdown(f"✓ {rec}")
    
    with pred_tab4:
        # Trend Analysis
        if st.session_state.sentiment_timeline:
            trend_analysis = analyze_sentiment_trend(st.session_state.sentiment_timeline)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Sentiment Trend Analysis")
                
                # Direction indicator
                if trend_analysis['direction'] == 'bullish':
                    st.success(f"📈 **Trend:** {trend_analysis['trend'].upper()} (Bullish)")
                elif trend_analysis['direction'] == 'bearish':
                    st.error(f"📉 **Trend:** {trend_analysis['trend'].upper()} (Bearish)")
                else:
                    st.info(f"➡️ **Trend:** {trend_analysis['trend'].upper()} (Neutral)")
                
                st.markdown(f"**Prediction:** {trend_analysis['prediction']}")
                
                # Trend strength
                st.progress(trend_analysis['strength'])
                st.caption(f"Trend Strength: {trend_analysis['strength']*100:.0f}%")
            
            with col2:
                # Safely get change_percent with fallback
                change_pct = trend_analysis.get('change_percent', 0.0)
                confidence = trend_analysis.get('confidence', 0.0)
                
                st.metric("Change", 
                         f"{change_pct:.1f}%",
                         delta=f"Confidence: {confidence*100:.0f}%")
                
                st.metric("Data Points", len(st.session_state.sentiment_timeline))
        else:
            st.info("📊 Sentiment trend analysis will appear after analyzing multiple articles over time")
    
    st.divider()

# ========================================
# CHAT INTERFACE
# ========================================

st.subheader("💬 Chat with EquityGPT")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Paste article URLs and ask your question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process message
    with st.chat_message("assistant"):
        # Extract URLs from prompt
        urls, cleaned_question = extract_urls_from_text(prompt)
        
        response_parts = []
        
        # Process new URLs
        if urls:
            with st.spinner(f"📥 Extracting {len(urls)} article(s)..."):
                extracted_count = 0
                for url in urls:
                    if url not in st.session_state.articles_cache:
                        article = extract_article(url)
                        if article:
                            st.session_state.articles_cache[url] = article
                            extracted_count += 1
                            
                            # Debug: Show first 100 chars of extracted text
                            text_preview = article.get('text', '')[:100]
                            garbled_check = sum(1 for c in text_preview if ord(c) > 1000)
                            
                            # Show success message
                            st.success(f"✅ **Loaded:** {article['title'][:60]}...")
                            
                            # Debug info
                            if garbled_check > 10:
                                st.error(f"⚠️ **DEBUG: Extracted text appears garbled!** ({garbled_check} weird chars in first 100)")
                                st.code(f"Preview: {text_preview}")
                            else:
                                st.info(f"✅ **DEBUG: Text looks clean!** Preview: {text_preview[:50]}...")
                            
                            # Auto sentiment analysis
                            if auto_sentiment:
                                sentiment = calculate_sentiment_score(article['text'])
                                st.info(f"🎯 **Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")
                                
                                # Track sentiment timeline
                                st.session_state.sentiment_timeline.append({
                                    'timestamp': datetime.now(),
                                    'article': article['title'],
                                    'sentiment': sentiment['score']
                                })
                
                if extracted_count > 0:
                    response_parts.append(f"📊 Successfully analyzed **{extracted_count}** new article(s).\n")
        
        # Build context from all cached articles
        # Convert articles_cache dict to list for AI processing
        articles_list = list(st.session_state.articles_cache.values())
        
        # Generate AI response using hybrid approach
        with st.spinner("🤔 Analyzing and generating insights..."):
            question = cleaned_question if cleaned_question and cleaned_question != '[ARTICLE]' else prompt
            
            # 🔮 CHECK FOR PREDICTIVE ANALYSIS COMMANDS
            predictive_commands = ['predict', 'forecast', 'analysis', 'deep analysis', 'risk', 'insights']
            is_predictive_request = any(cmd in question.lower() for cmd in predictive_commands)
            
            # If user asks for prediction/analysis, show comprehensive predictive report
            if is_predictive_request and articles_list:
                st.markdown("### 🔮 Comprehensive Predictive Analysis Report")
                
                # Calculate sentiment
                all_texts = [a['text'] for a in articles_list if a.get('text')]
                avg_sentiment = sum(calculate_sentiment_score(t)['score'] for t in all_texts) / len(all_texts) if all_texts else 0
                
                # Price prediction
                price_pred = predict_price_movement(articles_list, avg_sentiment)
                st.markdown(f"#### 📈 Price Movement Forecast")
                if price_pred['direction'] == 'upward':
                    st.success(f"**Direction:** Upward ({price_pred['probability']*100:.0f}% probability)")
                elif price_pred['direction'] == 'downward':
                    st.error(f"**Direction:** Downward ({price_pred['probability']*100:.0f}% probability)")
                else:
                    st.info(f"**Direction:** Neutral/Sideways")
                st.write(price_pred['reasoning'])
                
                # Risk assessment
                risk = assess_investment_risk(articles_list, avg_sentiment)
                st.markdown(f"#### ⚠️ Risk Assessment")
                st.write(f"**Risk Level:** {risk['risk_level']} ({risk['risk_score']*100:.0f}/100)")
                st.write(f"**Recommendation:** {risk['recommendation']}")
                
                # Deep insights
                insights = generate_deep_insights(articles_list, question)
                st.markdown(f"#### 💡 Key Insights")
                st.info(insights['summary'])
                
                if insights['opportunities']:
                    st.markdown("**🚀 Opportunities:**")
                    for opp in insights['opportunities'][:2]:
                        st.write(f"• {opp[:200]}...")
                
                if insights['risks']:
                    st.markdown("**⚠️ Risks:**")
                    for r in insights['risks'][:2]:
                        st.write(f"• {r[:200]}...")
                
                # Target price
                target = predict_target_price(articles_list)
                if target['predicted_target']:
                    st.markdown(f"#### 🎯 Price Target")
                    st.write(f"**Consensus Target:** ${target['predicted_target']} (Range: ${target['range_low']}-${target['range_high']})")
                
                ai_response = "✅ Comprehensive predictive analysis completed above. Use the dashboard tabs for more detailed visualizations."
                
            else:
                # 🧠 SMART CONTEXT ENHANCEMENT - Add conversation memory
                enhanced_question = question
                
                # 🧠 ADVANCED MULTI-ENTITY CONTEXT TRACKING
                import re
            
            # Extract ALL company names from question (not just one)
            company_patterns = [
                r'\b(Microsoft|MSFT|Tata Motors|Tata|Tesla|TSLA|Apple|AAPL|Amazon|AMZN|Google|GOOGL|Meta|META|Facebook|LSEG|London Stock Exchange|HDFC|ICICI|Reliance|Infosys|Wipro|TCS|Nvidia|NVDA|Netflix|NFLX|IBM|Oracle|SAP)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b(?=\s+(?:stock|share|price|target|earnings|investment))'
            ]
            
            current_companies = []
            for pattern in company_patterns:
                matches = re.finditer(pattern, question, re.IGNORECASE)
                for match in matches:
                    company = match.group(1)
                    if company not in current_companies and len(company) > 2:
                        current_companies.append(company)
            
            # Detect relationship/impact keywords
            relationship_keywords = ['impact', 'affect', 'helpful', 'benefit', 'influence', 'help', 'effect on', 'for', 'relationship', 'connect']
            has_relationship_query = any(keyword in question.lower() for keyword in relationship_keywords)
            
            # Detect topic
            current_topic = None
            if any(word in question.lower() for word in ['target', 'forecast', 'price', 'valuation']):
                current_topic = 'price_target'
            elif any(word in question.lower() for word in ['investment', 'invest', 'spending', 'capital']):
                current_topic = 'investment'
            elif any(word in question.lower() for word in ['earnings', 'revenue', 'profit', 'sales']):
                current_topic = 'earnings'
            elif any(word in question.lower() for word in ['why', 'reason', 'cause', 'analysis']):
                current_topic = 'analysis'
            
            # 🔍 SMART CONTEXT BUILDING
            enhanced_question = question
            context_info = []
            
            # Case 1: Multi-company relationship question (e.g., "How does Microsoft investment help LSEG?")
            if len(current_companies) >= 2 and has_relationship_query:
                # This is a cross-company analysis - use both companies
                enhanced_question = question  # Keep original, has all context
                context_info.append(f"🔗 Analyzing relationship between: {', '.join(current_companies)}")
                
                # Add context from conversation history
                recent_topics = st.session_state.conversation_context.get('topics_by_entity', {})
                for company in current_companies:
                    if company in recent_topics:
                        context_info.append(f"Previous context for {company}: {', '.join(recent_topics[company][-2:])}")
            
            # Case 2: Reference to previous topic ("these investments", "this impact", "it")
            elif has_relationship_query and not current_companies:
                # No company mentioned but asking about relationships
                # Use entities from recent conversation
                recent_entities = st.session_state.conversation_context.get('entities_discussed', [])
                last_topic = st.session_state.conversation_context.get('last_topic')
                
                if recent_entities:
                    # Build context from last 2-3 entities discussed
                    context_entities = recent_entities[-3:]
                    enhanced_question = f"{' '.join(context_entities)} {question}"
                    context_info.append(f"💡 Using context: {', '.join(context_entities)}")
            
            # Case 3: Single company with relationship keyword (e.g., "How helpful for LSEG?")
            elif len(current_companies) == 1 and has_relationship_query:
                # Asking about impact ON this company, use previous context
                target_company = current_companies[0]
                recent_entities = st.session_state.conversation_context.get('entities_discussed', [])
                recent_history = st.session_state.conversation_context.get('conversation_history', [])
                
                # Look for the subject from previous questions
                if recent_entities and recent_entities[-1] != target_company:
                    source_entity = recent_entities[-1]
                    last_topic = st.session_state.conversation_context.get('last_topic')
                    
                    # Build enhanced question with full context
                    if last_topic:
                        enhanced_question = f"How does {source_entity}'s {last_topic} {question}"
                    else:
                        enhanced_question = f"{source_entity} {question}"
                    
                    context_info.append(f"🔗 Connecting: {source_entity} → {target_company}")
            
            # Case 4: Follow-up without any company (pronoun references, short questions)
            elif not current_companies:
                is_follow_up = False
                
                # Multiple detection strategies
                follow_up_phrases = ['exact', 'specific', 'tell me', 'what is it', 'how much', 'give me', 'can you', 'show me']
                pronouns = ['it', 'this', 'that', 'its', 'the stock', 'the share', 'the company', 'the price', 'these', 'those']
                question_starters = ['what', 'which', 'when', 'where', 'who', 'how']
                
                word_count = len(question.split())
                
                if any(phrase in question.lower() for phrase in follow_up_phrases):
                    is_follow_up = True
                elif any(pronoun in question.lower() for pronoun in pronouns):
                    is_follow_up = True
                elif word_count < 6:
                    is_follow_up = True
                elif question.lower().split()[0] in question_starters and word_count < 8:
                    is_follow_up = True
                
                # Same topic continuation
                last_topic = st.session_state.conversation_context.get('last_topic')
                topic_keywords = {
                    'price_target': ['target', 'price', 'forecast', 'projection', 'valuation'],
                    'investment': ['investment', 'invest', 'spending', 'capital', 'plans'],
                    'earnings': ['earnings', 'revenue', 'profit', 'sales', 'quarterly'],
                    'analysis': ['why', 'reason', 'cause', 'impact', 'effect', 'analysis']
                }
                
                if last_topic and last_topic in topic_keywords:
                    if any(keyword in question.lower() for keyword in topic_keywords[last_topic]):
                        is_follow_up = True
                
                if is_follow_up:
                    last_company = st.session_state.conversation_context.get('last_company')
                    if last_company:
                        enhanced_question = f"{last_company} {question}"
                        context_info.append(f"💡 Continuing about: {last_company}")
            
            # Show context info to user
            if context_info:
                for info in context_info:
                    st.info(info)
            
            # 📝 UPDATE CONVERSATION TRACKING
            if current_companies:
                # Add to entities discussed (maintain order)
                for company in current_companies:
                    if company not in st.session_state.conversation_context['entities_discussed']:
                        st.session_state.conversation_context['entities_discussed'].append(company)
                    elif company != st.session_state.conversation_context['entities_discussed'][-1]:
                        # Move to end (most recent)
                        st.session_state.conversation_context['entities_discussed'].remove(company)
                        st.session_state.conversation_context['entities_discussed'].append(company)
                
                # Track topics by entity
                topics_by_entity = st.session_state.conversation_context.get('topics_by_entity', {})
                for company in current_companies:
                    if company not in topics_by_entity:
                        topics_by_entity[company] = []
                    if current_topic and current_topic not in topics_by_entity[company]:
                        topics_by_entity[company].append(current_topic)
                st.session_state.conversation_context['topics_by_entity'] = topics_by_entity
                
                # Update last company (for simple follow-ups)
                st.session_state.conversation_context['last_company'] = current_companies[-1]
            
            # Update last topic
            if current_topic:
                st.session_state.conversation_context['last_topic'] = current_topic
            
            # Save to conversation history
            st.session_state.conversation_context['conversation_history'].append({
                'question': question,
                'enhanced_question': enhanced_question,
                'companies': current_companies,
                'topic': current_topic
            })
            
            # Keep only last 10 for memory efficiency
            if len(st.session_state.conversation_context['conversation_history']) > 10:
                st.session_state.conversation_context['conversation_history'].pop(0)
            
            st.session_state.conversation_context['last_question'] = question
            
            # Use enhanced question for search
            search_question = enhanced_question
            
            if articles_list:
                handled = False  # track if we've produced a direct response
                # Summarization intent or URL-only message triggers direct summary
                question_for_detection = question.lower()
                
                # Check if question contains keywords from the loaded article title
                # This helps detect when user is asking about the specific article
                article_keywords_in_question = False
                most_recent_article = articles_list[-1]
                article_title = most_recent_article.get('title', '').lower()
                article_words = set([w for w in re.findall(r'\b\w{4,}\b', article_title) if w not in ['this', 'that', 'with', 'from', 'into', 'about']])
                question_words = set(re.findall(r'\b\w{4,}\b', question_for_detection))
                
                # If 2+ significant words from article title appear in question, treat as article-specific
                common_words = article_keywords_in_question = len(article_words & question_words) >= 2
                
                summarization_keywords = [
                    'summarize', 'summarise', 'summary', 'what is this about', 'what is this article',
                    'give me a summary', 'brief', 'overview', 'tell me about this', 'key points', 'tl;dr'
                ]
                is_summarization = any(k in question_for_detection for k in summarization_keywords)

                # URL-only detection: cleaned_question contains only [ARTICLE] tokens
                cq = cleaned_question.strip() if 'cleaned_question' in locals() else ''
                cq_sans_articles = re.sub(r"\[ARTICLE\]", "", cq).strip()
                url_only = (cq != '' and cq_sans_articles == '')

                if is_summarization or url_only:
                    try:
                        from app.ai_original import generate_article_summary
                    except Exception:
                        generate_article_summary = None

                    # Choose summary length from the question
                    if any(k in question_for_detection for k in ['short', 'brief', 'tl;dr']):
                        length = 'Short'
                    elif any(k in question_for_detection for k in ['detailed', 'long', 'in depth']):
                        length = 'Detailed'
                    else:
                        length = 'Medium'

                    selected = [articles_list[-1]] if 'all' not in question_for_detection else articles_list
                    parts = ["# 📄 Article Summary\n"]
                    for idx, art in enumerate(selected, 1):
                        title = art.get('title') or 'Untitled'
                        src = art.get('source') or urlparse(art.get('url', '')).netloc.replace('www.', '')
                        parts.append(f"### {title} ({src})\n")

                        if generate_article_summary:
                            try:
                                summary = generate_article_summary(art, length=length)
                            except Exception as e:
                                summary = f"⚠️ Failed to generate summary: {e}"
                        else:
                            text = (art.get('text') or '')
                            sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 50]
                            take = 3 if length == 'Short' else (5 if length == 'Medium' else 8)
                            summary = '. '.join(sentences[:take]) + ('.' if sentences[:take] else '')
                            if not summary:
                                summary = "⚠️ Article content is too short to generate a meaningful summary."

                        parts.append(summary + "\n")

                    ai_response = "\n".join(parts)
                    handled = True
                # Sentiment-intent: answer directly from loaded article(s) without web search
                if not handled:
                    sentiment_triggers = [
                        'positive or negative', 'positive or negetive', 'postive or negative', 'positive or negitive',
                        'positive', 'negative', 'negetive', 'negitive', 'postive', 'sentiment', 'bullish', 'bearish',
                        'good news', 'bad news', 'tone', 'optimistic', 'pessimistic', 'pos or neg'
                    ]
                    is_sentiment_question = any(k in question.lower() for k in sentiment_triggers)

                    if is_sentiment_question:
                        # Choose target articles: default to most recent one
                        target_articles = [articles_list[-1]]
                        # If user asks about "all" or "overall", compute both per-article and overall
                        if any(k in question.lower() for k in ['all', 'overall', 'combined', 'both']):
                            target_articles = articles_list

                        parts = ["### 🧭 Sentiment Assessment (Grounded in loaded article text)"]
                        per_scores = []
                        for art in target_articles:
                            text = art.get('text', '')
                            if not text:
                                continue
                            s = calculate_sentiment_score(text)
                            per_scores.append(s['score'])
                            label = s.get('label', 'Unknown')
                            src = art.get('source') or urlparse(art.get('url', '')).netloc.replace('www.', '')
                            title = art.get('title', 'Untitled')
                            parts.append(f"• {title} ({src}): **{label}** ({s['score']:.2f})")

                        if len(target_articles) > 1 and per_scores:
                            avg = sum(per_scores) / len(per_scores)
                            overall = 'Positive' if avg > 0.2 else 'Negative' if avg < -0.2 else 'Neutral'
                            parts.append(f"\n**Overall:** {overall} ({avg:.2f}) across {len(per_scores)} article(s)")

                        ai_response = "\n".join(parts)
                        handled = True

                if not handled:
                    # Try Q&A model first for specific questions
                    use_qa_fallback = False
                    qa_attempted = False
                    
                    try:
                        from transformers import pipeline
                        
                        if 'qa_model' not in st.session_state:
                            st.session_state.qa_model = pipeline(
                                "question-answering",
                                model="distilbert-base-cased-distilled-squad"
                            )
                        
                        qa_model = st.session_state.qa_model
                        
                        # Build context from articles
                        combined_context = " ".join([
                            article['text'][:1500] for article in articles_list
                        ])[:3000]  # Limit context
                        
                        # Get answer from the model
                        result = qa_model(question=question, context=combined_context)
                        answer = result['answer']
                        confidence = result['score']
                        qa_attempted = True
                        
                        # Check if answer quality is good enough
                        # If question contains article keywords, use lower threshold
                        confidence_threshold = 0.05 if article_keywords_in_question else 0.15
                        
                        if confidence < confidence_threshold or len(answer.split()) < 3:
                            # If question is clearly about the article, don't use web fallback
                            if article_keywords_in_question:
                                # Generate answer from article text directly
                                article_text = articles_list[-1].get('text', '')
                                # Find relevant sentences
                                sentences = [s.strip() for s in re.split(r'[.!?]\n', article_text) if len(s.strip()) > 30]
                                
                                # Score sentences by relevance
                                question_terms = set(question_for_detection.split())
                                scored_sentences = []
                                for sent in sentences[:50]:  # Check first 50 sentences
                                    sent_lower = sent.lower()
                                    score = sum(1 for term in question_terms if term in sent_lower and len(term) > 3)
                                    if score > 0:
                                        scored_sentences.append((score, sent))
                                
                                scored_sentences.sort(reverse=True, key=lambda x: x[0])
                                
                                if scored_sentences:
                                    top_sentences = [s[1] for s in scored_sentences[:3]]
                                    ai_response = f"**Based on the article:**\n\n"
                                    ai_response += " ".join(top_sentences)
                                    ai_response += f"\n\n**Source:** {articles_list[-1]['title']}"
                                    handled = True
                                else:
                                    # Use Q&A result anyway
                                    ai_response = f"**Answer from article:** {answer}\n\n"
                                    ai_response += f"**Source:** {articles_list[-1]['title']}"
                                    handled = True
                            else:
                                use_qa_fallback = True
                        else:
                            # Format response with context
                            ai_response = f"**Answer:** {answer}\n\n"
                            ai_response += f"*Confidence: {confidence:.1%}*\n\n"
                            
                            # Add article summary context
                            ai_response += f"**Based on {len(articles_list)} article(s):**\n"
                            for art in articles_list[:3]:
                                ai_response += f"• {art['title']} ({art['source']})\n"
                            handled = True
                            
                    except Exception as e:
                        use_qa_fallback = not article_keywords_in_question  # Don't fallback if question is about article
                        if article_keywords_in_question:
                            # Generate basic answer from article
                            article_text = articles_list[-1].get('text', '')
                            first_sentences = '. '.join([s.strip() for s in re.split(r'[.!?]\n', article_text) if len(s.strip()) > 30][:3])
                            ai_response = f"**Based on the article:**\n\n{first_sentences}\n\n**Source:** {articles_list[-1]['title']}"
                            handled = True
                    
                    # Fallback to comprehensive AI analysis if Q&A didn't work well
                    # BUT only if question is NOT clearly about the loaded article
                    if use_qa_fallback and not handled:
                        # Try web search first for better real-time data
                        tavily_key = os.getenv('TAVILY_API_KEY')
                        
                        if tavily_key and tavily_key.startswith('tvly-'):
                            try:
                                from app.ai_original import search_with_tavily
                                
                                # Use enhanced question with context for better search
                                search_results = search_with_tavily(search_question)
                                
                                if search_results:
                                    # Analyze question type for smart formatting
                                    question_lower = search_question.lower()
                                    
                                    # 1. Price/value questions - extract specific price
                                    if any(word in question_lower for word in ['price', 'cost', 'trading', 'worth', 'value', 'stock price']):
                                        import re
                                        price_content = ""
                                        for result in search_results[:2]:
                                            content = result.get('content', '')
                                            # Better price extraction patterns
                                            price_patterns = [
                                                r'current price[:\s]+\$?([\d,]+\.?\d*)',
                                                r'trading at[:\s]+\$?([\d,]+\.?\d*)',
                                                r'stock price[:\s]+\$?([\d,]+\.?\d*)',
                                                r'\$?([\d,]+\.?\d+)\s+per share'
                                            ]
                                            
                                            for pattern in price_patterns:
                                                match = re.search(pattern, content, re.IGNORECASE)
                                                if match:
                                                    price = match.group(1)
                                                    start = max(0, match.start() - 50)
                                                    end = min(len(content), match.end() + 100)
                                                    price_content = content[start:end].strip()
                                                    break
                                            
                                            if price_content:
                                                break
                                        
                                        if price_content:
                                            ai_response = f"**{question}**\n\n💰 {price_content}\n\n"
                                        else:
                                            ai_response = f"**{question}**\n\n{search_results[0].get('content', '')[:300]}\n\n"
                                        
                                        ai_response += f"**Source:** [{search_results[0].get('title', 'Latest Data')}]({search_results[0].get('url', '')})"
                                    
                                    # 2. Market analysis questions - comprehensive synthesis
                                    elif any(word in question_lower for word in ['market analysis', 'analysis', 'outlook', 'forecast', 'trends']):
                                        analysis_points = []
                                        
                                        for idx, result in enumerate(search_results[:3], 1):
                                            content = result.get('content', '')
                                            
                                            # Clean up content
                                            content = re.sub(r'^[^A-Z].*?\n', '', content)
                                            content = re.sub(r'\n[^A-Z].*?$', '', content)
                                            
                                            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 30]
                                            
                                            meaningful_content = []
                                            for sentence in sentences[:3]:
                                                noise_keywords = ['unlock', 'scorecard', 'highlights', 'metrics for', 'login', 'subscribe']
                                                if not any(noise in sentence.lower() for noise in noise_keywords):
                                                    if any(word in sentence.lower() for word in ['stock', 'price', 'market', 'analyst', 'growth', 'revenue', 'earnings', 'forecast', 'target']):
                                                        meaningful_content.append(sentence)
                                            
                                            if meaningful_content:
                                                key_point = '. '.join(meaningful_content[:2]).strip()
                                                if len(key_point) > 50:
                                                    key_point = re.sub(r'\s+', ' ', key_point)
                                                    analysis_points.append(f"**{idx}.** {key_point}.")
                                        
                                        ai_response = f"## {question}\n\n"
                                        
                                        if analysis_points:
                                            ai_response += "### 📊 Market Analysis:\n\n"
                                            ai_response += "\n\n".join(analysis_points)
                                            ai_response += f"\n\n*Based on {len(analysis_points)} insights + {len(articles_list)} loaded articles*"
                                        else:
                                            clean_content = search_results[0].get('content', '')
                                            paragraphs = clean_content.split('\n\n')
                                            for para in paragraphs:
                                                if len(para) > 100 and 'unlock' not in para.lower():
                                                    ai_response += para[:400]
                                                    break
                                        
                                        ai_response += "\n\n### 📚 Sources:\n"
                                        for result in search_results[:3]:
                                            ai_response += f"• [{result.get('title', 'Source')}]({result.get('url', '')})\n"
                                        
                                        ai_response += f"\n*Updated: {datetime.now().strftime('%B %d, %Y')}*"
                                    
                                    # 3. What/Who/Explain questions - brief definition
                                    elif any(word in question_lower for word in ['what is', 'who is', 'define', 'explain']):
                                        ai_response = f"**{question}**\n\n{search_results[0].get('content', '')[:300]}\n\n"
                                        ai_response += f"**Source:** [{search_results[0].get('title', 'Source')}]({search_results[0].get('url', '')})"
                                    
                                    # 4. Complex questions - smart summary
                                    else:
                                        ai_response = f"**{question}**\n\n"
                                        for idx, result in enumerate(search_results[:2], 1):
                                            content = result.get('content', '')[:250]
                                            if content:
                                                ai_response += f"**{idx}.** {content}\n\n"
                                        
                                        ai_response += "**Sources:**\n"
                                        for result in search_results[:2]:
                                            ai_response += f"• [{result.get('title', 'Source')}]({result.get('url', '')})\n"
                                else:
                                    # No web results, use AI analysis
                                    result = generate_realtime_ai_answer(
                                        question, 
                                        articles_list, 
                                        use_context=True,
                                        enable_web_search=False
                                    )
                                    ai_response = result[0] if isinstance(result, tuple) else result
                            except:
                                # Fallback to AI analysis
                                result = generate_realtime_ai_answer(
                                    question, 
                                    articles_list, 
                                    use_context=True,
                                    enable_web_search=False
                                )
                                ai_response = result[0] if isinstance(result, tuple) else result
                        else:
                            # No Tavily, use AI analysis
                            result = generate_realtime_ai_answer(
                                question, 
                                articles_list, 
                                use_context=True,
                                enable_web_search=False
                            )
                            ai_response = result[0] if isinstance(result, tuple) else result
            else:
                # No articles loaded - use web search silently to get latest data
                try:
                    # Check if web search is available (Tavily API key)
                    tavily_key = os.getenv('TAVILY_API_KEY')
                    
                    if tavily_key and tavily_key.startswith('tvly-'):
                        # Silently use web search for real-time data
                        from app.ai_original import search_with_tavily
                        
                        # Use enhanced question with context
                        search_results = search_with_tavily(search_question)
                        
                        if search_results:
                            # Analyze question type for smart formatting
                            question_lower = search_question.lower()
                            
                            # 🎯 ENHANCED ANSWER EXTRACTION
                            
                            # 1. Price questions - extract specific price
                            if any(word in question_lower for word in ['price', 'cost', 'trading at', 'worth', 'value', 'stock price']):
                                price_info = ""
                                for result in search_results[:2]:
                                    content = result.get('content', '')
                                    import re
                                    # Look for current price patterns
                                    price_patterns = [
                                        r'current price[:\s]+\$?([\d,]+\.?\d*)',
                                        r'trading at[:\s]+\$?([\d,]+\.?\d*)',
                                        r'stock price[:\s]+\$?([\d,]+\.?\d*)',
                                        r'\$?([\d,]+\.?\d+)\s+per share'
                                    ]
                                    
                                    for pattern in price_patterns:
                                        match = re.search(pattern, content, re.IGNORECASE)
                                        if match:
                                            price = match.group(1)
                                            # Extract surrounding context
                                            start = max(0, match.start() - 50)
                                            end = min(len(content), match.end() + 100)
                                            price_info = content[start:end].strip()
                                            break
                                    
                                    if price_info:
                                        break
                                
                                if price_info:
                                    ai_response = f"**{question}**\n\n💰 {price_info}\n\n"
                                else:
                                    # Fallback to first relevant content
                                    ai_response = f"**{question}**\n\n{search_results[0].get('content', '')[:300]}\n\n"
                                
                                ai_response += f"**Source:** [{search_results[0].get('title', 'Financial Data')}]({search_results[0].get('url', '')})\n"
                                ai_response += f"*Updated: {datetime.now().strftime('%B %d, %Y')}*"
                            
                            # 2. Market analysis questions - synthesize comprehensive answer
                            elif any(word in question_lower for word in ['market analysis', 'analysis', 'outlook', 'forecast', 'trends']):
                                # Combine insights from multiple sources with intelligent filtering
                                analysis_points = []
                                
                                for idx, result in enumerate(search_results[:3], 1):
                                    content = result.get('content', '')
                                    
                                    # Clean up the content - remove fragments and formatting issues
                                    # Remove partial lines that don't make sense
                                    content = re.sub(r'^[^A-Z].*?\n', '', content)  # Remove leading fragments
                                    content = re.sub(r'\n[^A-Z].*?$', '', content)  # Remove trailing fragments
                                    
                                    # Extract complete sentences
                                    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 30]
                                    
                                    # Take first 2-3 meaningful sentences
                                    meaningful_content = []
                                    for sentence in sentences[:3]:
                                        # Filter out noise (navigation, UI elements, etc.)
                                        noise_keywords = ['unlock', 'scorecard', 'highlights', 'metrics for', 'login', 'subscribe', 'read more', 'click here']
                                        if not any(noise in sentence.lower() for noise in noise_keywords):
                                            # Check if sentence is actually informative
                                            if any(word in sentence.lower() for word in ['stock', 'price', 'market', 'analyst', 'growth', 'revenue', 'earnings', 'forecast', 'target', 'expects', 'trading', 'valuation']):
                                                meaningful_content.append(sentence)
                                    
                                    if meaningful_content:
                                        # Join sentences into a coherent point
                                        key_point = '. '.join(meaningful_content[:2]).strip()
                                        if len(key_point) > 50:
                                            # Clean up any remaining formatting issues
                                            key_point = re.sub(r'\s+', ' ', key_point)  # Normalize whitespace
                                            analysis_points.append(f"**{idx}.** {key_point}.")
                                
                                ai_response = f"## {question}\n\n"
                                
                                if analysis_points:
                                    ai_response += "### 📊 Market Analysis:\n\n"
                                    ai_response += "\n\n".join(analysis_points)
                                    ai_response += f"\n\n*Based on {len(analysis_points)} key insights from real-time market data*"
                                else:
                                    # Fallback: use first clean result
                                    clean_content = search_results[0].get('content', '')
                                    # Remove noise and get first paragraph
                                    paragraphs = clean_content.split('\n\n')
                                    for para in paragraphs:
                                        if len(para) > 100 and 'unlock' not in para.lower():
                                            ai_response += para[:400]
                                            break
                                
                                ai_response += "\n\n### 📚 Sources:\n"
                                for result in search_results[:3]:
                                    ai_response += f"• [{result.get('title', 'Source')}]({result.get('url', '')})\n"
                                
                                ai_response += f"\n*Updated: {datetime.now().strftime('%B %d, %Y')}*"
                            
                            # 3. Definitional questions
                            elif any(word in question_lower for word in ['what is', 'who is', 'define', 'explain']):
                                ai_response = f"**{question}**\n\n{search_results[0].get('content', '')[:300]}\n\n"
                                ai_response += f"**Source:** [{search_results[0].get('title', 'Source')}]({search_results[0].get('url', '')})"
                            
                            # 4. General questions - smart summary
                            else:
                                # General question - show top 2 insights
                                ai_response = f"**{question}**\n\n"
                                for idx, result in enumerate(search_results[:2], 1):
                                    content = result.get('content', '')[:250]
                                    if content:
                                        ai_response += f"**{idx}.** {content}\n\n"
                                
                                ai_response += "**Sources:**\n"
                                for result in search_results[:2]:
                                    ai_response += f"• [{result.get('title', 'Source')}]({result.get('url', '')})\n"
                        else:
                            # No search results
                            ai_response = f"""**{question}**\n\nI couldn't find current information. Please try:

1. **Load specific articles** by pasting URLs
2. **Enable web search** - Add TAVILY_API_KEY to .env file
3. **Rephrase your question** for better results

💡 For real-time financial data, paste article URLs in your message."""
                    else:
                        # Tavily not configured - use basic AI
                        result = generate_realtime_ai_answer(
                            question, 
                            [], 
                            use_context=False,
                            enable_web_search=False
                        )
                        
                        if isinstance(result, tuple):
                            ai_response = result[0]
                        else:
                            ai_response = result
                        
                        # Add helpful tip
                        ai_response += f"""\n\n---
**💡 For Latest Data:**
• Paste article URLs in your message, or
• Add TAVILY_API_KEY to .env for real-time web search
• Example: `Analyze https://reuters.com/lseg-stock`"""
                        
                except Exception as e:
                    # Final fallback
                    ai_response = f"""**{question}**\n\n**To get accurate, up-to-date answers:**

📰 **Paste article URLs** directly in your message:
```
What's the analysis? https://reuters.com/your-article
```

🌐 **Enable real-time search:**
1. Get free API key from https://tavily.com
2. Add to .env: `TAVILY_API_KEY=tvly-your-key`
3. Ask any financial question!

📊 **Compare multiple sources:**
```
Compare https://bloomberg.com/article1 and https://wsj.com/article2
```"""
            
            response_parts.append(str(ai_response))
        
        # Combine response - ensure all parts are strings
        response_parts = [str(part) for part in response_parts]
        full_response = "\n\n".join(response_parts)
        
        # Display response
        st.markdown(full_response)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': prompt,
            'answer': full_response,
            'articles_count': len(st.session_state.articles_cache)
        })
        
        # Suggested follow-ups
        if st.session_state.articles_cache:
            st.markdown("---")
            st.markdown("**💡 You might also ask:**")
            cols = st.columns(3)
            
            suggestions = [
                "Compare these articles",
                "Extract key companies mentioned",
                "What's the overall market sentiment?",
            ]
            
            for col, suggestion in zip(cols, suggestions):
                col.button(suggestion, key=f"suggest_{suggestion}_{len(st.session_state.messages)}")

# ========================================
# FOOTER
# ========================================

st.divider()
st.markdown("""
<div style="text-align: center; opacity: 0.6; font-size: 12px;">
    <p>
        <strong>EquityGPT</strong> - Your Unique Financial Intelligence Platform<br>
        Real-time Analysis • Visual Insights • Expert Reports • Beyond Generic AI<br>
        Powered by Advanced NLP & Multi-Source Intelligence
    </p>
</div>
""", unsafe_allow_html=True)
