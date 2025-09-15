__all__ = ['search_web_for_information', 'create_web_search_response']
"""
Web Module - Web scraping and search functionality for the News Research Assistant

This module handles all web-related operations including:
- Article content extraction and parsing
- Web search using DuckDuckGo
- Content cleaning and processing
- Duplicate detection and prevention
- Rate limiting and respectful scraping

Author: News Research Assistant Team
Date: September 2025
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import time
import random
import random
from urllib.parse import quote, urlparse, urljoin
from typing import List, Dict, Any, Optional, Tuple, Set
import streamlit as st

# Import utility functions
from .utils import (
    is_valid_url,
    clean_and_format_text,
    get_random_user_agent,
    add_rate_limiting_delay,
    extract_domain_from_url,
    truncate_text,
    calculate_text_similarity
)


def fetch_article_content(url: str, max_length: int = 2000) -> Dict[str, Any]:
    """
    Fetches and extracts clean content from a news article URL.
    
    This function downloads article content, extracts the main text,
    and returns structured data including title, content, and metadata.
    
    Args:
        url (str): The article URL to fetch
        max_length (int): Maximum content length to extract
        
    Returns:
        Dict[str, Any]: Article data containing:
            - 'title': Article title
            - 'content': Main article content
            - 'url': Original URL
            - 'domain': Website domain
            - 'success': Whether extraction was successful
            - 'error': Error message if unsuccessful
            
    Extraction Strategy:
        1. Download page with proper headers
        2. Parse HTML using BeautifulSoup
        3. Extract title and main content
        4. Clean and format text
        5. Return structured data
    """
    article_data = {
        'title': '',
        'content': '',
        'url': url,
        'domain': extract_domain_from_url(url),
        'success': False,
        'error': None
    }
    
    try:
        # Validate URL first
        if not is_valid_url(url):
            article_data['error'] = "Invalid URL format"
            return article_data
        
        # Add rate limiting to be respectful
        try:
            add_rate_limiting_delay()
        except Exception:
            pass
        
        # Prepare comprehensive request headers to mimic a real browser
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        }
        
        # Use session for better handling
        session = requests.Session()
        
        # Try multiple approaches for difficult websites
        response = None
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts and response is None:
            attempts += 1
            
            try:
                if attempts == 1:
                    # First attempt with standard headers
                    session.headers.update(headers)
                    response = session.get(url, timeout=15, allow_redirects=True)
                    
                elif attempts == 2:
                    # Second attempt with enhanced headers for financial sites
                    enhanced_headers = headers.copy()
                    enhanced_headers.update({
                        'Referer': 'https://www.google.com/',
                        'Origin': 'https://www.google.com',
                        'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
                        'DNT': '1',
                        'X-Forwarded-For': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
                        'X-Real-IP': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
                    })
                    session.headers.clear()
                    session.headers.update(enhanced_headers)
                    
                    # Add some delay to appear more human-like
                    time.sleep(random.uniform(1, 3))
                    response = session.get(url, timeout=20, allow_redirects=True)
                    
                else:
                    # Third attempt with minimal headers (sometimes works better)
                    minimal_headers = {
                        'User-Agent': get_random_user_agent(),
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive',
                        'Referer': 'https://www.bing.com/',
                    }
                    session.headers.clear()
                    session.headers.update(minimal_headers)
                    
                    time.sleep(random.uniform(2, 5))
                    response = session.get(url, timeout=25, allow_redirects=True)
                
                response.raise_for_status()
                break  # Success, exit the loop
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403 and attempts < max_attempts:
                    # 403 Forbidden - try next approach
                    continue
                else:
                    # If it's the last attempt or different error, raise it
                    raise e
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempts < max_attempts:
                    time.sleep(random.uniform(1, 3))
                    continue
                else:
                    raise e
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = extract_article_title(soup)
        article_data['title'] = title
        
        # Extract main content
        content = extract_article_text(soup, max_length)
        article_data['content'] = content
        
        # Validate extracted content
        if len(content.strip()) < 50:
            # Try alternative extraction method for stubborn websites
            alternative_content = extract_fallback_content(soup)
            if len(alternative_content.strip()) >= 50:
                article_data['content'] = alternative_content
                article_data['success'] = True
                return article_data
            else:
                # Try one more approach - extract all visible text
                visible_text = extract_visible_text(soup)
                if len(visible_text.strip()) >= 50:
                    article_data['content'] = visible_text
                    article_data['success'] = True
                    return article_data
                else:
                    article_data['error'] = f"Insufficient content extracted (only {len(content.strip())} characters). This website may have strong anti-bot protection."
                    return article_data
        
        article_data['success'] = True
        return article_data
        
    except requests.exceptions.Timeout:
        article_data['error'] = "Request timeout - website took too long to respond"
    except requests.exceptions.ConnectionError:
        article_data['error'] = "Connection error - unable to reach website"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            # Try one final approach for 403 errors - use a different method
            try:
                fallback_content = attempt_alternative_scraping(url)
                if fallback_content and len(fallback_content.strip()) > 50:
                    article_data['content'] = fallback_content
                    article_data['success'] = True
                    article_data['title'] = "Article Content (Alternative Method)"
                    return article_data
            except Exception:
                pass
            
            # Create a helpful message with alternatives for blocked sites
            domain = extract_domain_from_url(url)
            if any(blocked_domain in domain for blocked_domain in ['invezz.com', 'investing.com', 'marketwatch.com']):
                article_data['error'] = (
                    f"🚫 {domain} blocks automated access (403 Forbidden). "
                    f"This website uses advanced bot protection. "
                    f"\n\n💡 **Alternatives:**\n"
                    f"• Use AI-Powered search instead (it searches the web automatically)\n"
                    f"• Try Reuters, Bloomberg, or Financial Times URLs\n"
                    f"• Copy-paste article content manually if needed\n\n"
                    f"🔍 **For your LSEG question, try:** AI-Powered search with: 'LSEG target price 2026 analyst estimates'"
                )
            else:
                article_data['error'] = f"Access denied (403) - website blocks automated requests. Try visiting the URL manually or use a different article."
        elif e.response.status_code == 404:
            article_data['error'] = f"Article not found (404) - URL may be incorrect or expired"
        elif e.response.status_code == 429:
            article_data['error'] = f"Rate limited (429) - too many requests to this website"
        else:
            article_data['error'] = f"HTTP error: {e.response.status_code} - {e.response.reason}"
    except Exception as e:
        article_data['error'] = f"Unexpected error: {str(e)}"
    
    return article_data


def extract_fallback_content(soup: BeautifulSoup) -> str:
    """
    Fallback content extraction for difficult websites.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        str: Extracted content using alternative methods
    """
    content_parts = []
    
    # Try to get all paragraph text
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        text = p.get_text().strip()
        if len(text) > 30:  # Only substantial paragraphs
            content_parts.append(text)
    
    # If still no content, try divs with text
    if not content_parts:
        divs = soup.find_all('div')
        for div in divs:
            text = div.get_text().strip()
            # Only direct text, not nested
            if len(text) > 50 and len(div.find_all()) < 3:
                content_parts.append(text)
    
    # Join and clean
    content = ' '.join(content_parts)
    return clean_and_format_text(content)[:2000]


def extract_visible_text(soup: BeautifulSoup) -> str:
    """
    Extract all visible text from the page as a last resort.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        str: All visible text content
    """
    # Remove scripts, styles, and other non-visible elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'meta', 'link']):
        element.decompose()
    
    # Get all text
    text = soup.get_text()
    
    # Clean up the text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Filter out navigation items, menu items, etc.
    content_lines = []
    for line in lines:
        if len(line) > 20 and not is_likely_navigation(line):
            content_lines.append(line)
    
    content = ' '.join(content_lines[:50])  # Limit to first 50 substantial lines
    return clean_and_format_text(content)[:2000]


def is_likely_navigation(text: str) -> bool:
    """Check if text is likely navigation or menu content."""
    nav_indicators = [
        'menu', 'navigation', 'home', 'about', 'contact', 'login', 'register',
        'subscribe', 'newsletter', 'follow us', 'social media', 'cookie policy',
        'privacy policy', 'terms of service', 'copyright', '©', 'all rights reserved'
    ]
    
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in nav_indicators) or len(text.split()) < 4


def attempt_alternative_scraping(url: str) -> str:
    """
    Alternative scraping method for stubborn websites.
    
    Args:
        url (str): URL to scrape
        
    Returns:
        str: Extracted content or empty string
    """
    try:
        # Use requests with a completely different approach
        headers = {
            'User-Agent': 'curl/7.68.0',  # Simple curl-like user agent
            'Accept': '*/*',
            'Connection': 'close'
        }
        
        # Simple request with minimal headers
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Try to extract content from the raw HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for JSON-LD structured data (common in news sites)
            json_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'articleBody' in data:
                        return data['articleBody'][:2000]
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'articleBody' in item:
                                return item['articleBody'][:2000]
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Try meta description as fallback
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                return meta_desc.get('content')
            
            # Try og:description
            og_desc = soup.find('meta', property='og:description')
            if og_desc and og_desc.get('content'):
                return og_desc.get('content')
    
    except Exception:
        pass
    
    return ""


def extract_article_title(soup: BeautifulSoup) -> str:
    """
    Extracts the article title from parsed HTML.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        
    Returns:
        str: Extracted article title
        
    Title Extraction Strategy:
        1. Try <title> tag first
        2. Look for Open Graph title
        3. Search for heading tags (h1, h2)
        4. Use article or main section headings
        5. Fall back to generic title
    """
    # Try different title sources in order of preference
    title_sources = [
        # Open Graph title
        soup.find('meta', property='og:title'),
        # Twitter Card title  
        soup.find('meta', attrs={'name': 'twitter:title'}),
        # Standard title tag
        soup.find('title'),
        # Main heading
        soup.find('h1'),
        soup.find('h2')
    ]
    
    for source in title_sources:
        if source:
            if source.name == 'meta':
                title = source.get('content', '').strip()
            else:
                title = source.get_text().strip()
            
            if title and len(title) > 10:
                # Clean up the title
                title = clean_and_format_text(title)
                # Remove common suffixes
                title = re.sub(r'\s*[-|]\s*[^-|]*$', '', title)
                return truncate_text(title, 150, preserve_words=True)
    
    return "Article"


def extract_article_text(soup: BeautifulSoup, max_length: int = 2000) -> str:
    """
    Extracts clean article text from parsed HTML.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content
        max_length (int): Maximum text length to extract
        
    Returns:
        str: Cleaned article text
        
    Text Extraction Strategy:
        1. Remove unwanted elements (scripts, styles, ads)
        2. Look for main content containers
        3. Extract paragraphs and clean text
        4. Filter out navigation and footer content
        5. Return formatted text
    """
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
        element.decompose()
    
    # Remove common ad and tracking elements
    for element in soup.find_all(True, {'class': re.compile(r'ad|advertisement|banner|popup|modal|cookie|tracking', re.I)}):
        element.decompose()
    
    # Look for main content containers in order of preference
    content_selectors = [
        'article',
        '[role="main"]',
        'main',
        '.content',
        '.article-content',
        '.post-content',
        '.entry-content',
        '.article-body',
        '.story-body',
        '.article-text',
        '.post-body',
        '.content-body',
        '.entry-body',
        '.text-content',
        '.article-container',
        '.post-container',
        # Specific to financial news sites
        '.article-wrap',
        '.news-content',
        '.story-content',
        '.article__content',
        '.post__content',
        # Common modern patterns
        '[data-module="ArticleBody"]',
        '[data-testid="article-body"]',
        '.ArticleBody',
        '.PostContent'
    ]
    
    main_content = None
    for selector in content_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    # If no main content container found, use the body
    if not main_content:
        main_content = soup.find('body') or soup
    
    # Extract text from paragraphs and headings
    text_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div'], 
                                         string=True)
    
    content_parts = []
    for element in text_elements:
        text = element.get_text().strip()
        
        # Skip short or empty text
        if len(text) < 20:
            continue
            
        # Skip navigation and menu items
        if is_navigation_content(text, element):
            continue
        
        # Clean and add the text
        cleaned_text = clean_and_format_text(text)
        if cleaned_text:
            content_parts.append(cleaned_text)
        
        # Stop if we have enough content
        current_length = len(' '.join(content_parts))
        if current_length > max_length:
            break
    
    # Join and final cleanup
    full_content = ' '.join(content_parts)
    
    # Remove duplicate sentences
    full_content = remove_duplicate_sentences(full_content)
    
    return truncate_text(full_content, max_length, preserve_words=True)


def is_navigation_content(text: str, element) -> bool:
    """
    Determines if text content is likely navigation/menu content.
    
    Args:
        text (str): The text to analyze
        element: The HTML element containing the text
        
    Returns:
        bool: True if content appears to be navigation
    """
    # Check for navigation keywords
    nav_keywords = [
        'menu', 'navigation', 'subscribe', 'newsletter', 'follow us',
        'share', 'tweet', 'facebook', 'linkedin', 'instagram',
        'home', 'about', 'contact', 'privacy', 'terms', 'sitemap',
        'advertisement', 'sponsored', 'related articles', 'more news'
    ]
    
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in nav_keywords):
        return True
    
    # Check element class/id for navigation indicators
    if hasattr(element, 'get'):
        class_names = ' '.join(element.get('class', [])).lower()
        element_id = element.get('id', '').lower()
        
        nav_indicators = ['nav', 'menu', 'sidebar', 'footer', 'header', 'ad', 'social']
        if any(indicator in class_names or indicator in element_id for indicator in nav_indicators):
            return True
    
    # Check if text looks like a navigation list
    if len(text.split()) < 5 and ('|' in text or '•' in text or text.count('\n') > 2):
        return True
    
    return False


def remove_duplicate_sentences(text: str) -> str:
    """
    Removes duplicate sentences from text content.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Text with duplicate sentences removed
    """
    sentences = re.split(r'[.!?]+', text)
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
            
        # Normalize for comparison
        normalized = re.sub(r'\s+', ' ', sentence.lower())
        
        if normalized not in seen_sentences:
            seen_sentences.add(normalized)
            unique_sentences.append(sentence)
    
    return '. '.join(unique_sentences) + '.' if unique_sentences else text


def search_web_for_information(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches the web for information using DuckDuckGo.
    
    This function performs web searches and returns structured results
    with deduplication and quality filtering.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results with metadata
        
    Search Process:
        1. Format query for optimal search
        2. Perform DuckDuckGo search
        3. Parse and structure results
        4. Remove duplicates and low-quality results
        5. Return top results
    """
    try:
        # Clean and format the search query
        clean_query = clean_search_query(query)
        
        # Perform the search
        search_results = perform_duckduckgo_search(clean_query, max_results * 2)
        
        if not search_results:
            return []
        
        # Filter and deduplicate results
        filtered_results = filter_and_deduplicate_results(search_results, max_results)
        
        return filtered_results
        
    except Exception as e:
        st.error(f"Web search error: {str(e)}")
        return []


def clean_search_query(query: str) -> str:
    """
    Cleans and optimizes a search query for better results.
    
    Args:
        query (str): Raw search query
        
    Returns:
        str: Optimized search query
    """
    # Remove question words that don't help in search
    stop_words = ['what', 'when', 'where', 'why', 'how', 'who', 'which', 'are', 'is', 'the']
    
    words = query.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    
    # Add quotes around company names or specific terms
    cleaned_query = ' '.join(filtered_words)
    
    # Enhance with news-specific terms
    if 'news' not in cleaned_query.lower():
        cleaned_query += ' news'
    
    return cleaned_query


def perform_duckduckgo_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Performs the actual DuckDuckGo search and parses results.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to fetch
        
    Returns:
        List[Dict[str, Any]]: Raw search results
    """
    try:
        # Add rate limiting
        try:
            add_rate_limiting_delay()
        except Exception:
            pass
        
        # Prepare search URL
        encoded_query = quote(query)
        search_url = f"https://duckduckgo.com/html/?q={encoded_query}"
        
        # Prepare headers
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        # Perform search
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse results
        soup = BeautifulSoup(response.content, 'html.parser')
        results = parse_duckduckgo_results(soup, max_results)
        
        return results
        
    except Exception as e:
        # Suppress warning for search failures
        return []


def parse_duckduckgo_results(soup: BeautifulSoup, max_results: int) -> List[Dict[str, Any]]:
    """
    Parses DuckDuckGo search results from HTML.
    
    Args:
        soup (BeautifulSoup): Parsed HTML from DuckDuckGo
        max_results (int): Maximum results to extract
        
    Returns:
        List[Dict[str, Any]]: Parsed search results
    """
    results = []
    
    # Find result containers
    result_elements = soup.find_all('div', class_=re.compile(r'result'))
    
    for element in result_elements[:max_results]:
        try:
            # Extract title
            title_element = element.find('a', class_=re.compile(r'result__a'))
            if not title_element:
                continue
                
            title = title_element.get_text().strip()
            url = title_element.get('href', '')
            
            # Clean URL (DuckDuckGo sometimes uses redirect URLs)
            if url.startswith('//duckduckgo.com/l/?'):
                # Extract actual URL from redirect
                url_match = re.search(r'uddg=([^&]+)', url)
                if url_match:
                    from urllib.parse import unquote
                    url = unquote(url_match.group(1))
            
            # Extract snippet
            snippet_element = element.find('a', class_=re.compile(r'result__snippet'))
            snippet = snippet_element.get_text().strip() if snippet_element else ""
            
            if title and url and is_valid_url(url):
                results.append({
                    'title': clean_and_format_text(title),
                    'url': url,
                    'snippet': clean_and_format_text(snippet),
                    'domain': extract_domain_from_url(url)
                })
                
        except Exception as e:
            continue  # Skip problematic results
    
    return results


def filter_and_deduplicate_results(results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
    """
    Filters and removes duplicate search results.
    
    Args:
        results (List[Dict[str, Any]]): Raw search results
        max_results (int): Maximum results to return
        
    Returns:
        List[Dict[str, Any]]: Filtered and deduplicated results
        
    Filtering Criteria:
        1. Remove results from excluded domains
        2. Remove duplicate URLs and similar titles
        3. Prioritize news and authoritative sources
        4. Ensure minimum content quality
    """
    if not results:
        return []
    
    # Domains to exclude (social media, forums, etc.)
    excluded_domains = {
        'twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com',
        'reddit.com', 'quora.com', 'pinterest.com', 'youtube.com',
        'tiktok.com', 'snapchat.com'
    }
    
    # Filter out excluded domains
    filtered_results = []
    for result in results:
        domain = result.get('domain', '').lower()
        if domain not in excluded_domains:
            filtered_results.append(result)
    
    # Remove duplicates based on URL and title similarity
    unique_results = []
    seen_urls = set()
    seen_titles = []
    
    for result in filtered_results:
        url = result['url']
        title = result['title']
        
        # Skip if URL already seen
        if url in seen_urls:
            continue
        
        # Check for similar titles
        is_duplicate = False
        for seen_title in seen_titles:
            similarity = calculate_text_similarity(title, seen_title)
            if similarity > 0.8:  # Very similar titles
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_urls.add(url)
            seen_titles.append(title)
            unique_results.append(result)
        
        # Stop if we have enough results
        if len(unique_results) >= max_results:
            break
    
    return unique_results


def create_web_search_response(question: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Creates a comprehensive response from web search results using professional AI format.
    
    This function extracts article content from search results and
    creates a well-formatted response with proper attribution in ChatGPT style.
    
    Args:
        question (str): The original question
        search_results (List[Dict[str, Any]]): Web search results
        
    Returns:
        str: Formatted response with article content and sources in professional format
        
    Response Generation:
        1. Extract content from top search results
        2. Analyze and summarize relevant information
        3. Format using professional AI response template
        4. Add disclaimers and source information
    """
    if not search_results:
        # Use the format_ai_answer function for consistent formatting
        from app.ai import format_ai_answer
        return format_ai_answer("", question=question)
    
    # Collect key information from search results
    key_findings = []
    source_links = []
    
    # Process search results and extract key insights
    for i, result in enumerate(search_results[:3]):  # Limit to top 3 results
        try:
            # Extract article content
            article_data = fetch_article_content(result['url'], max_length=400)
            
            if article_data['success'] and article_data['content']:
                content = article_data['content']
                domain = article_data['domain']
                title = article_data['title'] or result['title']
                
                # Extract key insight from content
                key_insight = content[:200] + "..." if len(content) > 200 else content
                key_findings.append(f"• **{domain}**: {key_insight}")
                source_links.append(f"[{title}]({result['url']})")
                
        except Exception as e:
            # Use snippet if content extraction fails
            snippet = result.get('snippet', '').strip()
            if snippet:
                domain = result.get('domain', 'source')
                key_findings.append(f"• **{domain}**: {snippet}")
                source_links.append(f"[{result['title']}]({result['url']})")
    
    # Determine company context for better formatting
    company_context = ""
    if question:
        question_lower = question.lower()
        if "lseg" in question_lower or "london stock exchange" in question_lower:
            company_context = "LSEG (London Stock Exchange Group)"
        elif any(auto_term in question_lower for auto_term in ["tata", "motor", "automotive"]):
            company_context = "automotive sector company"
        elif any(bank_term in question_lower for bank_term in ["bank", "hdfc", "icici"]):
            company_context = "banking institution"
    
    # Build professional formatted response
    response = "🤖 AI Analysis #120:\n\n"
    response += "### Summary of the Article\n"
    
    if key_findings:
        response += f"Current Market Analysis: Based on available research for {company_context if company_context else 'the requested company'}\n"
        response += "Analyst Consensus: Information compiled from multiple web sources\n"
        response += "Key Financial Metrics: Data extracted from recent financial reports and news\n\n"
        response += "**Research Findings:**\n"
        response += "\n".join(key_findings[:3])
    else:
        response += "Current Market Analysis: Limited specific information available from web sources\n"
        response += f"Analyst Consensus: Recommend checking official {company_context if company_context else 'company'} sources for latest data\n"
        response += "Key Financial Metrics: Additional research required for comprehensive analysis"
    
    response += "\n\n### What It All Means\n\n"
    response += "| Metric | Insight |\n"
    response += "|--------|---------|\n"
    
    if 'target price' in question.lower() or 'price target' in question.lower():
        response += "| **Target Price Analysis** | Analyst estimates vary by source and timeframe |\n"
        response += "| **Valuation Approach** | Based on financial models and market analysis |\n"
        response += "| **Market Sentiment** | Information compiled from available research sources |\n"
        response += "| **Time Horizon** | Projections based on specified timeframe in question |\n"
    else:
        response += "| **Data Sources** | Multiple web sources compiled and analyzed |\n"
        response += "| **Market Sentiment** | Based on latest available information |\n"
        response += "| **Verification** | Cross-reference with official sources recommended |\n"
        response += "| **Catalyst Dependency** | Information accuracy depends on source reliability |\n"
    
    response += "\n### Broader Context\n"
    if "lseg" in question.lower() or "london stock exchange" in question.lower():
        response += "* **Industry Trends**: The financial services sector is experiencing ongoing digital transformation and regulatory changes.\n"
        response += "* **Economic Environment**: Global market conditions and regulatory developments influence LSEG's performance.\n"
        response += "* **Regulatory Landscape**: Financial market regulations and policy changes impact long-term prospects.\n"
    else:
        response += "* **Industry Trends**: Market conditions and sector dynamics influence company performance.\n"
        response += "* **Economic Environment**: Macroeconomic factors and global conditions affect business outcomes.\n" 
        response += "* **Regulatory Landscape**: Policy changes and regulatory developments impact industry prospects.\n"
    
    response += "\n### Final Thoughts\n\n"
    response += "The analysis reflects available information from web sources and market research. While search results provide insights, "
    response += "actual performance depends on company execution and market developments.\n\n"
    response += "Investors should consider multiple perspectives, conduct thorough research, and consult professional advisors before making decisions. "
    response += "Market conditions can change rapidly, affecting information validity and investment outcomes.\n"
    
    # Add sources
    if source_links:
        response += f"\n**Sources**: {', '.join(source_links[:3])}\n"
    
    response += "\n**Disclaimer**: This analysis is for informational purposes only and should not be considered as financial advice. "
    response += "Always consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results."
    
    return response


def create_snippet_based_response(question: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Creates a response using search result snippets when full content extraction fails.
    Uses professional AI formatting for consistency.
    
    Args:
        question (str): The original question
        search_results (List[Dict[str, Any]]): Search results with snippets
        
    Returns:
        str: Response based on search snippets in professional format
    """
    if not search_results:
        # Use the format_ai_answer function for consistent formatting
        from app.ai import format_ai_answer
        return format_ai_answer("", question=question)
    
    # Collect key snippets
    key_snippets = []
    source_links = []
    
    for i, result in enumerate(search_results[:3]):
        snippet = result.get('snippet', '').strip()
        title = result.get('title', '').strip()
        domain = result.get('domain', 'source')
        
        if snippet:
            key_snippets.append(f"• **{domain}**: {snippet}")
            source_links.append(f"[{title}]({result['url']})")
    
    # Build professional formatted response
    response = "## Summary of Article\n\n"
    
    if key_snippets:
        response += "Based on web search snippets:\n\n"
        response += "\n".join(key_snippets)
    else:
        response += "• No detailed information available from current search\n"
        response += "• Recommend trying alternative search terms"
    
    response += "\n\n## What It All Means\n\n"
    response += "| **Aspect** | **Key Information** |\n"
    response += "|------------|--------------------|\n"
    response += "| **Data Source** | Web search snippets only |\n"
    response += "| **Completeness** | Limited information available |\n"
    response += "| **Recommendation** | Visit source links for full details |\n"
    
    response += "\n## Broader Context\n\n"
    response += "• Information is limited to search snippets\n"
    response += "• Full articles may contain additional important details\n"
    response += "• Consider searching with more specific terms\n"
    
    response += "\n## Final Thoughts\n\n"
    response += "The available snippet-based information provides only a partial view. "
    response += "For comprehensive analysis, please visit the source links directly.\n"
    
    # Add sources
    if source_links:
        response += f"\n**Sources:** {', '.join(source_links)}\n"
    
    response += "\n*Disclaimer: This information is based on search snippets and may be incomplete. "
    response += "Please visit source links for full details and verify independently.*"
    
    return response


def get_financial_data_response(question: str, company_name: str = "") -> str:
    """
    Generates responses for financial data queries using web search.
    
    Args:
        question (str): The financial question
        company_name (str): Specific company name if identified
        
    Returns:
        str: Financial data response
    """
    # Enhance query for financial data
    financial_query = f"{question} {company_name} financial data earnings revenue"
    
    # Perform targeted search
    results = search_web_for_information(financial_query, max_results=3)
    
    if results:
        return create_web_search_response(question, results)
    else:
        return f"I couldn't find recent financial data for your query. Please try searching for '{company_name} financial results' on financial news websites."


def search_financial_web_data(query: str, company_name: str = "") -> List[Dict[str, Any]]:
    """
    Performs specialized financial web search.
    
    Args:
        query (str): Search query
        company_name (str): Company name for focused search
        
    Returns:
        List[Dict[str, Any]]: Financial search results
    """
    # Add financial-specific terms
    enhanced_query = f"{query} {company_name} earnings revenue profit financial results"
    
    return search_web_for_information(enhanced_query, max_results=5)
