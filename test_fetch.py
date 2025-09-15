#!/usr/bin/env python3
"""Test script for the improved fetch_article_content function"""

import requests
from bs4 import BeautifulSoup
import re
import time

def fetch_article_content(url, max_length=2000):
    """Fetch and extract main text content from a web article, filtering out dashboard and unrelated info"""
    try:
        import requests
        from bs4 import BeautifulSoup
        import time

        # Enhanced headers to mimic real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }

        # Try with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                response.raise_for_status()

                # Handle encoding
                if response.encoding is None or response.encoding == 'ISO-8859-1':
                    response.encoding = response.apparent_encoding or 'utf-8'

                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "svg"]):
                    element.decompose()

                # Enhanced content selectors for modern websites
                content_selectors = [
                    # Common article selectors
                    'article', '[data-testid="article-body"]', '[data-component="ArticleBody"]',
                    '.article-content', '.post-content', '.entry-content', '.content', '.main-content',
                    'main', '.article-body', '.post-body', '.story-body', '.article-text',
                    # Financial/news specific
                    '.contentSection', '.story-content', '.main-area', '.content-wrapper',
                    '.news-content', '.news-body', '.article__body', '.post__content',
                    # Generic content areas
                    '.container', '.wrapper', '.page-content', '.main-wrapper',
                    # Specific site selectors
                    '.entry', '.single-post', '.post-single', '.article-single',
                    # JSON-LD content (sometimes embedded)
                    'script[type="application/ld+json"]'
                ]

                content = ""
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        # For JSON-LD, try to extract articleBody
                        if 'script[type="application/ld+json"]' in selector:
                            for script in elements:
                                try:
                                    import json
                                    data = json.loads(script.string)
                                    if isinstance(data, dict) and 'articleBody' in data:
                                        content = data['articleBody']
                                        break
                                except:
                                    continue
                        else:
                            content = elements[0].get_text(separator=' ', strip=True)
                        if content and len(content.strip()) > 100:
                            break

                # If no specific content area found, try body text
                if not content or len(content.strip()) < 100:
                    body = soup.body
                    if body:
                        content = body.get_text(separator=' ', strip=True)
                    else:
                        content = soup.get_text(separator=' ', strip=True)

                # Clean up the text
                lines = [line.strip() for line in content.split('\n') if line.strip()]

                # Enhanced filtering for financial/news sites
                ignore_keywords = [
                    'dashboard', 'login', 'register', 'watchlist', 'trade', 'follow', 'add to',
                    'vol', 'risk', 'mint', 'bse', 'nse', 'subscribe', 'newsletter', 'advertisement',
                    'related articles', 'popular stories', 'trending', 'most read', 'comments',
                    'share this', 'print', 'email', 'facebook', 'twitter', 'linkedin',
                    'copyright', 'all rights reserved', 'terms of use', 'privacy policy'
                ]

                filtered_lines = []
                for line in lines:
                    line_lower = line.lower()
                    # Skip if contains ignore keywords or too short
                    if any(keyword in line_lower for keyword in ignore_keywords):
                        continue
                    if len(line) < 20:
                        continue
                    # Skip lines that are mostly non-alphabetic
                    alpha_ratio = len(re.findall(r'[a-zA-Z]', line)) / len(line) if line else 0
                    if alpha_ratio < 0.3:
                        continue
                    filtered_lines.append(line)

                content = ' '.join(filtered_lines)

                # Robust fallback: Extract from specific tags
                if len(content.strip()) < 100:
                    tag_texts = []
                    priority_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
                    for tag in priority_tags:
                        for element in soup.find_all(tag):
                            txt = element.get_text(separator=' ', strip=True)
                            if txt and len(txt) > 15 and not any(k in txt.lower() for k in ignore_keywords):
                                tag_texts.append(txt)

                    # Remove duplicates while preserving order
                    seen = set()
                    unique_texts = []
                    for text in tag_texts:
                        if text not in seen and len(text) > 15:
                            seen.add(text)
                            unique_texts.append(text)

                    content = ' '.join(unique_texts)

                # Final fallback: Extract all readable text
                if len(content.strip()) < 100:
                    all_text = soup.get_text(separator=' ', strip=True)
                    # Split into sentences and filter
                    sentences = re.split(r'[.!?]+', all_text)
                    meaningful_sentences = []
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 20 and len(re.findall(r'[a-zA-Z]', sentence)) > 10:
                            if not any(k in sentence.lower() for k in ignore_keywords):
                                meaningful_sentences.append(sentence)

                    content = '. '.join(meaningful_sentences[:20])  # Limit to first 20 sentences

                # Clean up whitespace
                content = re.sub(r'\s+', ' ', content).strip()

                # Limit content length
                if len(content) > max_length:
                    content = content[:max_length] + "..."

                # Get title with fallbacks
                title = None
                title_selectors = ['h1', 'title', '.headline', '.article-title', '.post-title', '[data-testid="headline"]']
                for selector in title_selectors:
                    if selector == 'title':
                        title_tag = soup.find('title')
                        if title_tag:
                            title = title_tag.get_text(strip=True)
                            break
                    else:
                        elements = soup.select(selector)
                        if elements:
                            title = elements[0].get_text(strip=True)
                            break

                if not title:
                    title = 'Article Content Extracted'

                # Clean title
                title = re.sub(r'\s+', ' ', title).strip()
                if len(title) > 200:
                    title = title[:200] + "..."

                return {
                    'url': url,
                    'title': title,
                    'content': content,
                    'success': True,
                    'domain': url.split('/')[2] if '://' in url else 'unknown',
                    'word_count': len(content.split()) if content else 0
                }

            except requests.exceptions.RequestException as req_error:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                error_msg = f"Network error: {str(req_error)}"
                break
            except Exception as parse_error:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                error_msg = f"Parsing error: {str(parse_error)}"
                break

        # If all retries failed
        return {
            'url': url,
            'title': 'Failed to Load',
            'content': f"Unable to fetch content after {max_retries} attempts. Error: {error_msg}",
            'success': False,
            'domain': 'unknown',
            'error_details': error_msg
        }

    except Exception as e:
        return {
            'url': url,
            'title': 'Failed to Load',
            'content': f"Unexpected error: {str(e)}",
            'success': False,
            'domain': 'unknown',
            'error_details': str(e)
        }

if __name__ == "__main__":
    # Test URLs
    test_urls = [
        'https://www.bbc.com/news',
        'https://www.reuters.com/',
        'https://www.cnn.com/',
    ]

    for url in test_urls:
        print(f"\n{'='*50}")
        print(f"Testing URL: {url}")
        print(f"{'='*50}")

        result = fetch_article_content(url, max_length=500)

        print(f"Success: {result.get('success', False)}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Content length: {len(result.get('content', ''))}")
        print(f"Word count: {result.get('word_count', 0)}")
        print(f"Domain: {result.get('domain', 'N/A')}")

        if not result.get('success', True):
            print(f"Error: {result.get('content', 'Unknown error')}")
        else:
            print(f"Content preview: {result.get('content', '')[:200]}...")