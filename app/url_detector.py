"""
URL Detection and Extraction Module
Automatically detects and extracts URLs from user chat messages
"""

import re
from typing import List, Tuple, Dict
from urllib.parse import urlparse


def extract_urls_and_question(user_input: str) -> Tuple[List[str], str]:
    """
    Extract URLs from user input and return cleaned question.
    
    Examples:
        "What's the sentiment of https://reuters.com/article" 
        → (["https://reuters.com/article"], "What's the sentiment of [ARTICLE]")
        
        "Compare https://url1 and https://url2"
        → (["https://url1", "https://url2"], "Compare [ARTICLE] and [ARTICLE]")
    
    Args:
        user_input: Raw user message
        
    Returns:
        (urls, cleaned_question)
    """
    # Comprehensive URL pattern
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    # Find all URLs
    urls = re.findall(url_pattern, user_input)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    # Replace URLs with placeholder in question
    cleaned_question = user_input
    for url in urls:
        cleaned_question = cleaned_question.replace(url, '[ARTICLE]')
    
    # Clean up multiple spaces and placeholders
    cleaned_question = re.sub(r'\s+', ' ', cleaned_question).strip()
    cleaned_question = re.sub(r'\[ARTICLE\]\s*,?\s*and\s*\[ARTICLE\]', 'these articles', cleaned_question)
    cleaned_question = re.sub(r'of\s*\[ARTICLE\]', 'of this article', cleaned_question)
    
    return unique_urls, cleaned_question


def validate_url(url: str) -> bool:
    """
    Validate if a URL is properly formed and from a news source.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def categorize_url(url: str) -> Dict[str, str]:
    """
    Categorize URL by source type.
    
    Args:
        url: URL to categorize
        
    Returns:
        dict with 'type' and 'source' keys
    """
    domain = urlparse(url).netloc.lower().replace('www.', '')
    
    # Financial news sources
    financial_sources = {
        'reuters.com': {'type': 'financial_news', 'name': 'Reuters'},
        'bloomberg.com': {'type': 'financial_news', 'name': 'Bloomberg'},
        'wsj.com': {'type': 'financial_news', 'name': 'Wall Street Journal'},
        'ft.com': {'type': 'financial_news', 'name': 'Financial Times'},
        'cnbc.com': {'type': 'financial_news', 'name': 'CNBC'},
        'marketwatch.com': {'type': 'financial_news', 'name': 'MarketWatch'},
        'seekingalpha.com': {'type': 'financial_analysis', 'name': 'Seeking Alpha'},
        'fool.com': {'type': 'financial_analysis', 'name': 'Motley Fool'},
        'benzinga.com': {'type': 'financial_news', 'name': 'Benzinga'},
        'barrons.com': {'type': 'financial_news', 'name': "Barron's"},
        'economist.com': {'type': 'financial_news', 'name': 'The Economist'},
        'moneycontrol.com': {'type': 'financial_news', 'name': 'MoneyControl'},
        'livemint.com': {'type': 'financial_news', 'name': 'Mint'},
        'economictimes.indiatimes.com': {'type': 'financial_news', 'name': 'Economic Times'},
    }
    
    # General news sources
    general_sources = {
        'cnn.com': {'type': 'general_news', 'name': 'CNN'},
        'bbc.com': {'type': 'general_news', 'name': 'BBC'},
        'nytimes.com': {'type': 'general_news', 'name': 'New York Times'},
        'theguardian.com': {'type': 'general_news', 'name': 'The Guardian'},
        'washingtonpost.com': {'type': 'general_news', 'name': 'Washington Post'},
    }
    
    # Check financial sources first
    for source_domain, info in financial_sources.items():
        if source_domain in domain:
            return {
                'type': info['type'],
                'source': info['name'],
                'domain': domain,
                'priority': 'high'  # Financial sources get priority
            }
    
    # Check general sources
    for source_domain, info in general_sources.items():
        if source_domain in domain:
            return {
                'type': info['type'],
                'source': info['name'],
                'domain': domain,
                'priority': 'medium'
            }
    
    # Unknown source
    return {
        'type': 'unknown',
        'source': domain,
        'domain': domain,
        'priority': 'low'
    }


def suggest_related_urls(url: str) -> List[str]:
    """
    Suggest related URLs based on the input URL.
    
    Args:
        url: Original URL
        
    Returns:
        List of suggested related URLs
    """
    category = categorize_url(url)
    source = category['source']
    
    # This is a placeholder - in production, you might use an API
    # or web scraping to find actual related articles
    suggestions = []
    
    if category['type'] == 'financial_news':
        suggestions = [
            f"Related articles on {source}",
            "Analyst reports on the same topic",
            "Market reaction coverage"
        ]
    
    return suggestions


def extract_article_metadata(url: str) -> Dict:
    """
    Extract metadata from URL structure.
    
    Args:
        url: URL to analyze
        
    Returns:
        dict with metadata
    """
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p]
    
    metadata = {
        'url': url,
        'domain': parsed.netloc.replace('www.', ''),
        'path_parts': path_parts,
        'has_date': False,
        'possible_date': None,
        'slug': path_parts[-1] if path_parts else None
    }
    
    # Try to extract date from URL
    date_patterns = [
        r'(\d{4})/(\d{2})/(\d{2})',  # YYYY/MM/DD
        r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, parsed.path)
        if match:
            metadata['has_date'] = True
            metadata['possible_date'] = '-'.join(match.groups())
            break
    
    return metadata


def batch_validate_urls(urls: List[str]) -> List[Dict]:
    """
    Validate and categorize multiple URLs.
    
    Args:
        urls: List of URLs to process
        
    Returns:
        List of dicts with validation results
    """
    results = []
    
    for url in urls:
        result = {
            'url': url,
            'is_valid': validate_url(url),
            'category': categorize_url(url),
            'metadata': extract_article_metadata(url)
        }
        results.append(result)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_inputs = [
        "What's the sentiment of https://reuters.com/markets/tesla-stock",
        "Compare https://bloomberg.com/news/article1 and https://wsj.com/article2",
        "Analyze this: https://seekingalpha.com/article/apple-earnings",
        "https://moneycontrol.com/news/india-market what do you think?",
    ]
    
    print("URL Extraction Tests:")
    print("=" * 50)
    
    for test_input in test_inputs:
        urls, question = extract_urls_and_question(test_input)
        print(f"\nInput: {test_input}")
        print(f"URLs: {urls}")
        print(f"Question: {question}")
        
        if urls:
            for url in urls:
                category = categorize_url(url)
                print(f"  → {category['source']} ({category['type']}) - Priority: {category['priority']}")
