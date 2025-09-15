"""
Utils Module - Utility functions for text processing and validation
"""

import re
import unicodedata
import html
import random
import time
from urllib.parse import urlparse


def clean_text_content(text):
    """
    Clean and normalize text content to remove junk characters and formatting issues

    Args:
        text (str): Raw text content

    Returns:
        str: Cleaned text content
    """
    if not text or not isinstance(text, str):
        return ""

    try:
        # Remove control characters and normalize unicode
        text = unicodedata.normalize('NFKD', text)

        # Remove HTML entities and tags
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)

        # Fix common encoding issues
        text = text.replace('\xa0', ' ')  # Non-breaking space
        text = text.replace('\u2019', "'")  # Smart apostrophe
        text = text.replace('\u201c', '"').replace('\u201d', '"')  # Smart quotes
        text = text.replace('\u2013', '-').replace('\u2014', '-')  # Em and en dash

        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Remove garbled character sequences
        text = re.sub(r'[^\w\s.,;:!?()\-$%"\']+', ' ', text)

        return text

    except Exception as e:
        # Return original text if cleaning fails
        return str(text).strip()


def clean_and_format_text(text):
    """
    Clean and format text content (alias for clean_text_content)

    Args:
        text (str): Raw text content

    Returns:
        str: Cleaned and formatted text
    """
    return clean_text_content(text)


def is_valid_url(url):
    """
    Check if URL is valid

    Args:
        url (str): URL to validate

    Returns:
        bool: True if valid URL
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except:
        return False


def extract_domain_from_url(url):
    """
    Extract domain from URL

    Args:
        url (str): URL to extract domain from

    Returns:
        str: Domain name
    """
    if not url or not isinstance(url, str):
        return "unknown"

    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "unknown"


def get_random_user_agent():
    """
    Get a random user agent string

    Returns:
        str: Random user agent string
    """
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]
    return random.choice(user_agents)


def add_rate_limiting_delay():
    """
    Add a random delay to respect rate limiting
    """
    delay = random.uniform(1.0, 3.0)  # Random delay between 1-3 seconds
    time.sleep(delay)


def truncate_text(text, max_length=200, preserve_words=True):
    """
    Truncate text to maximum length

    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        preserve_words (bool): Whether to preserve word boundaries

    Returns:
        str: Truncated text
    """
    if not text or not isinstance(text, str):
        return ""

    if len(text) <= max_length:
        return text

    if preserve_words:
        # Try to truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # Only use word boundary if it's not too far back
            truncated = truncated[:last_space]

        return truncated + "..."
    else:
        return text[:max_length] + "..."


def calculate_text_similarity(text1, text2):
    """
    Calculate similarity between two texts

    Args:
        text1 (str): First text
        text2 (str): Second text

    Returns:
        float: Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0

    # Simple Jaccard similarity based on words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 1.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def extract_main_topic(text):
    """
    Extract main topic from text

    Args:
        text (str): Text to analyze

    Returns:
        str: Main topic
    """
    if not text:
        return "General"

    # Simple topic extraction based on keywords
    text_lower = text.lower()

    if any(word in text_lower for word in ['apple', 'iphone', 'mac', 'tim cook']):
        return "Apple Inc."
    elif any(word in text_lower for word in ['tesla', 'elon musk', 'electric vehicle']):
        return "Tesla"
    elif any(word in text_lower for word in ['tata motors', 'automotive', 'cars']):
        return "Tata Motors"
    elif any(word in text_lower for word in ['reliance', 'mukesh ambani']):
        return "Reliance Industries"
    else:
        return "Financial News"
