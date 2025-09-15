"""
Utils Module - Basic Implementation
This module provides utility functions for the News Equity Research Tool.
"""

def clean_text_content(text):
    """
    Clean and normalize text content
    """
    if not text:
        return ""

    # Basic text cleaning
    import re
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()

    return text

def extract_key_insights(text):
    """
    Extract key insights from text
    """
    if not text:
        return []

    # Basic insight extraction
    sentences = text.split('.')
    insights = [s.strip() for s in sentences if len(s.strip()) > 20][:5]

    return insights</content>
<parameter name="filePath">/Users/nethimanikyapavan/Documents/augment-projects/News_Equity_Research_Tool/app/utils_original.py
