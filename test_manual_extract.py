#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import re
import sys
sys.path.append('.')

def extract_reuters_content_manually():
    """Extract content using the specific patterns we found in debug"""
    
    url = 'https://www.reuters.com/world/uk/lseg-rolls-outs-blockchain-based-platform-private-funds-2025-09-15/'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        all_text = soup.get_text()
        
        # Based on debug analysis, we know these sentences exist in the HTML:
        target_sentences = [
            "The London Stock Exchange Group (LSEG) said on Monday that it has made its first transaction on a blockchain-based infrastructure platform it has launched for private funds as the data and analytics group expands its offerings",
            "The Digital Markets Infrastructure platform, developed in partnership with Microsoft",
            "Private funds will be utilising the platform first, which will then be expanded to other assets, LSEG said"
        ]
        
        # Extract all sentences that contain key LSEG/blockchain terms
        sentences = re.split(r'[.!?]+', all_text)
        relevant_sentences = []
        
        key_indicators = ['lseg', 'london stock exchange', 'blockchain', 'digital markets infrastructure', 
                         'platform', 'private funds', 'microsoft', 'transaction', 'partnership']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:  # Reasonable length
                sentence_lower = sentence.lower()
                
                # Check if sentence contains key terms
                if any(term in sentence_lower for term in key_indicators):
                    # Filter out navigation and unrelated content
                    if not any(exclude in sentence_lower for exclude in [
                        'skip to', 'browse', 'sign up', 'learn more', 'reuters next', 
                        'my news', 'federal reserve', 'james bullard', 'treasury secretary',
                        'scott bessent', 'central bank chair', 'netanyahu', 'hamas', 'israel'
                    ]):
                        # Additional check for sentence quality
                        if ('lseg' in sentence_lower or 'london stock exchange' in sentence_lower or 
                            'blockchain' in sentence_lower or 'digital markets' in sentence_lower):
                            relevant_sentences.append(sentence)
        
        # Remove duplicates while preserving order
        unique_sentences = []
        seen = set()
        for sentence in relevant_sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        content = '. '.join(unique_sentences)
        if content and not content.endswith('.'):
            content += '.'
            
        return {
            'title': 'LSEG rolls outs blockchain-based platform for private funds',
            'content': content,
            'success': True,
            'domain': 'www.reuters.com',
            'word_count': len(content.split()) if content else 0
        }
        
    except Exception as e:
        return {
            'title': 'Failed to Load',
            'content': f"Error: {str(e)}",
            'success': False,
            'domain': 'unknown',
            'error_details': str(e)
        }

if __name__ == "__main__":
    result = extract_reuters_content_manually()
    print("=== MANUAL REUTERS EXTRACTION ===")
    print(f"Title: {result['title']}")
    print(f"Success: {result['success']}")
    print(f"Content Length: {len(result.get('content', ''))}")
    print(f"Word Count: {result.get('word_count', 0)}")
    print(f"\nExtracted Content:")
    print(result.get('content', 'No content'))
    
    # Test summary generation
    if result.get('success'):
        from streamlit_app import generate_article_summary
        summary = generate_article_summary(result, "Medium")
        print(f"\n=== GENERATED SUMMARY ===")
        print(summary)