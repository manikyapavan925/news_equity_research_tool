#!/usr/bin/env python3

import sys
sys.path.append('.')
from streamlit_app import fetch_article_content, generate_article_summary

def test_reuters_article():
    # Test the Reuters URL
    url = 'https://www.reuters.com/world/uk/lseg-rolls-outs-blockchain-based-platform-private-funds-2025-09-15/'
    
    print("=== TESTING REUTERS ARTICLE FETCH ===")
    result = fetch_article_content(url)
    
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Success: {result.get('success', 'N/A')}")
    print(f"Content Length: {len(result.get('content', ''))}")
    print(f"Domain: {result.get('domain', 'N/A')}")
    
    content = result.get('content', '')
    if content:
        print(f"\nFirst 500 chars of content:")
        print(content[:500])
        print(f"\nLast 200 chars of content:")
        print(content[-200:])
        
        # Test summary generation
        print("\n=== TESTING SUMMARY GENERATION ===")
        summary = generate_article_summary(result, "Medium")
        print(f"Summary: {summary}")
        
        # Check if the summary contains relevant content
        title_lower = result.get('title', '').lower()
        summary_lower = summary.lower()
        
        print(f"\nTitle contains 'lseg': {'lseg' in title_lower}")
        print(f"Title contains 'blockchain': {'blockchain' in title_lower}")
        print(f"Summary contains 'lseg': {'lseg' in summary_lower}")
        print(f"Summary contains 'blockchain': {'blockchain' in summary_lower}")
    else:
        print("No content extracted!")
        if 'error_details' in result:
            print(f"Error: {result['error_details']}")

if __name__ == "__main__":
    test_reuters_article()