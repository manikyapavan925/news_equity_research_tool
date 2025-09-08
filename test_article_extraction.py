#!/usr/bin/env python3

"""
Test the enhanced web search with article content extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_article_content_extraction():
    from streamlit_app import fetch_article_content, search_web_for_information, create_web_search_response
    
    print("🔍 Testing Enhanced Web Search with Article Content Extraction")
    print("=" * 70)
    
    # Test query
    question = "what is the target price of tata motors 2026"
    print(f"Query: {question}")
    print("-" * 50)
    
    # Get search results
    results = search_web_for_information(question, max_results=3)
    
    print(f"✅ Found {len(results)} results")
    print()
    
    # Test article content extraction for each result
    for i, result in enumerate(results, 1):
        print(f"{i}. Testing: {result['title']}")
        url = result['url']
        
        if url.startswith('http') and 'demo' not in result.get('source', '').lower():
            print(f"   Attempting to fetch content from: {url}")
            content = fetch_article_content(url, max_length=500)
            print(f"   Content preview: {content[:200]}...")
        else:
            print(f"   Demo/fallback data - using snippet instead")
        
        print()
    
    print("-" * 50)
    print("🎯 Testing Complete Response Generation")
    print("-" * 50)
    
    # Test the complete response creation with article content
    web_response = create_web_search_response(question, results)
    
    print("📝 Enhanced Response Preview:")
    print("-" * 30)
    print(web_response[:1000] + "..." if len(web_response) > 1000 else web_response)

if __name__ == "__main__":
    test_article_content_extraction()
