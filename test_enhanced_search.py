#!/usr/bin/env python3

"""
Test the enhanced web search functionality to ensure no duplicates
and better content quality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the enhanced search function
def test_enhanced_search():
    # Import the search function from streamlit_app
    from streamlit_app import search_web_for_information, create_web_search_response
    
    print("🔍 Testing Enhanced Web Search Function")
    print("=" * 60)
    
    # Test query
    question = "what are cars Tata Motors recently launched"
    print(f"Query: {question}")
    print("-" * 40)
    
    # Get search results
    results = search_web_for_information(question, max_results=5)
    
    print(f"✅ Found {len(results)} results")
    print()
    
    # Check for duplicates
    urls = [result['url'] for result in results]
    titles = [result['title'] for result in results]
    
    unique_urls = set(urls)
    unique_titles = set(titles)
    
    print(f"📊 Duplicate Analysis:")
    print(f"   Total URLs: {len(urls)}")
    print(f"   Unique URLs: {len(unique_urls)}")
    print(f"   Total Titles: {len(titles)}")
    print(f"   Unique Titles: {len(unique_titles)}")
    
    if len(urls) == len(unique_urls) and len(titles) == len(unique_titles):
        print("✅ No duplicates found!")
    else:
        print("❌ Duplicates detected!")
    
    print("\n" + "=" * 60)
    print("📋 Search Results:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Source: {result['source']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
    
    print("\n" + "=" * 60)
    print("🎯 Testing Create Web Search Response")
    print("=" * 60)
    
    # Test the complete response creation
    web_response = create_web_search_response(question, results)
    
    print("📝 Generated Response Preview:")
    print("-" * 40)
    print(web_response[:500] + "..." if len(web_response) > 500 else web_response)

if __name__ == "__main__":
    test_enhanced_search()
