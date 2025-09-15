#!/usr/bin/env python3

import sys
sys.path.append('.')
from streamlit_app import generate_article_summary

def test_summary_lengths():
    # Test article content
    test_article = {
        'title': 'LSEG rolls outs blockchain-based platform for private funds',
        'content': '''The London Stock Exchange Group (LSEG) said on Monday that it has made its first transaction on a blockchain-based infrastructure platform it has launched for private funds as the data and analytics group expands its offerings. The Digital Markets Infrastructure platform, developed in partnership with Microsoft, enables the exchange group to offer new services to private funds. Private funds will be utilising the platform first, which will then be expanded to other assets, LSEG said. The platform aims to streamline operations and reduce costs for institutional investors. This represents a significant step forward in LSEG's digital transformation strategy. The blockchain technology provides enhanced security and transparency for fund transactions.'''
    }
    
    print("=== TESTING DIFFERENT SUMMARY LENGTHS ===\n")
    
    # Test Short summary
    short_summary = generate_article_summary(test_article, "Short")
    print(f"SHORT SUMMARY:")
    print(f"{short_summary}\n")
    
    # Test Medium summary  
    medium_summary = generate_article_summary(test_article, "Medium")
    print(f"MEDIUM SUMMARY:")
    print(f"{medium_summary}\n")
    
    # Test Detailed summary
    detailed_summary = generate_article_summary(test_article, "Detailed")
    print(f"DETAILED SUMMARY:")
    print(f"{detailed_summary}\n")
    
    print("=== COMPARISON ===")
    print(f"Short length: {len(short_summary)} characters")
    print(f"Medium length: {len(medium_summary)} characters") 
    print(f"Detailed length: {len(detailed_summary)} characters")

if __name__ == "__main__":
    test_summary_lengths()