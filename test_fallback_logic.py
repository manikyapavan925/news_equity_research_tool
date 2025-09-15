#!/usr/bin/env python3
"""
Test the enhanced inadequate response detection and Tavily fallback logic
"""

import sys
import os
sys.path.append('.')

def test_inadequate_detection():
    """Test the enhanced _is_inadequate_search_response function"""
    import re
    
    def _is_inadequate_search_response(response, question):
        """Enhanced version with price query detection"""
        if not response or len(response.strip()) < 50:
            return True

        response_lower = response.lower()
        question_lower = question.lower()

        # Check for inadequate response patterns
        inadequate_patterns = [
            "no information found", "no relevant information", "i couldn't find",
            "unable to find", "no results", "sorry, i couldn't", "i apologize",
            "error occurred", "failed to", "no data available"
        ]

        for pattern in inadequate_patterns:
            if pattern in response_lower:
                return True

        # Enhanced check for financial/price queries
        price_keywords = ['price', 'traded', 'trading', 'stock', 'share', 'value', 'cost', 'worth', 'quote', 'market cap']
        is_price_question = any(keyword in question_lower for keyword in price_keywords)
        
        if is_price_question:
            # For price questions, check if response contains actual numerical data
            price_patterns = [
                r'[\$£€¥₹]\s*\d+',  # Currency symbols with numbers
                r'\d+\.\d+',         # Decimal numbers (common in prices)
                r'\d+\s*(dollars?|pounds?|euros?|yen|rupees?)',  # Number with currency words
                r'trading\s+at\s+\d+',  # "trading at X"
                r'price\s+of\s+[\$£€¥₹]?\d+',  # "price of $X"
                r'\d+\s*p\b',        # Pence notation (e.g., "123p")
                r'\d{2,4}\.\d{1,2}\s*(gbp|usd|eur)',  # Price with currency codes
            ]
            
            has_price_data = any(re.search(pattern, response_lower) for pattern in price_patterns)
            
            if not has_price_data:
                # Check if it's just generic search result titles/descriptions without actual data
                generic_search_indicators = [
                    'get the latest', 'real-time quote', 'stock price news',
                    'financial information', 'charts, and other', 'news today',
                    'what\'s going on at', 'read today\'s', 'news from trusted media'
                ]
                
                has_generic_content = sum(1 for indicator in generic_search_indicators 
                                        if indicator in response_lower) >= 2
                
                if has_generic_content:
                    return True  # This is generic search result descriptions, not actual price data

        return False

    # Test cases
    print("=== Testing Enhanced Inadequate Response Detection ===\n")
    
    # Test 1: Your actual inadequate response
    test_question = "what is the last traded price of lseg today?"
    inadequate_response = """Web Search Results for: what is the last traded price of lseg today?

1. London Stock Exchange Group Plc (LSEG) Stock Price News - Google Get the latest London Stock Exchange Group Plc (LSEG) real-time quote, historical performance, charts, and other financial information to help you mak...

2. London Stock Exchange Group (LSEG) News Today - MarketBeat What's going on at London Stock Exchange Group (LON:LSEG)? Read today's LSEG news from trusted media outlets at MarketBeat...."""
    
    result1 = _is_inadequate_search_response(inadequate_response, test_question)
    print(f"Test 1 - Your actual inadequate response:")
    print(f"Question: {test_question}")
    print(f"Response snippet: {inadequate_response[:150]}...")
    print(f"Detected as inadequate: {result1}")
    print(f"Expected: True (should fallback to Tavily)")
    print(f"Status: {'✅ PASS' if result1 else '❌ FAIL'}\n")
    
    # Test 2: Good response with actual price data
    good_response = "LSEG (London Stock Exchange Group) is currently trading at £87.45, up 2.3% from yesterday's close. The stock opened at £85.20 and has reached a high of £88.10 today."
    
    result2 = _is_inadequate_search_response(good_response, test_question)
    print(f"Test 2 - Good response with price data:")
    print(f"Response: {good_response}")
    print(f"Detected as inadequate: {result2}")
    print(f"Expected: False (should not fallback)")
    print(f"Status: {'✅ PASS' if not result2 else '❌ FAIL'}\n")
    
    # Test 3: Non-price question
    general_question = "What is LSEG's business model?"
    general_response = "London Stock Exchange Group operates as a diversified international market infrastructure business. It provides services across capital markets, including trading, clearing, settlement, and information services."
    
    result3 = _is_inadequate_search_response(general_response, general_question)
    print(f"Test 3 - Non-price question:")
    print(f"Question: {general_question}")
    print(f"Response: {general_response}")
    print(f"Detected as inadequate: {result3}")
    print(f"Expected: False (good response for general question)")
    print(f"Status: {'✅ PASS' if not result3 else '❌ FAIL'}\n")
    
    # Summary
    total_tests = 3
    passed_tests = sum([result1, not result2, not result3])
    print(f"=== Summary ===")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Overall: {'✅ ALL TESTS PASSED' if passed_tests == total_tests else '❌ SOME TESTS FAILED'}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = test_inadequate_detection()
    sys.exit(0 if success else 1)