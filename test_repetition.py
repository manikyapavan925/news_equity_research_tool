#!/usr/bin/env python3

import re

def is_repetitive_response(text):
    """Detect if the response is repetitive or stuck in a loop"""
    if not text or len(text) < 50:
        return False
    
    # Check for exact phrase repetition
    sentences = text.split('.')
    if len(sentences) > 3:
        # Check if the same sentence appears multiple times
        sentence_counts = {}
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only check meaningful sentences
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
                if sentence_counts[sentence] > 2:  # Same sentence repeated 3+ times
                    return True
    
    # Check for word repetition patterns
    words = text.split()
    if len(words) > 20:
        # Check if we have excessive repetition of the same phrases
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, it's likely repetitive
        max_word_count = max(word_counts.values()) if word_counts else 0
        if max_word_count > len(words) * 0.3:
            return True
    
    # Check for specific repetitive patterns
    repetitive_patterns = [
        "The purpose of the article is to provide",
        "The purpose of this article is to",
        "information about AI plans, technology strategies"
    ]
    
    for pattern in repetitive_patterns:
        if text.count(pattern) > 2:
            return True
    
    return False

def test_repetition_detection():
    """Test the repetition detection function"""
    
    print("=== TESTING REPETITION DETECTION ===")
    
    # Test case 1: Your actual repetitive response
    repetitive_text = """The purpose of the article is to provide information about AI plans, technology strategies, and future initiatives for Microsoft 2026. The purpose of this article is to provide information about AI plans, technology strategies, and future initiatives for Microsoft 2026. The purpose of this article is to provide information about AI plans, technology strategies, and future initiatives for Microsoft 2026. The purpose of this article is to provide information about AI plans, technology strategies, and future initiatives for Microsoft 2026. The purpose of this article is to provide information about AI plans, technology strategies, and future initiatives for Microsoft 2026."""
    
    print("Test 1: Your actual repetitive response")
    print(f"Is repetitive: {is_repetitive_response(repetitive_text)}")
    print(f"Text preview: {repetitive_text[:150]}...")
    print()
    
    # Test case 2: Normal good response
    normal_text = "Microsoft is focusing on AI integration across their cloud services. They plan to enhance Azure AI capabilities and integrate more AI features into Office 365. The company is investing heavily in OpenAI partnership."
    
    print("Test 2: Normal good response")
    print(f"Is repetitive: {is_repetitive_response(normal_text)}")
    print(f"Text: {normal_text}")
    print()
    
    # Test case 3: Short response
    short_text = "AI plans not available"
    
    print("Test 3: Short response")
    print(f"Is repetitive: {is_repetitive_response(short_text)}")
    print(f"Text: {short_text}")
    print()
    
    print("=== IMPROVED LLM PARAMETERS ===")
    print("✅ Lower temperature (0.1) for more focused responses")
    print("✅ Repetition penalty (1.5) to avoid loops")
    print("✅ Top-p sampling (0.9) for better variety")
    print("✅ Reduced max_length (300) to prevent long repetitions")
    print("✅ Early stopping enabled")
    print("✅ Repetition detection and fallback system")

if __name__ == "__main__":
    test_repetition_detection()
