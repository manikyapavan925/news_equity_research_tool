#!/usr/bin/env python3

import os
import sys
import re
import unicodedata
import html

def clean_text_content(text):
    """Clean and normalize text content to remove junk characters and formatting issues"""
    if not text or not isinstance(text, str):
        return ""
    
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
    
    # Fix broken words (like "callsonMicrosoft" -> "calls on Microsoft")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    return text

def extract_main_topic(content):
    """Extract the main topic from content for better error messages"""
    content_lower = content.lower()
    if 'stock' in content_lower and 'decline' in content_lower:
        return "stock market performance and decline factors"
    elif 'financial' in content_lower or 'earnings' in content_lower:
        return "financial performance and earnings"
    elif 'investment' in content_lower:
        return "investment analysis"
    elif 'market' in content_lower:
        return "market conditions and trends"
    else:
        # Get first few meaningful words
        words = content.split()[:10]
        return f"topics including {' '.join(words[:5])}..."

def test_improved_llm():
    """Test the improved LLM response generation"""
    
    print("=== TESTING IMPROVED LLM IMPLEMENTATION ===")
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        
        print("Loading Flan-T5-base...")
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        qa_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=400,
            temperature=0.7,
            do_sample=True
        )
        
        # Test content (the Microsoft stock article)
        test_content = """Why Microsoft Stock Is Sinking Today Keith Noonan, The Motley Fool Sat, Sep 6, 2025, 1:06 AM 3 min read MSFT -2.55% ^SPX -0.32% ^IXIC -0.03% Key Points Microsoft stock is pulling back today in response to weak August jobs data. Soft jobs numbers in August suggest that the Federal Reserve will cut interest rates this month, but the U.S. economy is looking weaker than expected. New tariffs on semiconductors could also negatively impact Microsoft."""
        
        clean_context = clean_text_content(test_content)
        questions = [
            "What are Microsoft's AI plans?",
            "What are Microsoft's investment plans for 2026?",
            "Why is Microsoft stock declining?",  # This should work
            "What is Microsoft's target price?"
        ]
        
        for question in questions:
            print(f"\n--- Testing Question: {question} ---")
            
            # Use improved prompt format
            prompt = f"Answer this question based on the provided article content. If the article doesn't contain information to answer the question, clearly state that the information is not available in the article.\n\nQuestion: {question}\n\nArticle content: {clean_context}\n\nDetailed answer:"
            
            response = qa_pipeline(prompt, max_length=400, num_return_sequences=1, do_sample=True, temperature=0.3)
            answer = response[0]['generated_text'].strip()
            
            print(f"Raw LLM Answer: {answer}")
            
            # Apply the improved logic
            if len(answer) < 10 or answer.lower() in ['data', 'jobs data', 'stock', 'microsoft']:
                main_topic = extract_main_topic(clean_context)
                improved_answer = f"Based on the provided article content, I cannot find specific information about '{question}'. The article primarily discusses {main_topic}. To get information about your question, you may need articles that specifically cover that topic."
                print(f"Improved Answer: {improved_answer}")
            else:
                print(f"Final Answer: {answer}")
            
            print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_improved_llm()
