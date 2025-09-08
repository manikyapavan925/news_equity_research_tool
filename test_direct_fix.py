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

def analyze_question_type(question):
    """Analyze the question to understand what type of information is being requested"""
    question_lower = question.lower()
    
    # Question type patterns
    question_types = {
        'plans': ['plan', 'plans', 'strategy', 'roadmap', 'future', 'initiative', 'upcoming'],
        'financial': ['price', 'target price', 'revenue', 'profit', 'earnings', 'financial', 'valuation'],
        'ai_tech': ['ai', 'artificial intelligence', 'technology', 'innovation', 'digital'],
        'performance': ['performance', 'growth', 'decline', 'increase', 'decrease', 'change'],
        'investment': ['investment', 'invest', 'funding', 'capital', 'spending'],
        'market': ['market', 'competition', 'industry', 'sector'],
        'general': ['what', 'why', 'how', 'when', 'where']
    }
    
    detected_types = []
    for q_type, keywords in question_types.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_types.append(q_type)
    
    return detected_types if detected_types else ['general']

def test_direct_llm_fix():
    """Test a direct, simple approach to fix the LLM issue"""
    
    print("=== TESTING DIRECT LLM FIX ===")
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        
        print("Loading Flan-T5-base...")
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        qa_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer
        )
        
        # Test content
        test_content = """Microsoft stock is declining due to weak jobs data and tariff concerns. The Federal Reserve may cut interest rates. New tariffs on semiconductors could negatively impact Microsoft."""
        clean_context = clean_text_content(test_content)
        
        question = "What are AI strategies for Microsoft 2026"
        question_types = analyze_question_type(question)
        
        print(f"Question: {question}")
        print(f"Question types: {question_types}")
        print(f"Context: {clean_context}")
        
        # Test the best approach
        print(f"\n=== RECOMMENDED SOLUTION ===")
        best_prompt = f"""Answer this question about Microsoft based on the article. If the article doesn't contain information about AI strategies, clearly state what it does discuss instead.

Question: {question}

Article: {clean_context}

Clear answer:"""
        
        print(f"Best prompt: {best_prompt}")
        
        response = qa_pipeline(
            best_prompt,
            max_length=150,
            min_length=25,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.5,
            early_stopping=True
        )
        
        final_answer = response[0]['generated_text'].strip()
        print(f"Final answer: {final_answer}")
        
        # Test a specific fallback approach
        if len(final_answer) < 30:
            print("\n=== USING FALLBACK APPROACH ===")
            fallback_answer = f"""This article does not contain information about Microsoft's AI strategies for 2026. 

The article discusses: {clean_context[:100]}...

For Microsoft's AI strategies, you would need to check:
- Microsoft's official AI announcements
- Investor presentations and earnings calls
- Microsoft Build conference updates
- Azure AI service documentation"""
            
            print(f"Fallback answer: {fallback_answer}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_direct_llm_fix()
