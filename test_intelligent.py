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

def create_smart_prompt(question, context, question_types):
    """Create an intelligent prompt based on question type and context relevance"""
    
    # Check if context is relevant to the question
    question_keywords = set(question.lower().split())
    context_words = set(context.lower().split())
    
    # Remove common stop words for better matching
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    meaningful_q_words = question_keywords - stop_words
    
    # Check for keyword overlap
    keyword_overlap = len(meaningful_q_words.intersection(context_words)) / max(len(meaningful_q_words), 1)
    
    # Create context-aware prompts
    if 'ai_tech' in question_types or 'plans' in question_types:
        if keyword_overlap < 0.2:  # Low relevance
            return f"Read this article carefully and answer: {question}\n\nIf the article doesn't discuss AI, technology plans, or future initiatives, clearly state: 'This article does not contain information about {question.lower()}. The article focuses on [main topic].'\n\nArticle: {context}\n\nAnswer:"
        else:
            return f"Based on this article about technology and business plans, answer: {question}\n\nArticle: {context}\n\nDetailed answer:"
    
    elif 'financial' in question_types:
        if 'price' in question.lower() and 'target' not in context.lower() and 'price' not in context.lower():
            return f"Read this article and answer: {question}\n\nIf the article doesn't mention specific price targets or valuations, respond: 'This article does not provide price targets or specific valuations for {question.lower()}. The article discusses [what it actually covers].'\n\nArticle: {context}\n\nAnswer:"
        else:
            return f"Based on this financial article, answer: {question}\n\nArticle: {context}\n\nAnswer:"
    
    elif keyword_overlap > 0.3:  # Good relevance
        return f"Based on this relevant article, provide a detailed answer to: {question}\n\nArticle: {context}\n\nAnswer:"
    
    else:  # Low relevance - be explicit about mismatch
        return f"Read this article and answer the question. If the article doesn't contain relevant information, clearly state what the article actually discusses instead.\n\nQuestion: {question}\n\nArticle: {context}\n\nAnswer (be specific if information is not available):"

def test_intelligent_prompting():
    """Test the intelligent prompting system"""
    
    print("=== TESTING INTELLIGENT PROMPTING SYSTEM ===")
    
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
        test_content = """Why Microsoft Stock Is Sinking Today Keith Noonan, The Motley Fool Sat, Sep 6, 2025, 1:06 AM 3 min read MSFT -2.55% Key Points Microsoft stock is pulling back today in response to weak August jobs data. Soft jobs numbers in August suggest that the Federal Reserve will cut interest rates this month, but the U.S. economy is looking weaker than expected. New tariffs on semiconductors could also negatively impact Microsoft."""
        
        clean_context = clean_text_content(test_content)
        
        test_cases = [
            {
                'question': "What are Microsoft's AI plans?",
                'expected_behavior': "Should detect this is about AI/plans and context is irrelevant"
            },
            {
                'question': "What is Microsoft's target price?", 
                'expected_behavior': "Should detect this is financial but no price info in article"
            },
            {
                'question': "Why is Microsoft stock declining?",
                'expected_behavior': "Should detect good keyword overlap and answer well"
            },
            {
                'question': "What are the Federal Reserve rate changes?",
                'expected_behavior': "Should detect some relevance but limited info"
            }
        ]
        
        for test_case in test_cases:
            question = test_case['question']
            expected = test_case['expected_behavior']
            
            print(f"\n{'='*60}")
            print(f"QUESTION: {question}")
            print(f"EXPECTED: {expected}")
            print(f"{'='*60}")
            
            # Analyze question
            question_types = analyze_question_type(question)
            print(f"Detected Question Types: {question_types}")
            
            # Create smart prompt
            smart_prompt = create_smart_prompt(question, clean_context, question_types)
            print(f"Smart Prompt: {smart_prompt[:200]}...")
            
            # Get LLM response
            response = qa_pipeline(smart_prompt, max_length=400, num_return_sequences=1, do_sample=True, temperature=0.3)
            answer = response[0]['generated_text'].strip()
            
            print(f"LLM Response: {answer}")
            
            # Apply post-processing
            if len(answer) < 15 or answer.lower() in ['data', 'jobs data', 'stock', 'microsoft', 'ai', 'not available']:
                main_topic = extract_main_topic(clean_context)
                final_answer = f"**Analysis Result:** The provided article does not contain specific information to answer '{question}'. **Article Focus:** The content primarily discusses {main_topic}. **Recommendation:** To get detailed information about your question, please provide articles that specifically cover this topic."
                print(f"Final Answer (Post-processed): {final_answer}")
            else:
                print(f"Final Answer: **Answer:** {answer}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_intelligent_prompting()
