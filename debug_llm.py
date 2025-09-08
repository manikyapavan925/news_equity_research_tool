#!/usr/bin/env python3

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

# Test the flow
sample_content = """Why Microsoft Stock Is Sinking Today Keith
 Noonan, The Motley Fool Sat, Sep 6, 2025, 1:06 AM 3 min read MSFT -2.55% ^SPX -0.32% ^IXIC -0.03% Key Points Microsoft stock is pulling back today in response to weak August jobs data. Soft jobs numbers in August suggest that the Federal Reserve will cut interest rates this month, but the U.S. economy is looking weaker than expected. New tariffs on semiconductors could also negatively impact Microsoft. 10 stocks we like better than Microsoft › Microsoft"""

question = "what are AI plans for Microsoft"

print("=== DEBUGGING LLM PROCESSING ===")
print(f"Question: {question}")
print(f"Raw content length: {len(sample_content)}")
print(f"Raw content preview: {sample_content[:200]}...")

clean_content = clean_text_content(sample_content)
print(f"\nCleaned content length: {len(clean_content)}")
print(f"Cleaned content preview: {clean_content[:300]}...")

# Simulate the combined content formatting
combined_content = f"\n\n=== Why Microsoft Stock Is Sinking Today ===\n{clean_content[:1500]}"
print(f"\nCombined content length: {len(combined_content)}")
print(f"Combined content preview: {combined_content[:400]}...")

# Simulate the prompt generation for Flan-T5
if len(combined_content) > 2500:
    context_for_llm = combined_content[:2500] + "..."
else:
    context_for_llm = combined_content

prompt = f"Based on the following content, answer this question in detail: {question}\n\nContent: {context_for_llm}\n\nAnswer:"
print(f"\nPrompt length: {len(prompt)}")
print(f"Prompt preview: {prompt[:500]}...")

print("\n=== ANALYSIS ===")
print("The content is about Microsoft stock decline due to jobs data and tariffs.")
print("The question asks about AI plans, which is NOT covered in this article.")
print("The LLM should respond that the article doesn't contain information about AI plans.")
print("If it's giving the same generic response, there might be an issue with:")
print("1. Model caching/persistence")
print("2. Context not being properly passed")
print("3. Model not actually being used (fallback to rule-based)")
