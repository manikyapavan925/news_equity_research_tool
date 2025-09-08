#!/usr/bin/env python3

import os
import sys

def test_improved_prompts():
    """Test different prompt formats for better LLM responses"""
    
    print("=== TESTING IMPROVED PROMPTS ===")
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
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
        
        # Test different prompt formats
        context = "Microsoft stock is declining due to weak jobs data and tariff concerns. The Federal Reserve may cut interest rates. New tariffs on semiconductors could negatively impact Microsoft. The article discusses market reactions and economic factors affecting the stock price."
        question = "What are Microsoft's AI plans?"
        
        prompts = [
            # Format 1: Instruction-based
            f"Answer the following question based on the given context. If the information is not in the context, say so clearly.\n\nQuestion: {question}\nContext: {context}\nAnswer:",
            
            # Format 2: Direct instruction
            f"Based on this article about Microsoft, answer: {question}\n\nArticle: {context}\n\nAnswer:",
            
            # Format 3: Flan-T5 preferred format
            f"Please answer this question: {question}\n\nUsing this information: {context}",
            
            # Format 4: Simple and direct
            f"Question: {question}\nContext: {context}\nIf the context doesn't contain information to answer the question, please say that clearly."
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt length: {len(prompt)}")
            print(f"Prompt preview: {prompt[:150]}...")
            
            try:
                response = qa_pipeline(prompt, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.3)
                answer = response[0]['generated_text'].strip()
                print(f"Response: {answer}")
                
                # Check if response is meaningful
                if len(answer) > 10 and "AI" in answer or "information" in answer.lower() or "not" in answer.lower():
                    print("✅ This prompt format looks promising!")
                else:
                    print("⚠️ Response seems too short or unclear")
                    
            except Exception as e:
                print(f"❌ Error with prompt {i}: {e}")
        
        print("\n=== TESTING WITH RELEVANT CONTENT ===")
        # Test with content that actually mentions AI
        ai_context = "Microsoft announced new AI initiatives for 2024, including expanded ChatGPT integration, enhanced Azure AI services, and partnerships with OpenAI. The company plans to invest $10 billion in AI infrastructure and integrate AI capabilities across Office 365, Azure cloud services, and Windows operating system."
        
        ai_prompt = f"Answer this question: {question}\n\nBased on this information: {ai_context}"
        print(f"Testing with AI-relevant content...")
        response = qa_pipeline(ai_prompt, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.3)
        print(f"Response with AI content: {response[0]['generated_text']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_improved_prompts()
