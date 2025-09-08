#!/usr/bin/env python3

import os
import sys

# Add the current directory to Python path
sys.path.append('/Users/nethimanikyapavan/Documents/augment-projects/News_Equity_Research_Tool')

def test_llm_loading():
    """Test if LLM models can be loaded and used"""
    
    print("=== TESTING LLM LOADING ===")
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        print("✅ Transformers library imported successfully")
        
        # Test 1: Try Flan-T5-base
        print("\n1. Testing Google Flan-T5-base...")
        try:
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
            
            # Test with a simple question
            test_prompt = "Based on the following content, answer this question in detail: What are the AI plans for Microsoft?\n\nContent: Microsoft stock is declining due to weak jobs data and tariff concerns. The article discusses market reactions but doesn't mention AI initiatives.\n\nAnswer:"
            
            response = qa_pipeline(test_prompt, max_length=200, num_return_sequences=1)
            print(f"✅ Flan-T5 loaded successfully!")
            print(f"Test response: {response[0]['generated_text']}")
            return qa_pipeline, "google/flan-t5-base"
            
        except Exception as e:
            print(f"❌ Flan-T5 failed: {e}")
            
        # Test 2: Try DistilBERT
        print("\n2. Testing DistilBERT...")
        try:
            model_name = "distilbert-base-cased-distilled-squad"
            qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name
            )
            
            # Test with Q&A format
            test_context = "Microsoft stock is declining due to weak jobs data and tariff concerns. The article discusses market reactions but doesn't mention AI initiatives."
            test_question = "What are Microsoft's AI plans?"
            
            response = qa_pipeline(question=test_question, context=test_context)
            print(f"✅ DistilBERT loaded successfully!")
            print(f"Test response: {response}")
            return qa_pipeline, "distilbert-base-cased-distilled-squad"
            
        except Exception as e:
            print(f"❌ DistilBERT failed: {e}")
            
        # Test 3: Try T5-small
        print("\n3. Testing T5-small...")
        try:
            model_name = "t5-small"
            qa_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                max_length=300
            )
            
            test_prompt = "question: What are Microsoft's AI plans? context: Microsoft stock is declining due to weak jobs data and tariff concerns. The article discusses market reactions but doesn't mention AI initiatives."
            
            response = qa_pipeline(test_prompt, max_length=150)
            print(f"✅ T5-small loaded successfully!")
            print(f"Test response: {response}")
            return qa_pipeline, "t5-small"
            
        except Exception as e:
            print(f"❌ T5-small failed: {e}")
            
        print("\n❌ All models failed to load!")
        return None, None
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Transformers library might not be installed properly")
        return None, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None, None

if __name__ == "__main__":
    test_llm_loading()
