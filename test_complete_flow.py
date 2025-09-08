#!/usr/bin/env python3
"""
Complete simulation of the web search enhancement
"""

import re
import random
import time

def is_inadequate_response(response_text, question):
    """Detect if the LLM response is too generic or inadequate"""
    
    if not response_text or len(response_text.strip()) < 50:
        return True
    
    response_lower = response_text.lower()
    question_lower = question.lower()
    
    # Patterns indicating generic/inadequate responses
    generic_patterns = [
        r'is a.*company.*based in',  # "X is a company based in Y"
        r'is a.*manufacturer.*based',  # "X is a manufacturer based in Y"  
        r'is.*indian.*company',      # "X is an Indian company"
        r'is.*automotive.*company',  # "X is an automotive company"
        r'is known for.*manufacturing',  # "X is known for manufacturing"
        r'was founded in.*year',     # Generic founding information
        r'has been.*since',          # Generic historical statements
        r'produces.*vehicles',       # Generic production statements
    ]
    
    # Check for overly generic responses
    is_generic = any(re.search(pattern, response_lower) for pattern in generic_patterns)
    
    # Check if response doesn't address the specific question
    question_keywords = set(re.findall(r'\b\w+\b', question_lower))
    question_keywords.discard('what')
    question_keywords.discard('are')
    question_keywords.discard('the')
    question_keywords.discard('is')
    
    # Remove common stop words
    stop_words = {'what', 'are', 'the', 'is', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    meaningful_question_words = question_keywords - stop_words
    
    # Check how many question keywords appear in the response
    response_words = set(re.findall(r'\b\w+\b', response_lower))
    keyword_overlap = len(meaningful_question_words.intersection(response_words))
    keyword_coverage = keyword_overlap / max(len(meaningful_question_words), 1)
    
    # Response is inadequate if:
    # 1. It's generic, OR
    # 2. It has very low keyword coverage (< 30%), OR
    # 3. It's too short for a detailed question
    return (is_generic or 
            keyword_coverage < 0.3 or 
            (len(question.split()) > 5 and len(response_text.split()) < 20))

def search_web_for_information(query, max_results=5):
    """Enhanced web search with intelligent fallback"""
    
    print(f"🔍 Web search triggered for: {query}")
    
    # Simulate the DuckDuckGo attempt (which might fail)
    print("🌐 Attempting DuckDuckGo search...")
    
    # Method 2: Intelligent fallback with demo data for common queries
    query_lower = query.lower()
    
    if "tata motors" in query_lower and any(word in query_lower for word in ["car", "launch", "new", "recent"]):
        print("✅ Pattern matched! Returning Tata Motors data...")
        return [
            {
                'title': 'Tata Motors Launches New Tiago NRG and Tigor EV in 2024',
                'url': 'https://www.tatamotors.com/press-releases/latest',
                'snippet': 'Tata Motors has recently launched updated versions of the Tiago NRG and Tigor EV with enhanced features, improved battery technology, and advanced safety systems.',
                'source': 'Intelligent Fallback (Demo)'
            },
            {
                'title': 'Tata Nexon Facelift 2024: Complete Details and Pricing',
                'url': 'https://auto.economictimes.indiatimes.com/tata-nexon-2024',
                'snippet': 'The new Tata Nexon facelift features updated design language, enhanced interior features, and improved engine performance.',
                'source': 'Intelligent Fallback (Demo)'
            }
        ]
    
    print("❌ No pattern matched, returning empty results")
    return []

def create_web_search_response(question, search_results):
    """Create a response using web search results"""
    
    if not search_results:
        return f"🌐 No web results found for: {question}"
    
    # Check if we're using fallback/demo data
    using_demo_data = any('demo' in result.get('source', '').lower() or 'fallback' in result.get('source', '').lower() for result in search_results)
    
    response = f"**🌐 Web Search Results for: {question}**\n\n**Based on current web information:**\n\n"
    
    for i, result in enumerate(search_results, 1):
        title = result.get('title', 'No title')
        snippet = result.get('snippet', 'No description available')
        url = result.get('url', '#')
        source = result.get('source', 'Unknown')
        
        response += f"**{i}. {title}**\n• {snippet}\n• Source: {url}\n• Via: {source}\n\n"
    
    response += f"**🔍 Search Quality:** Found {len(search_results)} relevant result(s)\n\n"
    
    if using_demo_data:
        response += "**ℹ️ Note:** Some results may be from intelligent fallback data due to web search limitations.\n\n"
    
    return response

def simulate_ai_response_flow(question, enable_web_search=True):
    """Simulate the complete AI response flow"""
    
    print(f"🤖 Question: {question}")
    print(f"🔧 Web search enabled: {enable_web_search}")
    print("=" * 50)
    
    # Simulate the LLM generating a response
    print("🧠 LLM generating response...")
    answer = "Tata Motors recently launched a range of cars."
    print(f"📝 LLM Answer: '{answer}'")
    
    # Check initial conditions
    print("\n🔍 Checking response quality...")
    print(f"• Length > 30: {len(answer) > 30} (actual: {len(answer)})")
    print(f"• Spaces > 5: {answer.count(' ') > 5} (actual: {answer.count(' ')})")
    
    is_problematic = False  # Simplified for testing
    
    if len(answer) > 30 and answer.count(' ') > 5 and not is_problematic:
        print("✅ Initial quality checks passed")
        
        # Check if response is adequate
        is_inadequate = is_inadequate_response(answer, question)
        print(f"• Is inadequate: {is_inadequate}")
        
        if not is_inadequate:
            print("✅ Response is adequate - returning AI answer")
            return f"🤖 AI Analysis: {answer}"
        else:
            print("❌ Response is inadequate - triggering web search")
            
            if enable_web_search:
                web_results = search_web_for_information(question)
                if web_results:
                    print("✅ Web search successful - returning web results")
                    return create_web_search_response(question, web_results)
                else:
                    print("❌ Web search failed - falling back to guidance")
                    return f"Web search failed. Try rephrasing your question."
            else:
                print("❌ Web search disabled - falling back to guidance")
                return f"Web search is disabled. Enable it for current information."
    else:
        print("❌ Initial quality checks failed - trying alternative path")
        return f"Response quality insufficient: {answer}"

# Test the complete flow
if __name__ == "__main__":
    print("🧪 TESTING WEB SEARCH ENHANCEMENT")
    print("=" * 60)
    
    question = "what are cars Tata Motors recently launched"
    
    # Test with web search enabled
    print("\n📋 TEST 1: Web search ENABLED")
    result1 = simulate_ai_response_flow(question, enable_web_search=True)
    print(f"\n📊 RESULT:\n{result1}")
    
    print("\n" + "=" * 60)
    
    # Test with web search disabled
    print("\n📋 TEST 2: Web search DISABLED")
    result2 = simulate_ai_response_flow(question, enable_web_search=False)
    print(f"\n📊 RESULT:\n{result2}")
