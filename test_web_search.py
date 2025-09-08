#!/usr/bin/env python3
"""
Test script to demonstrate the new web search functionality
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import re

def search_web_for_information(query, max_results=5):
    """Enhanced web search for any type of query with better content extraction"""
    try:
        # Create a more natural search query
        search_query = query.strip()
        
        # Use DuckDuckGo search (doesn't require API key)
        search_url = f"https://duckduckgo.com/html/?q={quote(search_query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Try to get search results
        response = requests.get(search_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract search results with improved parsing
            results = []
            
            # Look for result links and snippets
            for result_div in soup.find_all('div', {'class': re.compile('result')}):
                try:
                    # Extract title
                    title_elem = result_div.find('a', href=True)
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text().strip()
                    href = title_elem.get('href', '')
                    
                    # Extract snippet/description
                    snippet = ""
                    snippet_elem = result_div.find('span', {'class': re.compile('snippet')})
                    if not snippet_elem:
                        # Try alternative selectors for snippet
                        snippet_elem = result_div.find('div', {'class': re.compile('snippet|desc')})
                    
                    if snippet_elem:
                        snippet = snippet_elem.get_text().strip()
                    
                    # Only include results with meaningful content
                    if title and len(title) > 10 and snippet and len(snippet) > 20:
                        results.append({
                            'title': title[:200],
                            'url': href[:500] if href.startswith('http') else f"https://duckduckgo.com{href}",
                            'snippet': snippet[:500],
                            'source': 'DuckDuckGo Search'
                        })
                        
                        if len(results) >= max_results:
                            break
                            
                except Exception as parse_error:
                    continue
            
            return results
            
    except Exception as e:
        print(f"Web search error: {e}")
        return []
    
    return []

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

def create_web_search_response(question, search_results):
    """Create a response using web search results"""
    
    if not search_results:
        return f"""**🌐 Web Search Results for: {question}**

⚠️ **No current web results found**

**Suggested Actions:**
• Try rephrasing your question with different keywords
• Search directly on reliable news websites
• Check official company websites or press releases
• Use specific model names or dates if applicable

**For latest information about companies, try:**
• [Company Name] + "latest news"
• [Company Name] + "recent launches" 
• [Company Name] + "new products 2024"
"""
    
    response = f"""**🌐 Web Search Results for: {question}**

**Based on current web information:**

"""
    
    for i, result in enumerate(search_results, 1):
        title = result.get('title', 'No title')
        snippet = result.get('snippet', 'No description available')
        url = result.get('url', '#')
        
        response += f"""**{i}. {title}**
• {snippet}
• Source: {url}

"""
    
    response += f"""**🔍 Search Quality:** Found {len(search_results)} relevant result(s)

**💡 For more detailed information:**
• Visit the source links above for complete articles
• Try more specific search terms
• Check multiple sources for verification

*Web search results are current as of search time and may contain varying levels of accuracy.*"""
    
    return response

# Test cases
def main():
    print("🧪 Testing Web Search Functionality")
    print("="*50)
    
    # Test case 1: Tata Motors question
    question1 = "what are cars Tata Motors recently launched"
    print(f"\n📝 Question: {question1}")
    
    # Simulate an inadequate LLM response
    llm_response = "Tata Motors is a car manufacturer based in Mumbai, India."
    print(f"🤖 LLM Response: {llm_response}")
    
    # Check if response is inadequate
    is_inadequate = is_inadequate_response(llm_response, question1)
    print(f"❓ Is response inadequate? {is_inadequate}")
    
    if is_inadequate:
        print("\n🌐 Triggering web search...")
        web_results = search_web_for_information(question1)
        web_response = create_web_search_response(question1, web_results)
        print(web_response)
    
    print("\n" + "="*50)
    
    # Test case 2: Current events question
    question2 = "latest Tesla Model Y price 2024"
    print(f"\n📝 Question: {question2}")
    print("🌐 Direct web search...")
    web_results = search_web_for_information(question2)
    web_response = create_web_search_response(question2, web_results)
    print(web_response)

if __name__ == "__main__":
    main()
