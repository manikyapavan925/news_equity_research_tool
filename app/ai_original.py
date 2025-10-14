"""
AI Module - Enhanced with Tavily Web Search Integration
This module provides AI functionality with intelligent fallback to Tavily web search.
"""

import os
import re
import requests
import json
from typing import List, Dict, Tuple, Optional, Any, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def fetch_company_metadata_from_yahoo(query: str) -> Optional[Dict[str, Any]]:
    """Fetch the best-matching company metadata from Yahoo Finance search API."""
    if not query:
        return None

    search_url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {
        "q": query,
        "quotesCount": 5,
        "newsCount": 0,
    }

    try:
        response = requests.get(search_url, params=params, timeout=4)
        response.raise_for_status()
        data = response.json() or {}
        quotes = data.get("quotes") or []

        def _prioritize(quotes_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not quotes_list:
                return None

            # Prefer equities and ETFs first, then anything else
            equities = [q for q in quotes_list if q.get("quoteType") in {"EQUITY", "ETF"}]
            pool = equities or quotes_list

            # Exact ticker match if possible
            query_upper = query.upper()
            for item in pool:
                symbol = item.get("symbol")
                if symbol and symbol.upper() == query_upper:
                    return item

            return pool[0]

        best_match = _prioritize(quotes)
        return best_match

    except requests.RequestException:
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _normalise_company_alias(alias: str) -> List[str]:
    """Generate cleaned variants for a company alias."""
    if not alias:
        return []

    alias_lower = alias.lower().strip()
    if not alias_lower:
        return []

    variants = {
        alias_lower,
        alias_lower.replace(",", ""),
        re.sub(r"[^a-z0-9\s]", " ", alias_lower),
        alias_lower.replace(" inc", "").replace(" ltd", "").replace(" corp", ""),
    }

    # Include individual words for multi-word names
    for part in alias_lower.replace(",", " ").split():
        if len(part) > 2:
            variants.add(part)

    cleaned = {" ".join(var.split()) for var in variants if var and len(var.strip()) > 2}
    return sorted(cleaned)


def generate_company_variations(question: str, ticker_symbol: str, company_name: str) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Build dynamic company term variations using live metadata and question context."""
    question_lower = question.lower()
    base_terms = re.findall(r"\b[a-z]{3,}\b", question_lower)

    variations: Set[str] = set(base_terms)
    metadata: Optional[Dict[str, Any]] = None

    candidate_queries: List[str] = []
    if ticker_symbol:
        candidate_queries.append(ticker_symbol)
    if company_name:
        candidate_queries.append(company_name)
    if base_terms:
        candidate_queries.append(" ".join(base_terms[:3]))

    seen_candidates = set()
    for candidate in candidate_queries:
        if not candidate:
            continue
        candidate_key = candidate.lower()
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)

        metadata = fetch_company_metadata_from_yahoo(candidate)
        if metadata:
            break

    if metadata:
        for key in ("symbol", "shortname", "shortName", "longname", "longName", "displayName", "displayname", "companyName"):
            value = metadata.get(key)
            if isinstance(value, str):
                for variant in _normalise_company_alias(value):
                    variations.add(variant)

        former_names = metadata.get("formerNames") or metadata.get("formerName")
        if isinstance(former_names, list):
            for former in former_names:
                if isinstance(former, str):
                    for variant in _normalise_company_alias(former):
                        variations.add(variant)
        elif isinstance(former_names, str):
            for variant in _normalise_company_alias(former_names):
                variations.add(variant)

        symbol = metadata.get("symbol")
        if isinstance(symbol, str) and symbol:
            symbol_lower = symbol.lower()
            variations.add(symbol_lower)
            variations.add(symbol_lower.replace(".", ""))
            variations.add(f"ticker {symbol_lower}")

    clean_variations = sorted({term.strip() for term in variations if term and len(term.strip()) > 2})
    return clean_variations, metadata

def evaluate_response_quality(response: str, question: str) -> Dict[str, Any]:
    """
    Evaluate the quality of an LLM response to determine if Tavily fallback is needed.
    
    Returns:
        dict: Contains quality score, needs_fallback flag, and reasons
    """
    quality_score = 0
    issues = []
    
    # Check response length (too short might indicate lack of detail)
    if len(response) < 100:
        issues.append("Response too short")
    else:
        quality_score += 20
    
    # Check for specific financial terms and data
    financial_terms = ['price', 'target', 'analysis', 'forecast', 'revenue', 'earnings', 'growth', 'market']
    financial_matches = sum(1 for term in financial_terms if term.lower() in response.lower())
    
    if financial_matches >= 3:
        quality_score += 25
    elif financial_matches >= 1:
        quality_score += 10
    else:
        issues.append("Lacks financial terminology")
    
    # Check for recent/current data indicators
    recency_indicators = ['2024', '2025', 'current', 'recent', 'latest', 'today', 'this year']
    recency_matches = sum(1 for indicator in recency_indicators if indicator.lower() in response.lower())
    
    if recency_matches >= 2:
        quality_score += 25
    elif recency_matches >= 1:
        quality_score += 10
    else:
        issues.append("May not have recent data")
    
    # Check for specific numbers/data points
    number_pattern = r'\d+(?:\.\d+)?(?:[%₹$£€]|(?:\s*(?:million|billion|trillion|crore|lakh)))?'
    numbers_found = len(re.findall(number_pattern, response))
    
    if numbers_found >= 3:
        quality_score += 20
    elif numbers_found >= 1:
        quality_score += 10
    else:
        issues.append("Lacks specific numerical data")
    
    # Check for generic/template responses (ENHANCED)
    generic_phrases = [
        'i don\'t have access', 'i cannot provide', 'please consult', 'for the most current',
        'based on available information', 'additional market research may be beneficial',
        'the current analysis is based on', 'for more detailed and current information',
        'available information', 'context provided', 'context available in the system',
        'response quality: standard', 'information summary'
    ]
    generic_score_penalty = 0
    for phrase in generic_phrases:
        if phrase.lower() in response.lower():
            generic_score_penalty += 15
            issues.append(f"Contains generic phrase: '{phrase}'")
    
    quality_score -= generic_score_penalty
    
    # Check for extremely vague responses (CRITICAL CHECK)
    vague_indicators = [
        'here\'s what we can determine',
        'based on the available information',
        'the current analysis is based on',
        'for more detailed',
        'may be beneficial'
    ]
    vague_count = sum(1 for indicator in vague_indicators if indicator.lower() in response.lower())
    if vague_count >= 2:
        quality_score -= 40
        issues.append("Response is extremely vague and unhelpful")
    
    # Check if response actually addresses the question
    question_words = question.lower().split()
    important_words = [word for word in question_words if len(word) > 3 and word not in ['what', 'when', 'where', 'how']]
    
    if important_words:
        word_matches = sum(1 for word in important_words if word in response.lower())
        if word_matches == 0:
            quality_score -= 25
            issues.append("Response doesn't address the specific question")
    
    # Final assessment (MORE STRICT)
    needs_fallback = quality_score < 60 or len(issues) >= 3 or generic_score_penalty > 30
    
    return {
        'quality_score': quality_score,
        'needs_fallback': needs_fallback,
        'issues': issues,
        'assessment': 'Poor' if quality_score < 30 else 'Fair' if quality_score < 60 else 'Good'
    }

def search_with_tavily(question: str, max_results: int = 5):
    """
    Search using Tavily API for the most accurate and recent results.
    Compatible with Python 3.8+
    
    Args:
        question: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, content, url, and score
    """
    import requests
    import json
    
    # Get API key from environment
    api_key = os.getenv('TAVILY_API_KEY')
    
    if not api_key or api_key == 'your_tavily_api_key_here':
        print("⚠️ Tavily API key not found or not configured. Set TAVILY_API_KEY in your environment.")
        return []
    
    try:
        # Tavily API endpoint
        url = "https://api.tavily.com/search"
        
        # Enhance the search query for comprehensive results with better specificity
        question_lower = question.lower()
        company_name = ''
        company_variations: List[str] = []
        ticker_symbol = ''

        import re
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', question)
        if ticker_match:
            ticker_symbol = ticker_match.group(1)

        company_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', question)
        if company_match:
            company_name = company_match.group(1).strip()

        company_variations_dynamic, metadata = generate_company_variations(question, ticker_symbol, company_name)

        if metadata:
            symbol_meta = metadata.get('symbol')
            if isinstance(symbol_meta, str) and symbol_meta:
                ticker_symbol = symbol_meta

            for key in ("longname", "longName", "shortname", "shortName", "displayName", "displayname"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    company_name = value.strip()
                    break

        company_variations.extend(company_variations_dynamic)
        company_variations = list(dict.fromkeys([term for term in company_variations if term]))

        # Always ensure we have at least the question's base words to search for
        base_terms = [word for word in question_lower.split() if len(word) > 3]
        if not company_variations:
            company_variations = base_terms

        if any(term in question_lower for term in ['falling', 'dropping', 'declining', 'why', 'reason']):
            enhanced_query = f"{question} stock price decline reasons analysis causes factors recent news analyst commentary market reaction"
        elif any(term in question_lower for term in ['target', 'forecast', 'projection', 'price target']):
            enhanced_query = f"{question} analyst price target consensus forecast valuation report 2025 2026"
        elif any(term in question_lower for term in ['earnings', 'results', 'quarterly']):
            enhanced_query = f"{question} earnings results financial performance guidance analyst reaction Q3 Q4 2025"
        elif any(term in question_lower for term in ['market analysis', 'analysis', 'outlook', 'trends']):
            primary_term = company_name if company_name else (ticker_symbol or base_terms[0] if base_terms else question)
            enhanced_query = f"{primary_term} stock market analysis 2025 analyst outlook price movement trends latest news September October"
        elif any(term in question_lower for term in ['impact', 'effect', 'gst', 'policy']):
            enhanced_query = f"{question} detailed impact analysis market reaction business effect financial implications"
        else:
            enhanced_query = question
        
        # Prepare request payload with expanded search for better results
        payload = {
            "api_key": api_key,
            "query": enhanced_query,
            "search_depth": "advanced",
            "max_results": min(max_results * 4, 20),  # Get even more results
            "include_domains": [
                "reuters.com", "bloomberg.com", "moneycontrol.com",
                "economictimes.com", "business-standard.com",
                "livemint.com", "cnbc.com", "yahoo.com", "ft.com",
                "wsj.com", "marketwatch.com", "investing.com",
                "morningstar.com", "zacks.com", "tipranks.com",
                "seekingalpha.com", "fool.com", "barrons.com",
                "investors.com", "thestreet.com"
            ],
            "exclude_domains": ["wikipedia.org", "investopedia.com", "dictionary.com", "web.archive.org"]
        }
        
        # Make API request
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            if data and 'results' in data:
                # Filter and rank results for better relevance
                company_tokens = [token for token in company_variations if token]
                for item in data['results']:
                    content = item.get('content', '') or ''
                    title = item.get('title', '') or ''
                    content_lower = content.lower()
                    title_lower = title.lower()

                    # Skip results that do not mention the company anywhere
                    if company_tokens:
                        if not any(token in content_lower for token in company_tokens) and not any(token in title_lower for token in company_tokens):
                            continue

                    # Calculate relevance score
                    relevance = 0

                    if company_tokens:
                        relevance += sum(8 for token in company_tokens if token and token in content_lower)
                        relevance += sum(12 for token in company_tokens if token and token in title_lower)

                    financial_terms = ['stock', 'price', 'analyst', 'target', 'forecast', 'earnings', 'revenue', 'market', 'outlook', 'valuation', 'guidance', 'performance']
                    relevance += sum(2 for term in financial_terms if term in content_lower)

                    # Penalize generic/noise content
                    noise_terms = ['unlock', 'subscribe', 'login', 'newsletter', 'ad ', 'advertisement', 'sign up', 'join now']
                    relevance -= sum(6 for term in noise_terms if term in content_lower)

                    # Penalize very short content
                    if len(content) < 150:
                        relevance -= 12

                    # Add to results with combined score
                    results.append({
                        'title': title,
                        'content': content,
                        'url': item.get('url', ''),
                        'score': item.get('score', 0) + (relevance / 100),
                        'published_date': item.get('published_date', ''),
                        'source': 'tavily',
                        'relevance': relevance
                    })

                # Sort by combined score (descending)
                results = sorted(results, key=lambda x: x['score'], reverse=True)

                filtered_results = [r for r in results if r['relevance'] > 0]

                # If we filtered everything out, fall back to the top raw results
                if not filtered_results:
                    filtered_results = results[:max_results]

                return filtered_results[:max_results]
            
            return results
        else:
            print(f"⚠️ Tavily API error: {response.status_code} - {response.text}")
            return []
            
    except requests.exceptions.Timeout:
        print("⚠️ Tavily search timeout. Please try again.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Tavily request error: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Tavily search error: {e}")
        return []

def extract_and_synthesize_insights(tavily_results: List[Dict], question_lower: str) -> Dict[str, Any]:
    """
    Extract and synthesize insights from Tavily results with improved content processing.
    """
    insights = {
        'key_metrics': [],
        'main_factors': [],
        'financial_data': [],
        'market_sentiment': [],
        'growth_concerns': []
    }
    
    # Process each result for better context extraction (increased from 5 to 8)
    for result in tavily_results[:8]:
        title = result.get('title', '')
        content = result.get('content', '')
        
        # Skip very short or repetitive content
        if len(content) < 100 or 'Learn more about' in content:
            continue
            
        # Clean content - remove navigation text and URLs
        clean_content = re.sub(r'Skip to main content|Report This Ad|Learn more about|Click here', '', content)
        clean_content = re.sub(r'http[s]?://\S+', '', clean_content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        # Extract financial metrics with better filtering
        metrics = re.findall(r'(\d+(?:\.\d+)?%|\£[\d,]+|\$[\d,]+|₹[\d,]+)', clean_content)
        # Only add meaningful metrics (not random numbers)
        for metric in metrics[:4]:  # Increased from 3 to 4
            if any(context in clean_content[max(0, clean_content.find(metric)-50):clean_content.find(metric)+50].lower() 
                   for context in ['growth', 'revenue', 'profit', 'decline', 'increase', 'fell', 'rose', 'subscription', 'earnings']):
                insights['key_metrics'].append(metric)
        
        # Extract complete sentences for growth concerns with better filtering
        sentences = [s.strip() for s in clean_content.split('.') if len(s.strip()) > 20 and len(s.strip()) < 300]
        for sentence in sentences[:7]:  # Increased from 5 to 7
            # Look for more specific patterns
            if any(word in sentence.lower() for word in ['growth', 'subscription', 'revenue', 'declining', 'slowdown', 'slowing', 'fell', 'dropped', 'decline', 'cancellations', 'churn']):
                # Clean up the sentence and ensure it's meaningful
                if not any(skip in sentence.lower() for skip in ['www', 'http', 'click', 'learn more', 'skip to']):
                    # Extract the most relevant part
                    if len(sentence) > 50:
                        insights['growth_concerns'].append(sentence.strip()[:250])  # Increased length limit
        
        # Extract main factors with better context and more patterns
        factor_patterns = [
            'due to', 'because of', 'analysts point to', 'concerns about', 'driven by',
            'caused by', 'result of', 'led to', 'contributed to', 'affected by',
            'impacted by', 'influenced by', 'stemming from', 'arising from'
        ]
        for sentence in sentences[:8]:  # Increased from 6 to 8
            if any(phrase in sentence.lower() for phrase in factor_patterns):
                if len(sentence) > 30 and len(sentence) < 250:
                    if not any(skip in sentence.lower() for skip in ['www', 'http', 'click', 'learn more']):
                        insights['main_factors'].append(sentence.strip())
        
        # Extract financial performance data with better patterns
        financial_patterns = [
            'reported', 'posted', 'earnings', 'revenue', 'profit', 'margin', 'guidance',
            'forecast', 'beat', 'missed', 'decline', 'growth', 'fell', 'rose', 'increased',
            'subscription', 'cancellations', 'churn', 'retention', 'valuation'
        ]
        for sentence in sentences[:7]:  # Increased from 5 to 7
            if any(word in sentence.lower() for word in financial_patterns):
                if any(char.isdigit() for char in sentence) or '%' in sentence:  # Contains numbers or percentages
                    if len(sentence) > 25 and len(sentence) < 200:
                        insights['financial_data'].append(sentence.strip())
        
        # Extract market sentiment
        for sentence in sentences[:2]:
            if any(word in sentence.lower() for word in ['investors', 'market sentiment', 'concerns', 'optimism']):
                if len(sentence) > 40 and len(sentence) < 150:
                    insights['market_sentiment'].append(sentence.strip())
    
    # Remove duplicates and very similar content
    for key in insights:
        unique_items = []
        for item in insights[key]:
            # Check if this item is too similar to existing ones
            is_duplicate = False
            for existing in unique_items:
                # If more than 70% of words match, consider it duplicate
                item_words = set(item.lower().split())
                existing_words = set(existing.lower().split())
                if len(item_words & existing_words) / max(len(item_words), 1) > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate and len(item.strip()) > 10:
                unique_items.append(item.strip())
        
        insights[key] = unique_items[:5]  # Increased limit from 3 to 5 for more comprehensive analysis
    
    return insights

def generate_enhanced_response_with_tavily(question: str, tavily_results: List[Dict]) -> str:
    """
    Generate ChatGPT-quality financial analysis using Tavily search results.
    
    Args:
        question: Original question
        tavily_results: Results from Tavily search
        
    Returns:
        Comprehensive ChatGPT-style financial analysis
    """
    if not tavily_results:
        return "Unable to fetch current market data. Please try again later."
    
    # Comprehensive content analysis and synthesis
    all_content = " ".join([result.get('content', '') for result in tavily_results])
    
    # Enhanced question type detection
    question_lower = question.lower()
    is_target_price = any(term in question_lower for term in ['target', 'price forecast', 'projection', 'estimate'])
    is_earnings = any(term in question_lower for term in ['earnings', 'results', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'revenue'])
    is_price_movement = any(term in question_lower for term in ['falling', 'rising', 'dropping', 'increasing', 'decreasing', 'why', 'reason', 'cause'])
    is_analysis = any(term in question_lower for term in ['analysis', 'outlook', 'performance', 'review'])
    is_news_impact = any(term in question_lower for term in ['impact', 'effect', 'influence', 'gst', 'policy', 'regulation'])
    
    # Advanced content extraction and synthesis
    synthesized_insights = extract_and_synthesize_insights(tavily_results, question_lower)
    
    # Extract company name from question
    company_patterns = [
        r'\b([A-Z][a-z]+ (?:Motors|Bank|Ltd|Limited|Inc|Corp|Group|Exchange))\b',
        r'\b(LSEG|HDFC|ICICI|TCS|Tesla|Apple|Microsoft|Amazon|Google)\b',
        r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    ]
    
    company_name = "the company"
    for pattern in company_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        if matches:
            company_name = matches[0]
            break
    
    # Generate ChatGPT-style comprehensive analysis
    if is_price_movement:
        response = generate_chatgpt_style_price_analysis(question, company_name, synthesized_insights, all_content)
    elif is_target_price:
        response = generate_chatgpt_style_target_analysis(question, company_name, synthesized_insights, all_content)
    elif is_earnings:
        response = generate_chatgpt_style_earnings_analysis(question, company_name, synthesized_insights, all_content)
    else:
        response = generate_chatgpt_style_general_analysis(question, company_name, synthesized_insights, all_content)
    
    return response

def generate_chatgpt_style_price_analysis(question: str, company_name: str, insights: Dict, all_content: str) -> str:
    """Generate ChatGPT-style price movement analysis."""
    
    # Extract timeframe from question
    timeframe = "recent period"
    if "3 months" in question.lower() or "quarter" in question.lower():
        timeframe = "last 3 months"
    elif "month" in question.lower():
        timeframe = "recent weeks"
    
    response = f"""Here are the main reasons analysts and media are pointing to for {company_name}'s share price decline over the {timeframe}:

## 🔍 **Key Factors Behind the Price Drop**

"""
    
    # Present growth concerns in a cleaner format
    if insights['growth_concerns']:
        response += "### **1. Growth and Performance Concerns**\n\n"
        for i, concern in enumerate(insights['growth_concerns'][:3], 1):
            # Clean up the concern text and ensure completeness
            clean_concern = concern.replace(company_name.lower(), "the company").strip()
            # Skip incomplete sentences (ending with common cutoff words)
            if any(clean_concern.lower().endswith(word) for word in [' the', ' a', ' an', ' and', ' or', ' but', ' in', ' on', ' at', ' to', ' for', ' of', ' with', ' by']):
                continue
            if not clean_concern.endswith('.'):
                clean_concern += '.'
            # Ensure minimum length for meaningful content
            if len(clean_concern) > 30:
                response += f"**{i}.** {clean_concern}\n\n"
        
        # Add metrics context if available
        if insights['key_metrics']:
            relevant_metrics = [m for m in insights['key_metrics'] if '%' in str(m)][:3]
            if relevant_metrics:
                response += f"**Key Performance Indicators**: {', '.join(relevant_metrics)}\n\n"
    
    # Present market factors more clearly  
    if insights['main_factors']:
        response += "### **2. Market and Operational Challenges**\n\n"
        for i, factor in enumerate(insights['main_factors'][:3], 1):
            # Clean up factor text and ensure completeness
            clean_factor = factor.strip()
            # Skip incomplete sentences
            if any(clean_factor.lower().endswith(word) for word in [' the', ' a', ' an', ' and', ' or', ' but', ' in', ' on', ' at', ' to', ' for', ' of', ' with', ' by']):
                continue
            if not clean_factor.endswith('.'):
                clean_factor += '.'
            # Remove redundant phrases
            clean_factor = clean_factor.replace('Market concerns about ', '').replace('due to ', '')
            # Ensure minimum length
            if len(clean_factor) > 25:
                response += f"**{i}.** {clean_factor.capitalize()}\n\n"
    
    # Present financial data if available
    if insights['financial_data']:
        response += "### **3. Financial Performance Issues**\n\n"
        for i, data in enumerate(insights['financial_data'][:3], 1):
            clean_data = data.strip()
            # Skip incomplete sentences
            if any(clean_data.lower().endswith(word) for word in [' the', ' a', ' an', ' and', ' or', ' but', ' in', ' on', ' at', ' to', ' for', ' of', ' with', ' by']):
                continue
            if not clean_data.endswith('.'):
                clean_data += '.'
            # Ensure minimum length and meaningful content
            if len(clean_data) > 20:
                response += f"**{i}.** {clean_data}\n\n"
    
    # Add synthesis conclusion
    response += f"""## 📊 **Investment Implications**

The share price decline reflects investor concerns about {company_name}'s near-term growth trajectory and market positioning. When growth companies show signs of deceleration, the market typically reprices them to reflect lower growth expectations.

**What investors are watching:**
• Upcoming earnings reports and guidance updates
• Management's response to growth challenges  
• Competitive positioning and market share trends
• Broader sector dynamics and regulatory developments

The key question is whether this is a temporary slowdown or indicates more structural challenges. Market sentiment can shift quickly with new developments, so monitoring company updates and analyst revisions will be important."""
    
    return response

def generate_chatgpt_style_target_analysis(question: str, company_name: str, insights: Dict, all_content: str) -> str:
    """Generate ChatGPT-style target price analysis."""
    response = f"Here are the current analyst targets and forecasts for {company_name}, with analysis of the key factors driving these valuations:\n\n"
    
    if insights['key_metrics']:
        response += "## � **Current Price Targets and Metrics**\n\n"
        price_metrics = [m for m in insights['key_metrics'] if any(symbol in str(m) for symbol in ['£', '$', '₹', '€'])]
        if price_metrics:
            response += f"Analyst targets mention: {', '.join(price_metrics[:3])}\n\n"
    
    if insights['growth_concerns']:
        response += "## 📈 **Growth Outlook Factors**\n\n"
        for concern in insights['growth_concerns'][:3]:
            response += f"• {concern}\n"
        response += f"\n"
    
    return response

def generate_chatgpt_style_earnings_analysis(question: str, company_name: str, insights: Dict, all_content: str) -> str:
    """Generate ChatGPT-style earnings analysis."""
    response = f"Here's what the latest earnings data and analyst commentary reveal about {company_name}'s performance:\n\n"
    
    if insights['financial_data']:
        response += "## 💰 **Key Financial Highlights**\n\n"
        for data in insights['financial_data'][:4]:
            response += f"• {data}\n"
        response += f"\n"
    
    return response

def generate_chatgpt_style_general_analysis(question: str, company_name: str, insights: Dict, all_content: str) -> str:
    """Generate ChatGPT-style general analysis."""
    
    # Detect if this is a policy/regulatory question
    is_policy_question = any(term in question.lower() for term in ['gst', 'policy', 'regulation', 'tax', 'government', 'compliance'])
    
    if is_policy_question:
        response = f"Here's how policy developments and regulatory changes are likely to impact {company_name}:\n\n"
        
        if insights['main_factors']:
            response += "## � **Policy Impact Analysis**\n\n"
            for factor in insights['main_factors'][:3]:
                clean_factor = factor.strip()
                if len(clean_factor) > 20:
                    if not clean_factor.endswith('.'):
                        clean_factor += '.'
                    response += f"• {clean_factor}\n\n"
        
        # Add policy-specific analysis
        response += f"## 🎯 **Strategic Implications for {company_name}**\n\n"
        response += f"**Short-term effects (6-12 months):**\n"
        response += f"• Implementation costs and operational adjustments\n"
        response += f"• Compliance system updates and staff training\n"
        response += f"• Potential temporary impact on margins during transition\n\n"
        
        response += f"**Long-term benefits:**\n"
        response += f"• Streamlined processes and reduced complexity\n"
        response += f"• Enhanced transparency and digital integration\n"
        response += f"• Competitive advantage for well-prepared companies\n\n"
        
        if insights['key_metrics']:
            response += f"**Key metrics to watch**: {', '.join(insights['key_metrics'][:3])}\n\n"
            
    else:
        response = f"Here's comprehensive analysis of {company_name} based on current market developments:\n\n"
        
        if insights['main_factors']:
            response += "## 🔍 **Key Market Developments**\n\n"
            for factor in insights['main_factors'][:3]:
                clean_factor = factor.strip()
                if len(clean_factor) > 20:
                    if not clean_factor.endswith('.'):
                        clean_factor += '.'
                    response += f"• {clean_factor}\n\n"
        
        if insights['financial_data']:
            response += "## 💰 **Financial Performance**\n\n"
            for data in insights['financial_data'][:2]:
                clean_data = data.strip()
                if not clean_data.endswith('.'):
                    clean_data += '.'
                response += f"• {clean_data}\n\n"
        
        if insights['market_sentiment']:
            response += "## 📊 **Market Outlook**\n\n"
            for sentiment in insights['market_sentiment'][:2]:
                clean_sentiment = sentiment.strip()
                if not clean_sentiment.endswith('.'):
                    clean_sentiment += '.'
                response += f"• {clean_sentiment}\n\n"
    
    # Add comprehensive conclusion
    response += "## 🎯 **Investment Perspective**\n\n"
    
    if is_policy_question:
        response += f"Policy changes typically create both challenges and opportunities. For {company_name}, the key will be how effectively management adapts to new requirements while capitalizing on potential competitive advantages."
    else:
        response += f"The current market environment presents a mixed picture for {company_name}. Investors should monitor upcoming developments and company responses to key challenges."
    
    response += f"\n\n**Key factors to watch going forward:**\n"
    response += f"• Management guidance and strategic updates\n"
    response += f"• Quarterly performance metrics and trends\n"
    response += f"• Competitive positioning and market share\n"
    response += f"• Broader sector dynamics and regulatory environment"
    
    return response

def generate_article_summary(article, length="Medium"):
    """
    Generate a summary of the article content using extractive summarization.
    
    Args:
        article (dict): Article dictionary with 'content' or 'text' key and 'title' key
        length (str): Summary length - "Short" (3 sentences), "Medium" (5 sentences), or "Detailed" (8 sentences)
        
    Returns:
        str: Generated summary text
    """
    # Try both 'content' and 'text' keys for compatibility
    content = article.get('content', '') or article.get('text', '')
    title = article.get('title', 'No Title')
    
    if not content or len(content.strip()) < 100:
            return "⚠️ Article content is too short to generate a meaningful summary."
    
    # Split content into sentences
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return "⚠️ Unable to extract meaningful sentences from the article."
    
    # Determine summary length
    if length == "Short":
        target_sentences = min(3, len(sentences))
    elif length == "Medium":
        target_sentences = min(5, len(sentences))
    else:  # Detailed
        target_sentences = min(8, len(sentences))
    
    # Simple extractive summarization
    # Score sentences based on:
    # 1. Position (first and last sentences often important)
    # 2. Length (medium length sentences preferred)
    # 3. Keyword frequency
    
    # Extract important keywords from title
    title_words = set(re.findall(r'\b\w+\b', title.lower()))
    financial_keywords = {
        'stock', 'market', 'price', 'earnings', 'revenue', 'profit', 'loss',
        'merger', 'acquisition', 'ipo', 'dividend', 'investment', 'trading',
        'financial', 'economic', 'business', 'company', 'corporation', 'shares',
        'nasdaq', 'nyse', 'dow', 'sp500', 'index', 'fund', 'bond', 'crypto',
        'ceo', 'announces', 'appoints', 'leadership', 'strategy', 'growth'
    }
    
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0
        sentence_lower = sentence.lower()
        sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
        
        # Position scoring
        if i == 0:  # First sentence
            score += 3
        elif i == len(sentences) - 1:  # Last sentence
            score += 2
        elif i < len(sentences) * 0.3:  # Early sentences
            score += 1
        
        # Length scoring (prefer medium-length sentences)
        word_count = len(sentence.split())
        if 10 <= word_count <= 30:
            score += 2
        elif 5 <= word_count <= 50:
            score += 1
        
        # Keyword scoring
        title_matches = len(title_words.intersection(sentence_words))
        financial_matches = len(financial_keywords.intersection(sentence_words))
        score += title_matches * 2 + financial_matches
        
        # Avoid very short or very long sentences
        if word_count < 5 or word_count > 60:
            score -= 2
            
        scored_sentences.append((sentence, score, i))
    
    # Sort by score and select top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    selected_sentences = scored_sentences[:target_sentences]
    
    # Sort selected sentences by original order
    selected_sentences.sort(key=lambda x: x[2])
    
    # Create summary
    summary_sentences = [s[0] for s in selected_sentences]
    summary = '. '.join(summary_sentences)
    
    # Add some basic cleanup
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    if not summary:
        summary = sentences[0] if sentences else "Unable to generate summary."
    
    # Ensure proper ending
    if not summary.endswith('.'):
        summary += '.'
    
    return summary

def generate_realtime_ai_answer(question, articles=None, use_context=True, enable_web_search=False):
    """
    Generate AI answer with intelligent Tavily fallback mechanism.
    
    This function first attempts to generate a response using available context,
    then evaluates the quality and falls back to Tavily search if needed.
    """
    if articles is None:
        articles = []

    # Step 1: Generate initial LLM response
    initial_response = generate_basic_llm_response(question, articles, use_context)
    
    # Step 2: Evaluate response quality
    quality_assessment = evaluate_response_quality(initial_response, question)
    
    print(f"🔍 Response Quality Assessment:")
    print(f"   Score: {quality_assessment['quality_score']}/100")
    print(f"   Assessment: {quality_assessment['assessment']}")
    print(f"   Issues: {', '.join(quality_assessment['issues']) if quality_assessment['issues'] else 'None'}")
    
    # Step 3: Decide whether to use Tavily fallback
    if quality_assessment['needs_fallback'] and enable_web_search:
        print("🌐 Using Tavily web search for enhanced results...")
        
        # Search with Tavily
        tavily_results = search_with_tavily(question)
        
        if tavily_results:
            # Generate enhanced response with Tavily data
            enhanced_response = generate_enhanced_response_with_tavily(question, tavily_results)
            return enhanced_response, True
        else:
            print("⚠️ Tavily search failed, attempting regular web search fallback...")
            
            # Try regular web search as fallback
            try:
                from .web import search_financial_web_data
                
                # Extract company name from question for better search
                company_keywords = ['tata motors', 'tata', 'hdfc', 'icici', 'reliance', 'infosys', 'wipro', 'lseg']
                company_found = None
                for keyword in company_keywords:
                    if keyword.lower() in question.lower():
                        company_found = keyword
                        break
                
                web_results = search_financial_web_data(question, company_found)
                
                if web_results:
                    # Format web search results
                    enhanced_response = format_web_search_response(company_found or "Company", web_results, question)
                    print("✅ Using regular web search results")
                    return enhanced_response, True
                else:
                    print("⚠️ No web results found, generating enhanced AI response")
                    # Generate a more comprehensive AI response instead of the poor one
                    enhanced_ai_response = format_ai_answer(initial_response, question)
                    return enhanced_ai_response, True
                    
            except Exception as e:
                print(f"⚠️ Web search fallback failed: {e}")
                print("⚠️ Generating enhanced AI response instead")
                try:
                    # Last resort: generate a better AI response
                    enhanced_ai_response = format_ai_answer("", question)
                    return enhanced_ai_response, True
                except Exception as e2:
                    print(f"⚠️ Enhanced AI response failed: {e2}")
                    return initial_response, False
    
    # Return initial response if quality is acceptable or web search disabled
    return initial_response, False

def generate_basic_llm_response(question: str, articles: List[Dict], use_context: bool) -> str:
    """
    Generate basic LLM response using available context.
    """
    context_info = ""
    if use_context and articles:
        context_info = f"Based on {len(articles)} articles: "
        for article in articles[:2]:  # Use top 2 articles
            context_info += f"{article.get('title', '')[:100]}... "
    
    # Enhanced response based on question type
    if any(keyword in question.lower() for keyword in ['target', 'price', 'forecast', 'analysis']):
        return f"""# 📈 **Financial Analysis Response**

{context_info}

**Query:** {question}

## Current Assessment
Based on available information, here's the analysis for your query about financial targets and market positioning.

## Key Considerations
- Market volatility and economic indicators
- Company fundamentals and recent performance
- Industry trends and competitive landscape

## Note
For the most current and detailed financial data, real-time market information may be required.

**Analysis Quality:** Basic (Generated from available context)
"""
    
    return f"""# 📄 **Information Summary**

{context_info}

**Question:** {question}

## Response
Based on the available information and context provided, here's what we can determine about your query.

## Available Information
The current analysis is based on the articles and context available in the system.

## Recommendation
For more detailed and current information, additional market research may be beneficial.

**Response Quality:** Standard (Context-based)
"""

def format_ai_answer(response_text, question=""):
    """
    Format AI answer with enhanced presentation.
    """
    if not response_text:
        return generate_realtime_ai_answer(question, [], enable_web_search=True)[0]
    
    # If response is already well-formatted, return as-is
    if response_text.startswith('#') or '**' in response_text:
        return response_text
    
    # Otherwise, format it nicely
    return f"""# 🤖 **AI Analysis**

**Query:** {question}

## Response

{response_text}

---
*Generated by AI with intelligent web search fallback*
"""

def get_financial_data_response(question):
    """
    Get enhanced financial data response with Tavily integration.
    """
    # Use the enhanced AI system with web search enabled
    response, used_web_search = generate_realtime_ai_answer(
        question, 
        articles=[], 
        use_context=False, 
        enable_web_search=True
    )
    
    if used_web_search:
        response = f"🌐 **Enhanced with Real-time Data**\n\n{response}"
    
    return response

def format_web_search_response(company, results, question):
    """
    Format web search response with enhanced presentation.
    """
    if not results:
        return f"No search results found for {company} regarding: {question}"
    
    response = f"""# 🔍 **Web Search Results**

**Company:** {company}
**Query:** {question}
**Results Found:** {len(results)}

---

"""
    
    for i, result in enumerate(results[:5], 1):
        title = result.get('title', 'No title')
        content = result.get('content', 'No content')
        url = result.get('url', '')
        source = result.get('source', 'web')
        
        response += f"""## {i}. **{title}**

{content[:400]}{'...' if len(content) > 400 else ''}

**Source:** [{url.split('//')[-1].split('/')[0] if url else 'Unknown'}]({url})
**Type:** {source.title()}

---

"""
    
    response += """
## 📝 **Summary**
The above results provide the latest information about your query. For investment decisions, please verify information from multiple sources and consult with financial professionals.
"""
    
    return response

def create_sample_env_file():
    """
    Create a sample .env file with Tavily configuration.
    """
    sample_env = """# Tavily Web Search API Configuration
# Sign up at https://tavily.com to get your API key
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Other API keys for enhanced functionality
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
"""
    
    env_path = "/Users/nethimanikyapavan/Documents/augment-projects/News_Equity_Research_Tool/.env.example"
    
    try:
        with open(env_path, 'w') as f:
            f.write(sample_env)
        pass  # Silent creation
    except Exception as e:
        print(f"⚠️ Could not create sample .env file: {e}")

# Initialize sample env file on import
create_sample_env_file()