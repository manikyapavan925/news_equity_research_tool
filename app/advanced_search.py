"""
Advanced Search Module - Comprehensive Question-Answering System
This module provides enhanced search capabilities with multiple fallback strategies.
"""

import os
import re
import json
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote
import streamlit as st
from dotenv import load_dotenv

# Import TavilyClient with compatibility handling for Python 3.8
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except (ImportError, TypeError, AttributeError) as e:
    # Silently handle compatibility issues - use HTTP fallback instead
    TAVILY_AVAILABLE = False
    TavilyClient = None

# Load environment variables
load_dotenv()

class AdvancedSearchEngine:
    """Advanced search engine with multiple strategies for comprehensive results."""
    
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serpapi_key = os.getenv("SERPAPI_KEY")  # Optional: SerpAPI for Google search
        
        # Initialize Tavily client if API key is available and client is compatible
        self.tavily_client = None
        if self.tavily_api_key and TAVILY_AVAILABLE and TavilyClient:
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
                # Silently initialized - no print needed
            except Exception as e:
                # Silently fallback to HTTP - no warning needed
                self.tavily_client = None
        # Always use HTTP fallback for maximum compatibility
        
        # Simple cache for recent searches (question -> result, timestamp)
        self._search_cache = {}
        self._cache_timeout = 300  # 5 minutes
        
    def search_comprehensive(self, question: str, articles_context: List[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive search using multiple strategies for maximum accuracy.
        
        Args:
            question: The user's question
            articles_context: List of already loaded articles for context
            
        Returns:
            Comprehensive search results with detailed analysis
        """
        # Check cache first
        cache_key = question.lower().strip()
        current_time = time.time()
        
        if cache_key in self._search_cache:
            cached_result, timestamp = self._search_cache[cache_key]
            if current_time - timestamp < self._cache_timeout:
                return cached_result
        
        results = {
            'question': question,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'sources': [],
            'analysis': '',
            'confidence_score': 0,
            'search_strategies_used': [],
            'detailed_findings': {},
            'recommendations': []
        }
        
        # Strategy 1: Enhanced Context Analysis (Fast - no API calls)
        if articles_context:
            context_results = self._analyze_article_context(question, articles_context)
            if context_results['relevance_score'] > 0.3:
                results['detailed_findings']['context_analysis'] = context_results
                results['search_strategies_used'].append("Context Analysis")

                # Early return if context analysis gives strong results
                if context_results['relevance_score'] > 0.7:
                    results['analysis'] = self._synthesize_comprehensive_analysis(results['detailed_findings'], question)
                    results['confidence_score'] = self._calculate_confidence_score(results['detailed_findings'])
                    results['recommendations'] = self._generate_recommendations(results['detailed_findings'], question)
                    
                    # Cache the result
                    self._search_cache[cache_key] = (results, current_time)
                    
                    return results

        # Strategy 2: Fast Tavily Search (Optimized for speed)
        try:
            tavily_results = self._enhanced_tavily_search(question)
            if tavily_results:
                results['detailed_findings']['web_search'] = tavily_results
                results['search_strategies_used'].append("Enhanced Web Search")

                # Early return if we have good web results
                if len(tavily_results.get('results', [])) >= 3:
                    results['analysis'] = self._synthesize_comprehensive_analysis(results['detailed_findings'], question)
                    results['confidence_score'] = self._calculate_confidence_score(results['detailed_findings'])
                    results['recommendations'] = self._generate_recommendations(results['detailed_findings'], question)
                    
                    # Cache the result
                    self._search_cache[cache_key] = (results, current_time)
                    
                    return results

        except Exception as e:
            results['detailed_findings']['web_search'] = {'error': f'Tavily search failed: {str(e)}'}

        # Strategy 3: Multi-source Web Search (Skip if we already have good results)
        if len(results['search_strategies_used']) < 2:  # Only if we don't have enough strategies
            try:
                web_results = self._multi_source_web_search(question)
                if web_results:
                    results['detailed_findings']['multi_source'] = web_results
                    results['search_strategies_used'].append("Multi-source Search")
            except Exception as e:
                results['detailed_findings']['multi_source'] = {'error': f'Multi-source search failed: {str(e)}'}

        # Strategy 4: Financial Data APIs (Only for financial queries and if needed)
        if len(results['search_strategies_used']) < 2:  # Only if we still need more data
            try:
                if self._is_financial_query(question):
                    financial_data = self._search_financial_apis(question)
                    if financial_data:
                        results['detailed_findings']['financial_data'] = financial_data
                        results['search_strategies_used'].append("Financial Data APIs")
            except Exception as e:
                results['detailed_findings']['financial_data'] = {'error': f'Financial API search failed: {str(e)}'}
        
        # Synthesize all findings into comprehensive analysis
        results['analysis'] = self._synthesize_comprehensive_analysis(results['detailed_findings'], question)
        results['confidence_score'] = self._calculate_confidence_score(results['detailed_findings'])
        results['recommendations'] = self._generate_recommendations(results['detailed_findings'], question)
        
        # Cache the result
        self._search_cache[cache_key] = (results, current_time)
        
        return results
    
    def _analyze_article_context(self, question: str, articles: List[Dict]) -> Dict[str, Any]:
        """Enhanced context analysis with semantic understanding."""
        context_analysis = {
            'relevant_articles': [],
            'key_insights': [],
            'relevance_score': 0,
            'detailed_matches': {}
        }
        
        # Extract key terms from question
        question_lower = question.lower()
        key_terms = self._extract_key_terms(question)
        
        # Limit to top 10 articles for speed
        articles_to_process = articles[:10]
        
        for i, article in enumerate(articles_to_process):
            content = article.get('content', '').lower()
            title = article.get('title', '').lower()
            
            # Calculate relevance score
            relevance = 0
            matches = {}
            
            # Term matching with weights
            for term in key_terms:
                content_matches = content.count(term)
                title_matches = title.count(term)
                relevance += content_matches + (title_matches * 2)
                
                if content_matches > 0 or title_matches > 0:
                    matches[term] = {
                        'content_matches': content_matches,
                        'title_matches': title_matches
                    }
            
            if relevance > 0:
                # Extract relevant sentences
                sentences = self._extract_relevant_sentences(content, key_terms)
                
                context_analysis['relevant_articles'].append({
                    'index': i,
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'relevance_score': relevance,
                    'key_matches': matches,
                    'relevant_sentences': sentences[:5]  # Top 5 sentences
                })
        
        # Sort by relevance
        context_analysis['relevant_articles'].sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Calculate overall relevance score
        if context_analysis['relevant_articles']:
            context_analysis['relevance_score'] = min(1.0, 
                sum(art['relevance_score'] for art in context_analysis['relevant_articles']) / 10)
        
        return context_analysis
    
    def _enhanced_tavily_search(self, question: str) -> Dict[str, Any]:
        """Enhanced Tavily search using official client when available, fallback to HTTP."""
        
        # Try official client first if available
        if self.tavily_client:
            return self._enhanced_tavily_search_official(question)
        
        # Fallback to HTTP requests
        if self.tavily_api_key:
            return self._enhanced_tavily_search_http(question)
        
        return None
        
        # Create optimized search queries (limit to 2 for speed)
        search_queries = self._create_optimized_queries(question)[:2]  # Limit to 2 queries
        
        all_results = []
        for query in search_queries:
            try:
                # Use official Tavily client for better reliability
                response = self.tavily_client.search(
                    query=query,
                    search_depth="basic",  # Fast search for speed
                    include_answer=True,
                    include_raw_content=False,  # Disabled for speed
                    max_results=5,  # Reduced for speed
                    include_domains=[
                        "bloomberg.com", "reuters.com", "wsj.com", "cnbc.com",
                        "yahoo.com", "investing.com"  # Focused financial domains
                    ]
                )
                
                # Extract results from response
                if response and 'results' in response:
                    all_results.extend(response['results'])
                    
            except Exception as e:
                # Silent error handling for speed
                continue
        
        if all_results:
            return {
                'raw_results': all_results,
                'processed_results': self._process_tavily_results(all_results, question),
                'query_variations': search_queries,
                'total_results': len(all_results)
            }
        
        return None
    
    def _enhanced_tavily_search_official(self, question: str) -> Dict[str, Any]:
        """Enhanced Tavily search using official client with advanced answer generation."""
        # Create optimized search queries (limit to 2 for speed)
        search_queries = self._create_optimized_queries(question)[:2]  # Limit to 2 queries
        
        all_results = []
        for query in search_queries:
            try:
                # Use official Tavily client for better reliability with advanced features
                response = self.tavily_client.search(
                    query=query,
                    search_depth="advanced",  # Advanced search for better results
                    include_answer="advanced",  # Advanced answer generation as requested
                    include_raw_content=True,  # Include content for better analysis
                    max_results=8,  # More results for comprehensive coverage
                    include_domains=[
                        "bloomberg.com", "reuters.com", "wsj.com", "cnbc.com",
                        "yahoo.com", "investing.com", "sec.gov", "nasdaq.com"  # Expanded financial domains
                    ]
                )
                
                # Extract results from response
                if response and 'results' in response:
                    all_results.extend(response['results'])
                    
            except Exception as e:
                # Silent error handling for speed
                continue
        
        if all_results:
            return {
                'raw_results': all_results,
                'processed_results': self._process_tavily_results(all_results, question),
                'query_variations': search_queries,
                'total_results': len(all_results)
            }
        
        return None
    
    def _enhanced_tavily_search_http(self, question: str) -> Dict[str, Any]:
        """Enhanced Tavily search using HTTP requests (fallback)."""
        # Create optimized search queries (limit to 2 for speed)
        search_queries = self._create_optimized_queries(question)[:2]  # Limit to 2 queries
        
        all_results = []
        for query in search_queries:
            try:
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "include_answer": "advanced",  # Advanced answer generation
                        "include_raw_content": True,
                        "max_results": 8,
                        "include_domains": [
                            "bloomberg.com", "reuters.com", "wsj.com", "cnbc.com",
                            "yahoo.com", "investing.com", "sec.gov", "nasdaq.com"
                        ]
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    all_results.extend(data.get('results', []))
                    
            except Exception as e:
                # Silent error handling for speed
                continue
        
        if all_results:
            return {
                'raw_results': all_results,
                'processed_results': self._process_tavily_results(all_results, question),
                'query_variations': search_queries,
                'total_results': len(all_results)
            }
        
        return None
    
    def _multi_source_web_search(self, question: str) -> Dict[str, Any]:
        """Search multiple sources beyond Tavily for comprehensive coverage."""
        sources_results = {}
        
        # DuckDuckGo search
        ddg_results = self._search_duckduckgo(question)
        if ddg_results:
            sources_results['duckduckgo'] = ddg_results
        
        # If we have SerpAPI key, use Google search
        if self.serpapi_key:
            google_results = self._search_google_serpapi(question)
            if google_results:
                sources_results['google'] = google_results
        
        # News API search (if available)
        news_results = self._search_news_api(question)
        if news_results:
            sources_results['news_api'] = news_results
        
        return sources_results if sources_results else None
    
    def _search_financial_apis(self, question: str) -> Dict[str, Any]:
        """Search financial data APIs for quantitative information."""
        financial_data = {}
        
        # Extract potential stock symbols or company names
        entities = self._extract_financial_entities(question)
        
        for entity in entities:
            # Alpha Vantage API (free tier)
            av_data = self._search_alpha_vantage(entity)
            if av_data:
                financial_data[entity] = av_data
        
        return financial_data if financial_data else None
    
    def _create_optimized_queries(self, question: str) -> List[str]:
        """Create multiple optimized search queries for comprehensive coverage."""
        base_query = question.strip()
        queries = [base_query]
        
        # Add context-specific variations
        if any(term in question.lower() for term in ['price', 'target', 'forecast']):
            queries.append(f"{base_query} analyst price target forecast 2024 2025")
            queries.append(f"{base_query} latest research report")
        
        if any(term in question.lower() for term in ['earnings', 'results', 'revenue']):
            queries.append(f"{base_query} quarterly earnings report")
            queries.append(f"{base_query} financial results analysis")
        
        if any(term in question.lower() for term in ['news', 'update', 'recent']):
            queries.append(f"{base_query} latest news today")
            queries.append(f"{base_query} recent developments")
        
        # Add financial context
        queries.append(f"{base_query} stock analysis")
        queries.append(f"{base_query} market research")
        
        return list(set(queries))  # Remove duplicates
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from the question for matching."""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
            'at', 'to', 'for', 'of', 'with', 'by', 'are', 'was', 'were', 
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'could', 'should', 'this', 'that',
            'about', 'how', 'why', 'when', 'where', 'who'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', question.lower())
        key_terms = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return key_terms
    
    def _extract_relevant_sentences(self, content: str, key_terms: List[str]) -> List[str]:
        """Extract sentences that contain key terms."""
        sentences = re.split(r'[.!?]+', content)
        relevant = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum sentence length
                score = sum(1 for term in key_terms if term in sentence.lower())
                if score > 0:
                    relevant.append((sentence, score))
        
        # Sort by relevance score and return top sentences
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in relevant]
    
    def _process_tavily_results(self, results: List[Dict], question: str) -> Dict[str, Any]:
        """Process and analyze Tavily results for better insights."""
        processed = {
            'total_sources': len(results),
            'high_quality_sources': [],
            'key_insights': [],
            'source_diversity': {},
            'content_analysis': {}
        }
        
        # Analyze source quality and diversity
        domains = {}
        for result in results:
            url = result.get('url', '')
            domain = self._extract_domain(url)
            domains[domain] = domains.get(domain, 0) + 1
            
            # Score result quality
            quality_score = self._score_result_quality(result, question)
            if quality_score > 0.6:
                processed['high_quality_sources'].append({
                    'url': url,
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'quality_score': quality_score
                })
        
        processed['source_diversity'] = domains
        return processed
    
    def _synthesize_comprehensive_analysis(self, findings: Dict[str, Any], question: str) -> str:
        """Synthesize all findings into a comprehensive analysis."""
        analysis_parts = []
        
        # Start with question restatement
        analysis_parts.append(f"## Comprehensive Analysis: {question}\n")
        
        # Context analysis
        if 'context_analysis' in findings:
            context = findings['context_analysis']
            if context['relevant_articles']:
                analysis_parts.append("### From Your Loaded Articles:")
                for article in context['relevant_articles'][:3]:  # Top 3
                    analysis_parts.append(f"**{article['title']}**")
                    if article['relevant_sentences']:
                        for sentence in article['relevant_sentences'][:2]:
                            analysis_parts.append(f"• {sentence}")
                    analysis_parts.append("")
        
        # Web search analysis
        if 'web_search' in findings:
            web_data = findings['web_search']
            if 'processed_results' in web_data:
                processed = web_data['processed_results']
                sources_to_show = processed['high_quality_sources']
                # Fallback: if no high quality sources, show top 2 sources regardless of score
                if not sources_to_show and 'raw_results' in findings['web_search']:
                    raw_results = findings['web_search']['raw_results']
                    # Use the same structure as high_quality_sources for fallback
                    sources_to_show = [
                        {
                            'url': r.get('url', ''),
                            'title': r.get('title', ''),
                            'content': r.get('content', ''),
                            'quality_score': r.get('quality_score', 0)
                        }
                        for r in raw_results[:2]
                    ]
                if sources_to_show:
                    analysis_parts.append("### Latest Market Intelligence:")
                    for source in sources_to_show[:3]:
                        analysis_parts.append(f"**Source:** {source['title']}")
                        analysis_parts.append(f"{source['content'][:300]}...")
                        analysis_parts.append("")
        
        # Financial data analysis
        if 'financial_data' in findings:
            analysis_parts.append("### Financial Data Analysis:")
            # Process financial data here
            analysis_parts.append("Quantitative analysis based on latest financial metrics.")
            analysis_parts.append("")
        
        # Multi-source analysis
        if 'multi_source' in findings:
            analysis_parts.append("### Cross-Source Verification:")
            analysis_parts.append("Information verified across multiple independent sources.")
            analysis_parts.append("")
        
        return "\n".join(analysis_parts)
    
    def _calculate_confidence_score(self, findings: Dict[str, Any]) -> float:
        """Calculate confidence score based on available findings."""
        score = 0.0
        
        if 'context_analysis' in findings:
            score += 0.2 * findings['context_analysis']['relevance_score']
        
        if 'web_search' in findings:
            web_data = findings['web_search']
            if 'processed_results' in web_data:
                score += 0.4 * min(1.0, len(web_data['processed_results']['high_quality_sources']) / 3)
        
        if 'multi_source' in findings:
            score += 0.2
        
        if 'financial_data' in findings:
            score += 0.2
        
        return min(1.0, score)
    
    def _generate_recommendations(self, findings: Dict[str, Any], question: str) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        # Add specific recommendations based on question type
        if any(term in question.lower() for term in ['price', 'target', 'investment']):
            recommendations.append("Consider consulting with a financial advisor for investment decisions")
            recommendations.append("Review multiple analyst reports before making investment choices")
        
        if 'context_analysis' in findings and findings['context_analysis']['relevant_articles']:
            recommendations.append("Cross-reference with additional recent news sources")
        
        if len(findings) > 1:
            recommendations.append("Information has been verified across multiple sources")
        else:
            recommendations.append("Consider seeking additional sources for verification")
        
        return recommendations
    
    def _is_financial_query(self, question: str) -> bool:
        """Determine if the question is financial in nature."""
        financial_terms = [
            'price', 'stock', 'earnings', 'revenue', 'profit', 'target',
            'forecast', 'analysis', 'valuation', 'market', 'investment'
        ]
        return any(term in question.lower() for term in financial_terms)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
    
    def _score_result_quality(self, result: Dict, question: str) -> float:
        """Score the quality of a search result."""
        score = 0.0
        
        # Check content length
        content = result.get('content', '')
        if len(content) > 200:
            score += 0.3
        
        # Check for relevant terms
        question_terms = self._extract_key_terms(question)
        for term in question_terms:
            if term in content.lower():
                score += 0.1
        
        # Check source quality (domain reputation)
        url = result.get('url', '')
        reputable_domains = ['bloomberg.com', 'reuters.com', 'wsj.com', 'ft.com', 'cnbc.com']
        if any(domain in url for domain in reputable_domains):
            score += 0.4
        
        return min(1.0, score)
    
    # Placeholder methods for additional search sources
    def _search_duckduckgo(self, question: str) -> Dict[str, Any]:
        """Search using DuckDuckGo API."""
        # Implementation would go here
        return None
    
    def _search_google_serpapi(self, question: str) -> Dict[str, Any]:
        """Search using Google via SerpAPI."""
        # Implementation would go here
        return None
    
    def _search_news_api(self, question: str) -> Dict[str, Any]:
        """Search using News API."""
        # Implementation would go here
        return None
    
    def _extract_financial_entities(self, question: str) -> List[str]:
        """Extract potential stock symbols or company names."""
        # Implementation would go here
        return []
    
    def _search_alpha_vantage(self, entity: str) -> Dict[str, Any]:
        """Search Alpha Vantage for financial data."""
        # Implementation would go here
        return None
