"""
Web Search Module - Basic Implementation
This module provides basic web search functionality.
"""

def search_financial_web_data(query, company_name=""):
    """
    Search for financial web data
    """
    # Return mock results for debugging
    mock_results = [
        {
            'title': f'Search Results for: {query}',
            'content': f'This is a mock result for debugging. Query: {query}, Company: {company_name}',
            'url': 'https://example.com/debug',
            'source': 'debug_mode'
        }
    ]

    return mock_results

def parse_real_financial_data_from_web(query, company_name=""):
    """
    Parse financial data from web
    """
    return {
        'success': False,
        'error': 'Web parsing not implemented in debug mode',
        'data': {}
    }

def get_financial_data_response(question):
    """
    Get financial data response
    """
    return f"**Web Search Query:** {question}\n\n⚠️ Web search not fully implemented in debug mode"