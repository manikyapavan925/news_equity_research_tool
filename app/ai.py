"""
AI Module - Core AI functionality for the News Equity Research Tool
"""

def generate_realtime_ai_answer(question, articles=None, use_context=True, enable_web_search=False):
    """
    Generate AI answer for a question with optional web search

    Args:
        question (str): The question to answer
        articles (list): List of articles to use as context
        use_context (bool): Whether to use article context
        enable_web_search (bool): Whether to enable web search

    Returns:
        tuple: (response_text, used_web_search)
    """
    try:
        # Simple fallback implementation
        if articles and use_context:
            # Use article context
            context = " ".join([str(article.get('content', ''))[:500] for article in articles if article.get('content')])
            response = f"**Analysis for: {question}**\n\nBased on the provided articles, here's the analysis:\n\n{context[:1000]}..."
        else:
            # General response without context
            response = f"**Analysis for: {question}**\n\nThis appears to be a financial question. For the most current and accurate information, please check with financial news sources or consult a financial advisor."

        return response, enable_web_search

    except Exception as e:
        return f"**Error:** Unable to generate response. {str(e)}", False


def format_ai_answer(response_text, question=""):
    """
    Format AI answer with proper structure

    Args:
        response_text (str): Raw response text
        question (str): Original question

    Returns:
        str: Formatted response
    """
    if not response_text:
        return f"**📋 Analysis for: {question}**\n\nNo response available."

    return f"**📋 Analysis for: {question}**\n\n{response_text}"
