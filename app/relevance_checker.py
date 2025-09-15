"""
Relevance Checker Module - Basic Implementation
This module provides basic relevance checking functionality.
"""

def is_question_relevant_to_articles(question, articles):
    """
    Check if question is relevant to articles
    """
    if not articles:
        return False, {'relevance_score': 0.0, 'reason': 'No articles provided'}

    # Basic relevance check
    question_lower = question.lower()
    article_text = ' '.join([str(article.get('content', '')) + str(article.get('title', '')) for article in articles]).lower()

    # Simple keyword matching
    relevant_keywords = ['price', 'target', 'analysis', 'stock', 'company', 'financial']
    score = sum(1 for keyword in relevant_keywords if keyword in question_lower and keyword in article_text)

    is_relevant = score > 0

    return is_relevant, {
        'relevance_score': min(score / len(relevant_keywords), 1.0),
        'reason': f'Found {score} relevant keywords'
    }


def get_relevance_explanation(relevance_analysis):
    """
    Generate a human-readable explanation of relevance analysis

    Args:
        relevance_analysis (dict): Analysis result from is_question_relevant_to_articles

    Returns:
        str: Formatted explanation
    """
    if not relevance_analysis:
        return "No relevance analysis available."

    score = relevance_analysis.get('relevance_score', 0.0)
    reason = relevance_analysis.get('relevance_score', 'Unknown')

    # Format the explanation
    explanation = f"**Relevance Analysis:**\n\n"
    explanation += f"**Relevance Score:** {score:.2f}/1.0\n"
    explanation += f"**Analysis:** {reason}\n\n"

    if score > 0.5:
        explanation += "✅ **High Relevance:** Question appears to be well-matched to the available articles.\n"
    elif score > 0.2:
        explanation += "⚠️ **Moderate Relevance:** Question has some connection to the articles but may need additional context.\n"
    else:
        explanation += "❌ **Low Relevance:** Question may not be directly answerable from the current articles.\n"

    return explanation