"""
Predictive Analysis Module - Deep AI-Powered Insights
Provides forecasting, risk assessment, and advanced financial predictions
"""

import os
import re
import json
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def analyze_sentiment_trend(sentiment_timeline: List[Dict]) -> Dict[str, Any]:
    """
    Analyze sentiment trends over time to predict future direction.
    
    Args:
        sentiment_timeline: List of sentiment scores with timestamps
        
    Returns:
        Trend analysis with direction, strength, and prediction
    """
    if not sentiment_timeline or len(sentiment_timeline) < 2:
        return {
            'trend': 'insufficient_data',
            'direction': 'neutral',
            'strength': 0,
            'prediction': 'Need more data points for trend analysis'
        }
    
    # Extract sentiment scores
    scores = [item['sentiment'] for item in sentiment_timeline]
    
    # Calculate trend
    if len(scores) >= 3:
        recent_avg = sum(scores[-3:]) / 3
        earlier_avg = sum(scores[:-3]) / len(scores[:-3]) if len(scores) > 3 else scores[0]
    else:
        recent_avg = scores[-1]
        earlier_avg = scores[0]
    
    change = recent_avg - earlier_avg
    
    # Determine trend
    if change > 0.1:
        trend = 'improving'
        direction = 'bullish'
        strength = min(change * 10, 1.0)
    elif change < -0.1:
        trend = 'declining'
        direction = 'bearish'
        strength = min(abs(change) * 10, 1.0)
    else:
        trend = 'stable'
        direction = 'neutral'
        strength = 0.5
    
    # Prediction
    if direction == 'bullish':
        prediction = f"Sentiment improving by {abs(change)*100:.1f}%. Positive momentum likely to continue."
    elif direction == 'bearish':
        prediction = f"Sentiment declining by {abs(change)*100:.1f}%. Caution advised."
    else:
        prediction = "Sentiment stable. No significant trend detected."
    
    return {
        'trend': trend,
        'direction': direction,
        'strength': strength,
        'change_percent': change * 100,
        'prediction': prediction,
        'confidence': min(len(scores) / 10, 1.0)  # More data = higher confidence
    }


def predict_price_movement(articles: List[Dict], current_sentiment: float) -> Dict[str, Any]:
    """
    Predict potential price movement based on sentiment and news analysis.
    
    Args:
        articles: List of analyzed articles
        current_sentiment: Current sentiment score (-1 to 1)
        
    Returns:
        Price movement prediction with probabilities
    """
    if not articles:
        return {
            'direction': 'unknown',
            'probability': 0,
            'reasoning': 'Insufficient data for prediction'
        }
    
    # Analyze article content for price indicators
    bullish_signals = 0
    bearish_signals = 0
    
    bullish_keywords = [
        'upgrade', 'beat', 'exceeds', 'growth', 'profit', 'revenue increase',
        'expansion', 'partnership', 'innovation', 'strong', 'positive',
        'outperform', 'buy rating', 'raised target', 'breakthrough'
    ]
    
    bearish_keywords = [
        'downgrade', 'miss', 'decline', 'loss', 'layoff', 'concern',
        'warning', 'weak', 'negative', 'sell rating', 'cut target',
        'investigation', 'lawsuit', 'competition', 'threat'
    ]
    
    for article in articles:
        text = (article.get('title', '') + ' ' + article.get('text', '')).lower()
        
        for keyword in bullish_keywords:
            if keyword in text:
                bullish_signals += 1
        
        for keyword in bearish_keywords:
            if keyword in text:
                bearish_signals += 1
    
    # Calculate prediction
    total_signals = bullish_signals + bearish_signals
    
    if total_signals == 0:
        direction = 'neutral'
        probability = 0.5
        reasoning = "No clear directional signals detected in news flow."
    elif bullish_signals > bearish_signals:
        direction = 'upward'
        probability = min(0.5 + (bullish_signals / (total_signals * 2)), 0.95)
        reasoning = f"Found {bullish_signals} bullish vs {bearish_signals} bearish signals. Positive catalysts dominating."
    else:
        direction = 'downward'
        probability = min(0.5 + (bearish_signals / (total_signals * 2)), 0.95)
        reasoning = f"Found {bearish_signals} bearish vs {bullish_signals} bullish signals. Negative factors present."
    
    # Adjust for sentiment
    if current_sentiment > 0.5 and direction == 'upward':
        probability = min(probability + 0.1, 0.98)
    elif current_sentiment < -0.5 and direction == 'downward':
        probability = min(probability + 0.1, 0.98)
    
    return {
        'direction': direction,
        'probability': probability,
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'reasoning': reasoning,
        'confidence_level': 'high' if total_signals > 5 else 'medium' if total_signals > 2 else 'low'
    }


def assess_investment_risk(articles: List[Dict], sentiment: float) -> Dict[str, Any]:
    """
    Assess investment risk based on news analysis and sentiment.
    
    Args:
        articles: List of analyzed articles
        sentiment: Current sentiment score
        
    Returns:
        Risk assessment with score and recommendations
    """
    risk_factors = []
    risk_score = 0.5  # Start neutral
    
    # Check for high-risk keywords
    high_risk_keywords = [
        'investigation', 'lawsuit', 'fraud', 'scandal', 'bankruptcy',
        'warning', 'recall', 'regulatory', 'fine', 'penalty'
    ]
    
    medium_risk_keywords = [
        'competition', 'decline', 'weak', 'concern', 'challenge',
        'uncertainty', 'volatility', 'downgrade'
    ]
    
    for article in articles:
        text = (article.get('title', '') + ' ' + article.get('text', '')).lower()
        
        for keyword in high_risk_keywords:
            if keyword in text:
                risk_score += 0.15
                risk_factors.append(f"High risk: {keyword} detected")
        
        for keyword in medium_risk_keywords:
            if keyword in text:
                risk_score += 0.05
                risk_factors.append(f"Medium risk: {keyword} detected")
    
    # Adjust for sentiment
    if sentiment < -0.3:
        risk_score += 0.1
        risk_factors.append("Negative sentiment increases risk")
    elif sentiment > 0.3:
        risk_score -= 0.1
        risk_factors.append("Positive sentiment reduces risk")
    
    # Cap risk score
    risk_score = min(max(risk_score, 0), 1)
    
    # Determine risk level
    if risk_score < 0.3:
        risk_level = 'Low'
        recommendation = 'Favorable conditions for investment consideration'
    elif risk_score < 0.6:
        risk_level = 'Medium'
        recommendation = 'Moderate risk. Diversification recommended'
    else:
        risk_level = 'High'
        recommendation = 'Elevated risk detected. Caution advised'
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_factors': risk_factors[:5],  # Top 5
        'recommendation': recommendation
    }


def generate_deep_insights(articles: List[Dict], question: str) -> Dict[str, Any]:
    """
    Generate comprehensive deep analysis combining multiple factors.
    
    Args:
        articles: List of analyzed articles
        question: User's question for context
        
    Returns:
        Deep insights with predictions, risks, and recommendations
    """
    if not articles:
        return {
            'summary': 'Insufficient data for deep analysis',
            'predictions': [],
            'risks': [],
            'opportunities': [],
            'recommendations': []
        }
    
    insights = {
        'summary': '',
        'predictions': [],
        'risks': [],
        'opportunities': [],
        'recommendations': [],
        'key_metrics': {}
    }
    
    # Extract key themes
    themes = extract_key_themes(articles)
    
    # Generate predictions
    insights['predictions'] = [
        f"Based on {len(articles)} sources, {themes['dominant_theme']} is the main narrative",
        f"Sentiment trajectory: {themes['sentiment_direction']}",
        f"Market attention: {themes['attention_level']}"
    ]
    
    # Identify risks
    risk_keywords = ['concern', 'challenge', 'risk', 'threat', 'warning', 'decline']
    for article in articles:
        text = article.get('text', '').lower()
        for keyword in risk_keywords:
            if keyword in text:
                # Extract sentence with risk keyword
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        insights['risks'].append(sentence.strip().capitalize())
                        break
                break
    
    # Identify opportunities
    opp_keywords = ['growth', 'expansion', 'opportunity', 'innovation', 'partnership', 'increase']
    for article in articles:
        text = article.get('text', '').lower()
        for keyword in opp_keywords:
            if keyword in text:
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        insights['opportunities'].append(sentence.strip().capitalize())
                        break
                break
    
    # Generate recommendations
    insights['recommendations'] = [
        f"Monitor {themes['key_entities']} for updates",
        f"Track sentiment changes - current direction: {themes['sentiment_direction']}",
        "Diversify across multiple sources for comprehensive view"
    ]
    
    # Summary
    insights['summary'] = f"Analysis of {len(articles)} articles reveals {themes['dominant_theme']} as primary focus. " \
                          f"Sentiment is {themes['sentiment_direction']}. " \
                          f"{len(insights['opportunities'])} opportunities and {len(insights['risks'])} risks identified."
    
    return insights


def extract_key_themes(articles: List[Dict]) -> Dict[str, str]:
    """Extract dominant themes from articles."""
    theme_keywords = {
        'growth': ['growth', 'expansion', 'increase', 'rise'],
        'innovation': ['innovation', 'technology', 'AI', 'digital', 'transformation'],
        'financial': ['earnings', 'revenue', 'profit', 'financial', 'results'],
        'strategic': ['partnership', 'acquisition', 'merger', 'deal', 'investment'],
        'regulatory': ['regulation', 'compliance', 'legal', 'policy'],
        'market': ['market', 'competition', 'share', 'industry']
    }
    
    theme_counts = {theme: 0 for theme in theme_keywords}
    
    for article in articles:
        text = (article.get('title', '') + ' ' + article.get('text', '')).lower()
        for theme, keywords in theme_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    theme_counts[theme] += 1
    
    dominant_theme = max(theme_counts, key=theme_counts.get)
    
    # Extract entities (companies, people)
    entities = []
    for article in articles:
        title = article.get('title', '')
        # Simple entity extraction from title
        words = title.split()
        for word in words:
            if word[0].isupper() and len(word) > 3:
                entities.append(word)
    
    return {
        'dominant_theme': dominant_theme,
        'sentiment_direction': 'improving',  # Placeholder
        'attention_level': 'high' if len(articles) > 3 else 'moderate',
        'key_entities': ', '.join(set(entities[:3]))
    }


def predict_target_price(articles: List[Dict], current_price: Optional[float] = None) -> Dict[str, Any]:
    """
    Predict target price range based on analyst sentiment and news.
    
    Args:
        articles: List of articles
        current_price: Current stock price if available
        
    Returns:
        Price target prediction with range
    """
    # Extract price targets from articles
    price_targets = []
    
    for article in articles:
        text = article.get('text', '')
        # Look for price target patterns
        patterns = [
            r'target price[:\s]+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'price target[:\s]+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'target of\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    price = float(match.replace(',', ''))
                    if 10 < price < 10000:  # Reasonable range
                        price_targets.append(price)
                except:
                    pass
    
    if not price_targets:
        return {
            'predicted_target': None,
            'range_low': None,
            'range_high': None,
            'confidence': 'low',
            'basis': 'No analyst price targets found in articles'
        }
    
    # Calculate consensus
    avg_target = sum(price_targets) / len(price_targets)
    min_target = min(price_targets)
    max_target = max(price_targets)
    
    upside = None
    if current_price:
        upside = ((avg_target - current_price) / current_price) * 100
    
    return {
        'predicted_target': round(avg_target, 2),
        'range_low': round(min_target, 2),
        'range_high': round(max_target, 2),
        'num_analysts': len(price_targets),
        'upside_percent': round(upside, 1) if upside else None,
        'confidence': 'high' if len(price_targets) > 3 else 'medium',
        'basis': f'Consensus of {len(price_targets)} analyst targets'
    }
