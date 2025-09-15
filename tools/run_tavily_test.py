#!/usr/bin/env python3
"""Quick test runner: force HTTP Tavily search with a short timeout and print debug.

This is a test-only script to avoid full web search paths and get deterministic behavior.
"""
import time
import json
import requests
import sys
from pathlib import Path
# Ensure repo root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.advanced_search import AdvancedSearchEngine

# Short wrapper to force timeout and to restore later
orig_post = requests.post

def post_with_short_timeout(*args, **kwargs):
    kwargs.setdefault('timeout', 5)
    return orig_post(*args, **kwargs)

requests.post = post_with_short_timeout

try:
    engine = AdvancedSearchEngine()
    # Force HTTP fallback and disable official client if present
    engine.tavily_client = None

    q = "Comprehensive Analysis: what is the last traded price of lseg today?"
    start = time.time()
    try:
        res = engine._enhanced_tavily_search_http(q)
    except Exception as e:
        # Ensure debug is captured
        res = None
        print('Exception during tavily http call:', str(e))

    end = time.time()

    out = {
        'elapsed_s': end - start,
        'tavily_result': res,
        '_last_tavily_debug': getattr(engine, '_last_tavily_debug', None)
    }

    print(json.dumps(out, default=str, indent=2))
finally:
    # Restore requests.post
    requests.post = orig_post
