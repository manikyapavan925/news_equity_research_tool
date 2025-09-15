#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import re

def debug_reuters_structure():
    url = 'https://www.reuters.com/world/uk/lseg-rolls-outs-blockchain-based-platform-private-funds-2025-09-15/'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("=== DEBUGGING REUTERS HTML STRUCTURE ===")
        
        # Look for different content containers
        selectors_to_try = [
            'main',
            '[data-module="ArticleBody"]',
            '.ArticleBodyWrapper',
            '[data-testid="Body"]',
            '[data-testid="paragraph"]',
            '.StandardArticleBody_body',
            '.text__text',
            '.article-body',
            'article',
            '.content',
            'p'
        ]
        
        for selector in selectors_to_try:
            elements = soup.select(selector)
            print(f"\n--- Testing selector: {selector} ---")
            print(f"Found {len(elements)} elements")
            
            if elements:
                for i, elem in enumerate(elements[:3]):  # Show first 3 matches
                    text = elem.get_text(separator=' ', strip=True)[:200]
                    print(f"Element {i+1}: {text}...")
        
        # Also check for JSON-LD structured data
        print(f"\n--- Checking for JSON-LD structured data ---")
        json_scripts = soup.find_all('script', type='application/ld+json')
        print(f"Found {len(json_scripts)} JSON-LD scripts")
        
        for i, script in enumerate(json_scripts[:2]):
            print(f"Script {i+1}: {script.string[:200] if script.string else 'No content'}...")
            
        # Look for specific Reuters patterns
        print(f"\n--- Looking for specific text patterns ---")
        all_text = soup.get_text()
        
        # Search for LSEG-related content
        lseg_matches = re.findall(r'[^.]*LSEG[^.]*\.', all_text, re.IGNORECASE)
        print(f"LSEG mentions: {len(lseg_matches)}")
        for match in lseg_matches[:3]:
            print(f"  - {match.strip()}")
            
        blockchain_matches = re.findall(r'[^.]*blockchain[^.]*\.', all_text, re.IGNORECASE)
        print(f"Blockchain mentions: {len(blockchain_matches)}")
        for match in blockchain_matches[:3]:
            print(f"  - {match.strip()}")
            
        platform_matches = re.findall(r'[^.]*Digital Markets Infrastructure[^.]*\.', all_text, re.IGNORECASE)
        print(f"Digital Markets Infrastructure mentions: {len(platform_matches)}")
        for match in platform_matches[:3]:
            print(f"  - {match.strip()}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_reuters_structure()