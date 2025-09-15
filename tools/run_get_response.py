import json
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from streamlit_app import get_intelligent_response

if __name__ == '__main__':
    q = "Comprehensive Analysis: what is the last traded price of lseg today?"
    print('running get_intelligent_response...')
    res = get_intelligent_response(q)
    print('--- FULL RESPONSE KEYS ---')
    print(list(res.keys()))
    print('\n--- FULL RESPONSE DUMP ---')
    try:
        print(json.dumps(res, indent=2)[:4000])
    except Exception:
        print(str(res)[:4000])
