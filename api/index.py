from http.server import BaseHTTPRequestHandler
import json
import urllib.parse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL path
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/api' or parsed_path.path == '/api/':
            # API endpoint
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "name": "News Equity Research Tool",
                "version": "1.0.0",
                "description": "AI-powered financial news analysis assistant",
                "status": "active",
                "features": [
                    "Financial news article analysis",
                    "Company-specific research",
                    "Sentiment analysis",
                    "Interactive Q&A interface",
                    "Multiple data sources support"
                ],
                "tech_stack": [
                    "Streamlit",
                    "LangChain",
                    "HuggingFace Transformers",
                    "PyTorch",
                    "Sentence Transformers"
                ],
                "deployment_info": {
                    "current_platform": "Streamlit Cloud",
                    "recommended_platform": "Streamlit Cloud",
                    "repository": "https://github.com/manikyapavan925/news_equity_research_tool",
                    "streamlit_deploy": "https://share.streamlit.io"
                }
            }
            
            self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            # Landing page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            html_content = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>News Equity Research Tool</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .container { 
                        max-width: 900px;
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }
                    .header {
                        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                        color: white;
                        text-align: center;
                        padding: 40px 20px;
                    }
                    .header h1 { font-size: 2.5em; margin-bottom: 10px; }
                    .header p { font-size: 1.2em; opacity: 0.9; }
                    .content { padding: 40px; }
                    .features {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin: 30px 0;
                    }
                    .feature {
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 4px solid #007bff;
                    }
                    .feature h3 { color: #007bff; margin-bottom: 10px; }
                    .buttons {
                        text-align: center;
                        margin: 40px 0;
                    }
                    .btn { 
                        display: inline-block;
                        padding: 15px 30px;
                        margin: 10px;
                        background: #007bff;
                        color: white;
                        text-decoration: none;
                        border-radius: 8px;
                        font-weight: 600;
                        transition: all 0.3s ease;
                        box-shadow: 0 4px 15px rgba(0,123,255,0.3);
                    }
                    .btn:hover {
                        background: #0056b3;
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0,123,255,0.4);
                    }
                    .btn.secondary {
                        background: #6c757d;
                        box-shadow: 0 4px 15px rgba(108,117,125,0.3);
                    }
                    .btn.secondary:hover {
                        background: #545b62;
                        box-shadow: 0 6px 20px rgba(108,117,125,0.4);
                    }
                    .tech-stack {
                        background: #e7f3ff;
                        padding: 20px;
                        border-radius: 10px;
                        margin: 20px 0;
                    }
                    .tech-stack h3 { color: #0066cc; margin-bottom: 15px; }
                    .tech-tags {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 10px;
                    }
                    .tech-tag {
                        background: #007bff;
                        color: white;
                        padding: 5px 12px;
                        border-radius: 20px;
                        font-size: 0.9em;
                    }
                    .api-info {
                        background: #f8f9fa;
                        border: 1px solid #dee2e6;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 20px 0;
                        font-family: 'Courier New', monospace;
                    }
                    .footer {
                        text-align: center;
                        padding: 20px;
                        background: #f8f9fa;
                        color: #6c757d;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📰 News Equity Research Tool</h1>
                        <p>AI-Powered Financial News Analysis Assistant</p>
                    </div>
                    
                    <div class="content">
                        <div class="features">
                            <div class="feature">
                                <h3>🔍 Smart Analysis</h3>
                                <p>Analyze financial news articles with AI-powered insights and sentiment analysis.</p>
                            </div>
                            <div class="feature">
                                <h3>🏢 Company Research</h3>
                                <p>Get comprehensive research on specific companies and their market performance.</p>
                            </div>
                            <div class="feature">
                                <h3>💬 Interactive Q&A</h3>
                                <p>Ask questions about loaded articles and get intelligent responses.</p>
                            </div>
                            <div class="feature">
                                <h3>📊 Multi-Source Data</h3>
                                <p>Support for multiple financial news sources and data feeds.</p>
                            </div>
                        </div>
                        
                        <div class="tech-stack">
                            <h3>🛠 Technology Stack</h3>
                            <div class="tech-tags">
                                <span class="tech-tag">Streamlit</span>
                                <span class="tech-tag">LangChain</span>
                                <span class="tech-tag">HuggingFace</span>
                                <span class="tech-tag">PyTorch</span>
                                <span class="tech-tag">Transformers</span>
                                <span class="tech-tag">Python</span>
                            </div>
                        </div>
                        
                        <div class="buttons">
                            <a href="https://share.streamlit.io" class="btn">🚀 Deploy on Streamlit Cloud</a>
                            <a href="https://github.com/manikyapavan925/news_equity_research_tool" class="btn secondary">📋 View Source Code</a>
                        </div>
                        
                        <div class="api-info">
                            <h4>API Endpoint:</h4>
                            <p>GET <strong>/api</strong> - Get application information and deployment details</p>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>© 2025 News Equity Research Tool | Built with ❤️ for financial analysis</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            self.wfile.write(html_content.encode())

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "message": "POST requests received",
            "note": "This is a landing page API. Deploy the full app on Streamlit Cloud for complete functionality.",
            "streamlit_cloud": "https://share.streamlit.io"
        }
        
        self.wfile.write(json.dumps(response).encode())
