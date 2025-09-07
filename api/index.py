from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
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
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 50px auto; 
                    padding: 20px;
                    line-height: 1.6;
                }
                .container { 
                    text-align: center; 
                    background: #f8f9fa; 
                    padding: 40px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .btn { 
                    display: inline-block; 
                    padding: 12px 24px; 
                    background: #007bff; 
                    color: white; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    margin: 10px;
                    transition: background 0.3s;
                }
                .btn:hover { background: #0056b3; }
                .info { 
                    background: #e7f3ff; 
                    padding: 20px; 
                    border-radius: 5px; 
                    margin: 20px 0;
                    text-align: left;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📰 News Equity Research Tool</h1>
                <p>An AI-powered financial news analysis assistant built with Streamlit, LangChain, and HuggingFace.</p>
                
                <div class="info">
                    <h3>🚀 Deployment Options:</h3>
                    <ul>
                        <li><strong>Streamlit Cloud:</strong> Best for Streamlit apps (Recommended)</li>
                        <li><strong>Heroku:</strong> Good for production with more resources</li>
                        <li><strong>Railway:</strong> Excellent for ML/AI applications</li>
                    </ul>
                </div>
                
                <a href="https://share.streamlit.io" class="btn">Deploy on Streamlit Cloud</a>
                <a href="https://github.com/manikyapavan925/news_equity_research_tool" class="btn">View Source Code</a>
                
                <div class="info">
                    <h3>📝 Features:</h3>
                    <ul>
                        <li>Financial news article analysis</li>
                        <li>Company-specific research</li>
                        <li>Sentiment analysis</li>
                        <li>Interactive Q&A interface</li>
                        <li>Multiple data sources support</li>
                    </ul>
                </div>
                
                <p><em>Note: This application requires significant computational resources and is best deployed on platforms optimized for ML workloads.</em></p>
            </div>
        </body>
        </html>
        """
        
        self.wfile.write(html_content.encode())

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "message": "News Equity Research Tool API",
            "status": "active",
            "deployment_info": {
                "platform": "Vercel",
                "recommended_platform": "Streamlit Cloud",
                "repository": "https://github.com/manikyapavan925/news_equity_research_tool"
            }
        }
        
        self.wfile.write(json.dumps(response).encode())
