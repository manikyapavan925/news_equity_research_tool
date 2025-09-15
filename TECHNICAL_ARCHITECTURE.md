# News Equity Research Tool - Technical Architecture & Implementation

## Functional Components & Techniques

### 1. Article Content Fetching & Extraction
**Functionality:** Retrieve and parse news articles from various financial websites
**Techniques Used:**
- **HTTP Requests:** `requests` library with custom headers and user-agent rotation
- **HTML Parsing:** BeautifulSoup (bs4) for DOM traversal and content extraction
- **Selector Strategies:** Multiple CSS selector fallbacks for different website structures
- **Encoding Handling:** Automatic UTF-8 detection and fallback encoding support
- **Site-Specific Logic:** Custom parsing rules for LiveMint, Reuters, Bloomberg, etc.
- **Error Handling:** Retry mechanisms with exponential backoff
- **Content Cleaning:** Regex-based removal of navigation, ads, and boilerplate text

### 2. Article Summarization
**Functionality:** Generate concise summaries of news articles in multiple lengths
**Techniques Used:**
- **Sentence Scoring Algorithm:** Custom scoring based on position, length, and keyword relevance
- **Length-Aware Selection:** Different algorithms for Short (2-3 sentences), Medium (3-4 sentences), Detailed (4-5 sentences)
- **Financial Keyword Boosting:** Enhanced scoring for finance-specific terms (stock, earnings, revenue, etc.)
- **Content Filtering:** Removal of noisy or incomplete sentences
- **Word Count Optimization:** Dynamic sentence selection to meet target word limits
- **Fallback Handling:** Graceful degradation for short or problematic articles

### 3. Intelligent Question Answering
**Functionality:** AI-powered responses to user questions about financial news
**Techniques Used:**
- **Context Relevance Analysis:** Determine if question relates to loaded articles
- **Multi-Model LLM Integration:** Support for Flan-T5, DistilBERT, and T5-small models
- **Smart Prompt Engineering:** Dynamic prompt creation based on question type and context
- **Fallback Chain Architecture:** LLM → DuckDuckGo → Tavily Advanced Search
- **Question Type Classification:** Categorization (financial, AI/tech, plans, performance, etc.)
- **Confidence Scoring:** Quality assessment and automatic fallback triggering

### 4. Price Extraction & Analysis
**Functionality:** Extract stock prices and financial data from search results
**Techniques Used:**
- **Regex Pattern Matching:** Multiple currency and price format recognition
- **Context-Aware Extraction:** Distinguish between different price types (trading, target, etc.)
- **Currency Detection:** Support for USD, GBP, EUR, INR with symbol recognition
- **Confidence Scoring:** Reliability assessment based on pattern strength
- **Structured Data Output:** Consistent price object format with metadata
- **Fallback Integration:** Automatic price extraction from search fallbacks

### 5. Web Search Integration
**Functionality:** External web search for comprehensive information gathering
**Techniques Used:**
- **DuckDuckGo Integration:** Direct API calls for instant search results
- **Tavily Advanced Search:** Premium search with analysis and source credibility
- **Result Cleaning:** Removal of UI artifacts and boilerplate from search snippets
- **Source Credibility:** Ranking and filtering based on domain authority
- **Content Deduplication:** Elimination of redundant information
- **Error Handling:** Graceful degradation when search services are unavailable

### 6. Sentiment Analysis
**Functionality:** Analyze emotional tone and market sentiment in news articles
**Techniques Used:**
- **Keyword-Based Classification:** Positive/negative word dictionaries
- **Ratio Calculation:** Sentiment intensity based on word frequency
- **Confidence Scoring:** Statistical reliability assessment
- **Multi-Article Aggregation:** Overall market sentiment calculation
- **Visual Indicators:** Color-coded sentiment display in UI

### 7. Content Formatting & Display
**Functionality:** Clean presentation of extracted and generated content
**Techniques Used:**
- **Markdown Cleaning:** Convert headers to bold text to prevent CSS bypass
- **HTML Sanitization:** Safe rendering of user-generated content
- **Responsive Design:** Mobile-friendly layout with Streamlit components
- **Progressive Enhancement:** Fallback display for unsupported features
- **Accessibility:** Semantic HTML structure and ARIA attributes

### 8. Session Management & Caching
**Functionality:** Efficient data handling and performance optimization
**Techniques Used:**
- **Streamlit Session State:** Persistent data across user interactions
- **Model Caching:** `@st.cache_resource` for expensive LLM loading
- **Data Caching:** `@st.cache_data` for repeated computations
- **Memory Management:** Cleanup of unused resources and data structures
- **Background Processing:** Non-blocking operations for better UX

### 9. Error Handling & Resilience
**Functionality:** Robust operation under various failure conditions
**Techniques Used:**
- **Try-Catch Blocks:** Comprehensive exception handling
- **Graceful Degradation:** Continue operation with reduced functionality
- **User Feedback:** Clear error messages and troubleshooting guidance
- **Logging:** Detailed error tracking for debugging
- **Timeout Management:** Prevent hanging operations

### 10. UI/UX Design
**Functionality:** Intuitive user interface for research workflows
**Techniques Used:**
- **Streamlit Framework:** Reactive web app development
- **Custom CSS:** Enhanced styling with font size control and responsive design
- **Component Architecture:** Modular UI elements (cards, expanders, columns)
- **Progressive Disclosure:** Information revealed contextually
- **Visual Hierarchy:** Clear information organization and emphasis

## Technology Stack & Dependencies

### Core Framework
- **Streamlit:** Web application framework for data science and ML
- **Python 3.8+:** Primary programming language

### Web Scraping & HTTP
- **Requests:** HTTP library for API calls and web scraping
- **BeautifulSoup4 (bs4):** HTML/XML parsing library
- **lxml:** XML parser for improved performance

### Machine Learning & AI
- **Transformers:** Hugging Face library for LLM integration
- **Flan-T5 (Base):** Primary question-answering model
- **DistilBERT:** Lightweight BERT model for QA tasks
- **T5-Small:** Fallback text generation model
- **BART Large CNN:** Article summarization model

### Search & External APIs
- **DuckDuckGo Search:** Instant web search integration
- **Tavily AI:** Advanced search with analysis capabilities
- **Custom API Clients:** Modular search engine implementations

### Data Processing
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computing (version-controlled to <2.0)
- **Regular Expressions (re):** Pattern matching and text processing

### Utilities & Helpers
- **python-dotenv:** Environment variable management
- **Warnings:** Deprecation and error filtering
- **OS/IO:** File system operations and path handling
- **Time/DateTime:** Timestamp management and scheduling

### Development & Testing
- **Git:** Version control system
- **GitHub:** Code repository and collaboration platform
- **Unit Tests:** Custom test scripts for functionality validation
- **Syntax Validation:** Python compilation checks

### Deployment & Infrastructure
- **Virtual Environment:** Isolated Python environment (.venv)
- **Procfile:** Heroku deployment configuration
- **Requirements Files:** Dependency management (requirements.txt, pyproject.toml)
- **Environment Configuration:** .env file support

### LLM Models Used
1. **google/flan-t5-base:** Primary QA model (77M parameters)
2. **distilbert-base-cased-distilled-squad:** Secondary QA model (66M parameters)
3. **t5-small:** Fallback generation model (60M parameters)
4. **facebook/bart-large-cnn:** Summarization model (406M parameters)

### External Services
- **Tavily API:** Advanced web search and content analysis
- **DuckDuckGo Search API:** Instant web search results
- **Financial News Sources:** Reuters, Bloomberg, LiveMint, FT, etc.

This comprehensive architecture enables robust financial news analysis with intelligent question answering, multi-source information gathering, and user-friendly presentation of complex financial data.