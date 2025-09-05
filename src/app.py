import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="News Research Assistant", page_icon="📰", layout="wide", initial_sidebar_state="expanded")

# Theme detection script
st.markdown("""
    <script>
        // Check if the user's system is in dark mode
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
        }
        
        // Listen for theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
            document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
        });
    </script>
    """, unsafe_allow_html=True)

# Now import all other required modules
import os
import re
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except ImportError:
    # Fallback for older transformers versions
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    pipeline = None
    st.warning("Transformers library not fully available. Some features may be limited.")

# Initialize session state for models
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

def get_embeddings():
    if st.session_state.embeddings is None:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return st.session_state.embeddings

# Get embeddings only when needed
embeddings = get_embeddings()

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quick_action" not in st.session_state:
    st.session_state.quick_action = None


# Load environment variables
load_dotenv()

# Custom CSS for styling with enhanced visibility for both modes
custom_css = """
<style>
    /* Dark mode variables */
    [data-theme="dark"] {
        --text-color: #E0E0E0;
        --bg-color: #1E1E1E;
        --container-bg: #2D2D2D;
        --container-border: #404040;
        --button-bg: #2E7D32;
        --button-hover: #388E3C;
        --heading-color: #81C784;
        --link-color: #64B5F6;
        --input-bg: #333333;
        --input-text: #FFFFFF;
        --input-border: #555555;
    }

    /* Light mode variables */
    [data-theme="light"] {
        --text-color: #212121;
        --bg-color: #FFFFFF;
        --container-bg: #F8F9FA;
        --container-border: #E0E0E0;
        --button-bg: #2E7D32;
        --button-hover: #1B5E20;
        --heading-color: #1A237E;
        --link-color: #1565C0;
        --input-bg: #FFFFFF;
        --input-text: #212121;
        --input-border: #E0E0E0;
    }

    /* Main container styling */
    .main, .element-container, .stMarkdown {
        color: var(--text-color) !important;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: var(--button-bg) !important;
        color: white !important;
        border: none !important;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        background-color: var(--button-hover) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: var(--container-bg) !important;
        color: var(--text-color) !important;
        border-radius: 15px;
        padding: 15px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
    }

    /* User message styling */
    .stChatMessage[data-testid="user"] {
        background-color: rgba(99, 179, 237, 0.1) !important;
        border: 1px solid rgba(99, 179, 237, 0.2);
    }

    /* Assistant message styling */
    .stChatMessage[data-testid="assistant"] {
        background-color: rgba(104, 211, 145, 0.1) !important;
        border: 1px solid rgba(104, 211, 145, 0.2);
    }

    /* Text and headings */
    p, span, div {
        color: var(--text-color) !important;
        font-size: 14px !important;
    }

    h1 {
        color: var(--heading-color) !important;
        font-size: 24px !important;
    }

    h2 {
        color: var(--heading-color) !important;
        font-size: 20px !important;
    }

    h3 {
        color: var(--heading-color) !important;
        font-size: 18px !important;
    }

    h4 {
        color: var(--heading-color) !important;
        font-size: 16px !important;
    }

    h5, h6 {
        color: var(--heading-color) !important;
        font-size: 14px !important;
    }

    /* Analysis output styling */
    .stMarkdown {
        font-size: 14px !important;
    }
    
    .stMarkdown h3 {
        font-size: 18px !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
    }
    
    .stMarkdown h4 {
        font-size: 16px !important;
        margin-top: 15px !important;
        margin-bottom: 8px !important;
    }
    
    .stMarkdown ul li, .stMarkdown ol li {
        font-size: 14px !important;
        line-height: 1.4 !important;
        margin-bottom: 4px !important;
    }

    /* Links */
    a {
        color: var(--link-color) !important;
    }

    /* Markdown text */
    .markdown-text-container {
        color: var(--text-color) !important;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        color: var(--text-color) !important;
        background-color: var(--container-bg) !important;
        border-color: var(--border-color) !important;
    }

    /* Custom containers */
    .custom-info-container {
        background-color: var(--container-bg);
        color: var(--text-color);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        margin: 10px 0;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# App header with dark mode support
st.markdown("""
    <div class='custom-info-container'>
        <h1 style='text-align: center; margin-bottom: 20px;'>
            📰 News Research Assistant
        </h1>
        <div style='text-align: center;'>
            <p style='font-size: 1.1em; margin: 0;'>
                Analyze news articles, extract insights, and get instant summaries. Simply add article URLs and start chatting!
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add a divider
st.divider()

def load_llm():
    """Load the LLM model with proper error handling"""
    try:
        import torch
        
        # Initialize with a stable model for financial analysis
        model_name = "facebook/bart-base"  # Using base model for stability
        
        st.info("Loading model and tokenizer...")
        
        # Basic tokenizer setup
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure model with safe defaults
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            device_map="auto",  # Let the model handle device placement
            return_dict=True    # Ensure structured outputs
        )
        
        # Create pipeline with safe parameters
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,      # Increased for better coverage
            min_length=50,        # Reasonable minimum
            num_beams=4,          # Balanced beam search
            length_penalty=1.0,   # Neutral length penalty
            early_stopping=True,  # Enable early stopping
            framework="pt",       # Use PyTorch backend
            device_map="auto",    # Automatic device placement
            use_cache=True        # Enable caching for better performance
        )
        
        # Create and return the LangChain LLM with enhanced parameters
        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={
                "temperature": 0.7,        # Balanced temperature for reliable outputs
                "max_length": 1024,        # Reasonable length limit
                "min_length": 100,         # Minimum length for meaningful responses
                "num_beams": 4,            # Balanced beam search
                "length_penalty": 1.0,     # Neutral length penalty
                "early_stopping": True,    # Enable early stopping for efficiency
                "no_repeat_ngram_size": 3, # Avoid repetition
                "use_cache": True          # Enable caching
            }
        )
        return llm
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,  # Use float32 for better compatibility
            device_map="auto"
        ).to(device)
        
        # Create pipeline with explicit device and parameters
        pipe = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            device=device,
            framework="pt"  # Explicitly set framework to PyTorch
        )
        
        # Create and return the HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize session state
if "app_loaded" not in st.session_state:
    st.session_state.app_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state and not st.session_state.get("app_loaded"):
    with st.spinner("Setting up your research assistant..."):
        st.session_state.llm = load_llm()
        st.session_state.app_loaded = True

def format_docs(docs):
    """Format documents for context, removing duplicates and cleaning text while preserving structure."""
    try:
        if not docs or not isinstance(docs, list):
            return ""
            
        seen_content = set()
        formatted_docs = []
        
        for doc in docs:
            try:
                if not doc or not hasattr(doc, 'page_content'):
                    continue
                    
                content = doc.page_content.strip() if doc.page_content else ""
                
                # Skip very short or empty content
                if not content or len(content) < 50:
                    continue
                
                # Clean the content while preserving important structure
                content = clean_webpage_content(content)
                
                # Extract and format the title
                title = ""
                content_lines = content.split('\n')
                if content_lines and len(content_lines) > 0:
                    potential_title = content_lines[0].strip()
                    if potential_title and 10 < len(potential_title) < 200:  # Reasonable title length
                        title = potential_title
                        content = '\n'.join(content_lines[1:]).strip()
                
                # Extract key information sections with improved organization
                sections = {
                    "Title": title if title else "Untitled Article",
                    "Key Points": [],
                    "Facts & Figures": [],
                    "Market Data": [],
                    "Company Updates": [],
                    "Quotes": [],
                    "Main Content": content
                }
                
                # Extract market and financial data with context
                financial_patterns = {
                    'money': r'(?:[\$€£]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion)))',
                    'percentage': r'\d+(?:\.\d+)?%',
                    'growth': r'(?:increase|decrease|growth|decline)\s+(?:of\s+)?(?:\d+(?:\.\d+)?%)',
                    'market_cap': r'market\s+cap\w*\s+of\s+[\$€£]?\d+(?:\.\d+)?[BMT]?'
                }
                
                market_data = []
                for pattern in financial_patterns.values():
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Get surrounding sentence for context
                        start = max(0, match.start() - 100)
                        end = min(len(content), match.end() + 100)
                        surrounding = content[start:end]
                        sentence = re.findall(r'[^.!?]*' + re.escape(match.group()) + r'[^.!?]*[.!?]', surrounding)
                        if sentence:
                            market_data.append(sentence[0].strip())
                
                sections["Market Data"] = list(dict.fromkeys(market_data))
                
                # Extract quotes
                quotes = re.findall(r'"([^"]*)"', content)
                if quotes:
                    sections["Quotes"] = [quote.strip() for quote in quotes if quote.strip()]
                
                # Extract key points (sentences with important indicators)
                key_indicators = ['significant', 'important', 'major', 'key', 'critical', 'essential', 'primary']
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in key_indicators):
                        sections["Key Points"].append(sentence.strip())
                
                # Format the document with sections
                formatted_content = []
                for section, data in sections.items():
                    if isinstance(data, list) and data:
                        formatted_content.append(f"{section}:\n" + "\n".join(f"• {item}" for item in data))
                    elif isinstance(data, str) and data.strip():
                        formatted_content.append(f"{section}:\n{data}")
                
                # Only add non-empty formatted content
                final_content = "\n\n".join(c for c in formatted_content if c.strip())
                
                if final_content and final_content not in seen_content:
                    seen_content.add(final_content)
                    formatted_docs.append(final_content)
                    
            except Exception as doc_error:
                st.warning(f"Error processing document: {str(doc_error)}")
                continue
                
        # Return joined documents or empty string
        return "\n\n---\n\n".join(formatted_docs) if formatted_docs else ""
        
    except Exception as e:
        st.error(f"Error formatting documents: {str(e)}")
        return ""
    
    return "\n\n---\n\n".join(formatted_docs)

def format_response(response):
    """Format the response for better readability with enhanced structure"""
    try:
        if not response:
            return "No response generated."
            
        # Clean up the basic text
        response = str(response).strip()
        if not response:
            return "Empty response received."
        
        # Remove any markdown artifacts and clean up the text
        response = re.sub(r'\\n', '\n', response)
        response = re.sub(r'\\[rtn]', ' ', response)
        response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
        
        # Convert list-like text into proper bullet points
        response = re.sub(r'(?m)^\s*[\d#]+\.\s*', '\n• ', response)  # Convert "1." style lists
        response = re.sub(r'(?m)^\s*[-*]\s*', '\n• ', response)      # Convert existing bullet points
        
        # If the response doesn't have bullet points but has sentences, convert to bullets
        if '• ' not in response and len(response.split('. ')) > 1:
            sentences = [s.strip() for s in response.split('. ') if s.strip()]
            response = '\n• ' + '\n• '.join(sentences)
        
        # Format key terms and important phrases
        response = re.sub(r'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+):', r'**\1:**', response)  # Key terms
        response = re.sub(r'(?<!\d)(\d+\.?\d*%?)(?!\d)', r'`\1`', response)  # Numbers and percentages
        
        # Add section headers if response is long
        if len(response.split('\n• ')) > 3:
            response = "**Summary:**\n" + response
        
        # Clean up the formatting
        response = re.sub(r'\n{3,}', '\n\n', response)  # Remove extra newlines
        response = re.sub(r'[•]\s*[•]', '•', response)  # Remove duplicate bullets
        response = re.sub(r'(?<=[.!?])\s*(?=\n?[•])', '\n', response)  # Clean spacing around bullets
        
        # Add source attribution if present
        if 'source' in response.lower() or 'from' in response.lower():
            response += "\n\n*Data sourced from provided news articles*"
        
        formatted = response.strip()
        return formatted if formatted else "No meaningful content to display."
        
    except Exception as e:
        st.error(f"Error formatting response: {str(e)}")
        return str(response)  # Return unformatted response as fallback

def create_qa_prompt():
    """Create the QA prompt template"""
    template = """You are an expert financial analyst. Your task is to analyze the provided articles and present information in a clear, bullet-point format.

RESPONSE FORMAT:
• Always use bullet points (•) for each piece of information
• Group related points under clear subheadings
• Never use paragraphs - convert all information to concise bullet points
• Include source citations [1], [2], etc. for each point

CONTENT REQUIREMENTS:
• Extract and verify facts from source articles only
• Include ALL numerical data: prices, percentages, dates, market values
• Quote key statements from executives/analysts using exact quotes
• Highlight market movements and financial metrics
• Note any significant trends or changes

MUST INCLUDE:
• Market data and financial figures with exact values
• Dates and timeframes for all events
• Company names and ticker symbols exactly as written
• Comparisons with previous periods where available
• Source citations for every factual statement

DO NOT:
• Make assumptions or predictions not stated in articles
• Include general or vague statements without data
• Omit source citations
• Use paragraph format

Context from articles:
{context}

Question:
{question}

Remember:
• Every point must include a source citation
• Focus on concrete data and metrics
• Use bullet points exclusively
• Include exact figures and dates

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def process_query(query):
    """Process a query using enhanced semantic search and structured LLM output"""
    # Input validation
    if not query or not query.strip():
        return "Please provide a valid query."
        
    # Enhance query with specific requirements for structured output
    query = f"""
    Analyze the following query and provide a structured bullet-point response:
    {query}
    
    Requirements:
    - Extract all relevant facts and figures
    - Use bullet points for ALL information
    - Include source citations
    - Focus on concrete data and metrics
    """

    # Check required components
    if not st.session_state.vector_store:
        return "Please load some news articles first before asking questions."
        
    if not st.session_state.llm:
        return "The research assistant is not properly initialized. Please refresh the page."
        
    try:
        # Determine analysis mode and enhance query
        mode = "qa"  # Default mode
        if "summarize" in query.lower():
            mode = "summarize"
            enhanced_query = """FINANCIAL ANALYSIS SUMMARY
            Required sections:
            1. Executive Summary
            2. Key Events (with dates)
            3. Financials & Metrics
            4. Management Commentary
            5. Market Reaction
            6. Risks/Unknowns
            
            Include for each point:
            - Exact figures, dates, and currency values
            - Company names and ticker symbols
            - Direct quotes from executives
            - Source citations"""
        elif "insights" in query.lower():
            enhanced_query = """Extract key business insights from these articles:
            1. Focus on market implications
            2. Include financial metrics and trends
            3. Highlight strategic decisions
            4. Note industry impacts
            5. Present in concise bullet points"""
        elif "facts" in query.lower() or "figures" in query.lower():
            enhanced_query = """Extract key facts and figures from these articles:
            1. List all numerical data
            2. Include dates and timeframes
            3. Quote specific statistics
            4. Mention monetary values
            5. Present in clear bullet points"""
            
        try:
            # Safety check for vector store
            if not hasattr(st.session_state.vector_store, 'docstore') or not hasattr(st.session_state.vector_store.docstore, '_dict'):
                return "The document database is not properly initialized. Please try adding articles again."
            
            # Get all available documents
            total_docs = len(st.session_state.vector_store.docstore._dict)
            if total_docs == 0:
                return "No documents are available in the database. Please add some news articles first."
            
            # Configure retrieval with fallback options
            try:
                # Start with similarity search
                docs = st.session_state.vector_store.similarity_search(
                    enhanced_query,
                    k=min(5, total_docs)
                )
                
                # If no results, try with more lenient parameters
                if not docs:
                    docs = st.session_state.vector_store.similarity_search(
                        enhanced_query,
                        k=min(8, total_docs),
                        distance_threshold=0.8
                    )
                
            except Exception as retriever_error:
                st.warning("MMR retrieval failed, falling back to similarity search")
                # Fallback to similarity search with scoring
                docs_and_scores = st.session_state.vector_store.similarity_search_with_score(
                    enhanced_query,
                    k=min(4, total_docs)
                )
                # Only keep documents with good relevance scores
                docs = [doc for doc, score in docs_and_scores if score < 0.8]
            
            if not docs:
                return "No relevant information found in the loaded articles."
            
            # Format documents for context with validation
            context = format_docs(docs)
            if not context or not context.strip():
                return "Could not extract meaningful content from the articles. Please try rephrasing your query."
                
        except Exception as docs_error:
            st.error(f"Error retrieving documents: {str(docs_error)}")
            return "Error accessing the document database. Please try again or reload the articles."
        
        # Construct structured financial analysis prompt
        prompt = f"""FINANCIAL ANALYSIS REPORT

        MODE: {mode}
        
        SOURCE LIST:
        {[f"[{i+1}] {doc.metadata.get('source', 'Unknown Source')}" for i, doc in enumerate(docs)]}
        
        CONTEXT:
        {context}
        
        REQUIRED FORMAT:
        1. Each section must be clearly titled
        2. All facts must include source citations [1], [2], etc.
        3. Include exact figures, dates, and currency values
        4. Preserve all ticker symbols and company names exactly
        5. Direct quotes must be exact with proper attribution
        6. Each bullet <= 30 words, focused on verifiable facts
        7. No marketing language or calls-to-action
        
        SECTIONS REQUIRED:

        Required Details:
        - Specific dates and timeframes for all events
        - Exact numbers, percentages, and financial figures
        - Full names of companies and key individuals
        - Precise market data and statistics
        - Direct quotes from stakeholders
        - Detailed context for each development

        Output Format:
        - Main section with key highlights
        - Detailed bullet points grouped by topic
        - Supporting data under relevant points
        - Market implications clearly stated
        - Temporal sequence of events preserved

        2. Content Precision:
           - Include specific dates, numbers, and statistics
           - Quote exact figures, percentages, and values
           - Cite specific companies, individuals, and organizations
           - Include time frames and temporal context

        3. Information Quality:
           - Focus solely on factual information from the articles
           - Exclude speculation and assumptions
           - Prioritize the most recent and relevant information
           - Highlight significant changes or developments

        4. Source Attribution:
           - Reference specific articles when presenting key facts
           - Note any conflicting information between sources
           - Indicate if information comes from specific experts or authorities
           - Maintain traceability of important claims

        Articles content:
        {context}

        Remember: Be precise, factual, and well-organized in your response.
        """
        
        try:
            # Validate prompt
            if not prompt or not prompt.strip():
                return "Error: Invalid prompt generated. Please try again."
            
            # Call LLM with enhanced error handling
            max_retries = 2
            response = None
            last_error = None

            for attempt in range(max_retries):
                try:
                    # Ensure LLM is properly initialized
                    if not st.session_state.llm:
                        st.session_state.llm = load_llm()
                        if not st.session_state.llm:
                            raise ValueError("Failed to initialize LLM model")

                    # Prepare prompt with length limits
                    cleaned_prompt = prompt.strip()
                    if len(cleaned_prompt) > 1000:
                        cleaned_prompt = cleaned_prompt[:1000] + "..."

                    # Add safety wrapper around LLM call
                    with st.spinner(f"Generating response (attempt {attempt + 1}/{max_retries})..."):
                        try:
                            # LLM call with increased token limit
                            llm_response = st.session_state.llm(
                                cleaned_prompt,
                                temperature=0.7,
                                max_length=1024,  # Increased token limit
                                min_length=50,
                                num_beams=4,
                                early_stopping=True
                            )

                            # Handle response safely
                            if isinstance(llm_response, dict) and "generated_text" in llm_response:
                                response = llm_response["generated_text"]
                            elif isinstance(llm_response, str):
                                response = llm_response
                            elif isinstance(llm_response, list) and llm_response and isinstance(llm_response[0], str):
                                response = llm_response[0]
                            elif isinstance(llm_response, list) and llm_response and isinstance(llm_response[0], dict):
                                response = llm_response[0].get("generated_text", "")
                            else:
                                raise ValueError(f"Unexpected response format: {type(llm_response)}")

                            # Validate final response
                            if response and isinstance(response, str) and len(response.strip()) > 0:
                                break
                            else:
                                raise ValueError("Empty or invalid response from LLM")

                        except Exception as e:
                            last_error = str(e)
                            st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                            if attempt < max_retries - 1:
                                time.sleep(2)  # Slightly longer delay
                            continue

                except Exception as outer_error:
                    last_error = str(outer_error)
                    if attempt == max_retries - 1:
                        st.error(f"LLM error: {str(outer_error)}")
                        return "Error: Unable to generate a response. Please try again with a different question."
                    time.sleep(2)

            # If all attempts failed
            if not response:
                st.error(f"All attempts failed. Last error: {last_error}")
                return "Error: Could not generate a valid response after multiple attempts. Please try rephrasing your question."
            
            # Format structured response
            import json
            
            # Parse response into sections
            sections = []
            if "summarize" in query.lower():
                sections = [
                    {"title": "Executive Summary", "bullets": []},
                    {"title": "Key Events", "bullets": []},
                    {"title": "Financials & Metrics", "bullets": []},
                    {"title": "Management Commentary", "bullets": []},
                    {"title": "Market Reaction", "bullets": []},
                    {"title": "Risks/Unknowns", "bullets": []}
                ]
                mode = "summarize"
            else:
                sections = [{"title": "Analysis", "bullets": []}]
                mode = "qa"
            
            # Parse the response into bullets
            current_section = sections[0]
            for line in response.split('\n'):
                line = line.strip()
                if line:
                    if line.startswith('#') or line.endswith(':'):
                        # Try to match with a section
                        title = line.replace('#', '').replace(':', '').strip()
                        for section in sections:
                            if title.lower() in section["title"].lower():
                                current_section = section
                                break
                    elif line.startswith('•'):
                        bullet = line[1:].strip()
                        if '[' not in bullet:
                            bullet += ' [1]'  # Add default source
                        current_section["bullets"].append(bullet)
                    elif len(line) > 10:  # Meaningful line
                        if '[' not in line:
                            line += ' [1]'  # Add default source
                        current_section["bullets"].append(line)
            
            # Create citations
            citations = []
            for i, doc in enumerate(docs):
                if hasattr(doc, 'metadata'):
                    meta = doc.metadata
                    citations.append({
                        "id": i + 1,
                        "title": meta.get('title', 'Unknown Title'),
                        "site": meta.get('source', 'Unknown Source'),
                        "date": meta.get('date', 'Unknown Date'),
                        "url": meta.get('url', '')
                    })
            
            # Build final JSON response
            response_json = {
                "mode": mode,
                "sections": sections,
                "confidence": "High" if len(docs) >= 3 else "Medium",
                "citations": citations
            }
            
            # Convert JSON response to formatted text
            formatted_output = []
            
            # Add mode-specific header
            if response_json["mode"] == "summarize":
                formatted_output.append("#### Executive Summary")
            else:
                formatted_output.append("#### Analysis Report")
                
            # Process each section
            for section in response_json["sections"]:
                formatted_output.append(f"\n##### {section['title']}")
                for bullet in section["bullets"]:
                    # Clean up the bullet point and ensure proper citation format
                    cleaned_bullet = bullet.strip()
                    if not cleaned_bullet.endswith("]"):
                        # Add citation if missing
                        cleaned_bullet += " [1]"
                    formatted_output.append(f"• {cleaned_bullet}")
            
            # Add citations section
            if response_json["citations"]:
                formatted_output.append("\n#### Sources")
                for citation in response_json["citations"]:
                    source_text = f"[{citation['id']}] {citation['title']}"
                    if citation['url']:
                        source_text += f" - {citation['url']}"
                    if citation['date'] != "Unknown Date":
                        source_text += f" ({citation['date']})"
                    formatted_output.append(f"• {source_text}")
            
            # Add confidence level
            formatted_output.append(f"\n*Analysis Confidence: {response_json['confidence']}*")
            
            # Return formatted text
            return "\n".join(formatted_output)
            
            if not response or not response.strip():
                return "Error: Received empty response from assistant. Please try rephrasing your question."
            
            # Process and refine the response
            response_text = response.strip()
            
            # Remove any duplicate or redundant information
            sentences = [s.strip() for s in response_text.split('.') if s.strip()]
            unique_sentences = []
            seen_content = set()
            
            for sentence in sentences:
                # Create a simplified version for comparison (remove numbers and specific details)
                simplified = re.sub(r'\d+(?:\.\d+)?%?', 'NUM', sentence.lower())
                simplified = re.sub(r'[\$€£]\d+(?:\.\d+)?[BMT]?', 'MONEY', simplified)
                
                if simplified not in seen_content:
                    seen_content.add(simplified)
                    unique_sentences.append(sentence)
            
            # Rebuild response with unique content
            formatted_response = '. '.join(unique_sentences)
            
            # Clean up formatting
            formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)  # Remove excess newlines
            formatted_response = re.sub(r'(?<=[.!?])\s+(?=•)', '\n', formatted_response)  # Clean bullet spacing
            
            # Improve bullet point structure
            if not formatted_response.startswith('•'):
                points = []
                current_point = []
                
                for line in formatted_response.split('\n'):
                    if re.match(r'^[A-Z]', line.strip()):  # New main point
                        if current_point:
                            points.append(' '.join(current_point))
                        current_point = [line.strip()]
                    else:
                        current_point.append(line.strip())
                
                if current_point:
                    points.append(' '.join(current_point))
                
                formatted_response = '• ' + '\n• '.join(points)
            
            # Ensure proper bullet point formatting
            if not re.search(r'^\s*[•\-\*]', formatted_response, re.MULTILINE):
                # Convert numbered lists to bullets
                formatted_response = re.sub(r'(?m)^\s*\d+\.\s*', '• ', formatted_response)
                # Convert plain paragraphs to bullets if no bullet structure exists
                if '• ' not in formatted_response:
                    points = [p.strip() for p in formatted_response.split('\n') if p.strip()]
                    formatted_response = '• ' + '\n• '.join(points)
            
            # Add appropriate header based on query type
            header = None
            if "facts" in query.lower() or "figures" in query.lower():
                header = "**Key Facts & Figures:**"
            elif "insights" in query.lower():
                header = "**Key Business Insights:**"
            elif "summarize" in query.lower():
                header = "**Executive Summary:**"
            else:
                header = "**Detailed Analysis:**"
            
            # Format numbers and dates for better readability
            formatted_response = re.sub(r'(\d+\.?\d*%)', r'`\1`', formatted_response)  # Percentages
            formatted_response = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', r'`$\1`', formatted_response)  # Money
            formatted_response = re.sub(r'(\d{1,2}/\d{1,2}/\d{2,4})', r'`\1`', formatted_response)  # Dates
            
            # Add section header
            formatted_response = f"{header}\n\n{formatted_response}"
            
            # Add source attribution if available
            if docs and len(docs) > 0 and hasattr(docs[0], 'metadata'):
                sources = set(doc.metadata.get('source', '') for doc in docs if hasattr(doc, 'metadata') and doc.metadata.get('source'))
                if sources:
                    formatted_response += "\n\n**Sources:**\n" + "\n".join(f"• {source}" for source in sorted(sources))
            
            # Add timestamp for freshness
            formatted_response += "\n\n*Analysis generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "*"
            
            return formatted_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
            
    except Exception as e:
        return f"Error processing query: {str(e)}"



@st.cache_resource
def load_llm():
    """Load the LLM model"""
    try:
        # Initialize model and tokenizer
        model_name = "facebook/bart-base"  # Using base model for better stability
        
        import torch
        # Check if CUDA is available and set device accordingly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model with specific configuration
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32  # Use float32 for better compatibility
        ).to(device)
        
        # Create the pipeline with stable parameters
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            min_new_tokens=30,
            temperature=0.7,
            device=device,
            framework="pt"  # Explicitly set framework to PyTorch
        )
        
        # Convert to LangChain model
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def is_valid_url(url):
    """Check if the URL is valid"""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def clean_webpage_content(content):
    """Clean webpage content by removing unnecessary elements and formatting"""
    # Remove URLs and special characters
    content = re.sub(r'http[s]?://\S+', '', content)
    # Remove multiple spaces and newlines
    content = re.sub(r'\s+', ' ', content)
    # Remove special characters but keep basic punctuation
    content = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', content)
    # Remove any remaining artifacts
    content = re.sub(r'\.{2,}', '.', content)  # Replace multiple dots with single dot
    return content.strip()

def process_urls(urls):
    """Process multiple URLs and create vector store"""
    all_chunks = []  # Initialize list to store all chunks
    
    try:
        # Process each URL individually
        for url in urls:
            if not url or not isinstance(url, str):
                continue
            
            if not is_valid_url(url):
                st.warning(f"Skipping invalid URL: {url}")
                continue
            
            st.info(f"Processing URL: {url}")    
            try:
                # Load webpage content
                loader = WebBaseLoader([url])  # Expects a list of URLs
                documents = loader.load()
                
                if not documents:
                    st.warning(f"No content found at {url}")
                    continue
                
                # Clean the content
                for doc in documents:
                    doc.page_content = clean_webpage_content(doc.page_content)
                    doc.metadata['source'] = url  # Ensure source URL is in metadata
                
                # Debug: Show raw content length
                total_content = sum(len(doc.page_content) for doc in documents)
                st.info(f"Retrieved {total_content} characters of clean content from {url}")
                
                if total_content < 100:  # Sanity check for minimum content
                    st.warning(f"Very little content found at {url}. Might be access restricted.")
                    continue
                
                # Split text into optimized chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", ": ", ", ", " ", ""]
                )
                
                chunks = text_splitter.split_documents(documents)
                
                if chunks:
                    st.info(f"Created {len(chunks)} chunks from {url}")
                    all_chunks.extend(chunks)
                    st.success(f"Successfully processed: {url}")
                    
                    with st.expander("Preview first chunk"):
                        st.write(chunks[0].page_content[:200] + "...")
                else:
                    st.warning(f"No content chunks extracted from: {url}")
            
            except Exception as url_error:
                st.error(f"Error processing URL {url}: {str(url_error)}")
                continue
        
        if not all_chunks:
            st.error("No content was successfully loaded from any URLs.")
            return False
        
        st.write("Creating embeddings...")
        
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        st.info(f"Processing {len(all_chunks)} document chunks...")
        
        # Create vector store
        vector_store = FAISS.from_documents(all_chunks, embeddings)
        
        if not hasattr(vector_store, 'index') or not hasattr(vector_store, 'docstore'):
            st.error("Vector store creation failed. Invalid store structure.")
            return False
        
        doc_count = len(vector_store.docstore._dict)
        if doc_count == 0:
            st.error("No documents were successfully added to the vector store.")
            return False
        
        st.info(f"Successfully added {doc_count} documents to the vector store")
        st.session_state.vector_store = vector_store
        st.success("Vector store created successfully!")
        return True
        
    except Exception as e:
        st.error(f"Error in process_urls: {str(e)}")
        return False
            
    except Exception as e:
        st.error("Error in document processing")
        st.error(str(e))
        return False

def get_conversation_chain():
    """Create a conversation chain"""
    try:
        llm = load_llm()
        if llm is None:
            st.error("Could not initialize the language model. Please try again.")
            return None
            
        try:
            # Create a safer retriever with fallback
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",  # Use similarity search
                search_kwargs={
                    "k": 3,  # Reduced number of chunks
                    "score_threshold": 0.5,  # Only return relevant chunks
                }
            )
            
            # Create the chain with memory
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                max_tokens_limit=2000,  # Limit context size
                combine_docs_chain_kwargs={
                    "prompt": None  # Use default prompt
                }
            )
            return chain
        except Exception as e:
            st.error(f"Error creating conversation chain: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return None
    except Exception as e:
        st.error(f"Error in conversation chain setup: {str(e)}")
        return None

# Streamlit UI
# Initialize session state for URL inputs
if "url_inputs" not in st.session_state:
    st.session_state.url_inputs = [""]  # Start with one empty URL field

def add_url_field():
    st.session_state.url_inputs.append("")

def process_all_urls():
    urls = [url.strip() for url in st.session_state.url_inputs if url.strip()]
    if urls:
        if process_urls(urls):
            st.success("✅ URLs processed successfully!")
            return True
    return False

# Main app initialization is at the top of the file

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .url-input {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .chat-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.markdown("<h1 class='main-header'>📰 Financial News Research Assistant</h1>", unsafe_allow_html=True)

# URL input container
st.markdown("""
    <div class='custom-info-container'>
        <h3 style='margin-bottom: 15px;'>📥 Add News Articles</h3>
        <p style='font-size: 0.9em; margin-bottom: 20px;'>
            Enter URLs of news articles to analyze. Add up to 3 articles at once.
        </p>
    </div>
""", unsafe_allow_html=True)

# URL input form
with st.form("url_form", clear_on_submit=False):
    # Create columns for better layout
    cols = st.columns([1])
    
    with cols[0]:
        for i, url in enumerate(st.session_state.url_inputs):
            st.text_input(
                f"Article #{i+1}",
                value=url,
                key=f"url_{i}",
                placeholder="https://example.com/article",
                help="Enter the full URL of a news article"
            )
            st.session_state.url_inputs[i] = st.session_state[f"url_{i}"]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            add_url = st.form_submit_button("➕ Add Another Source")
        with col2:
            process_button = st.form_submit_button("🔄 Analyze Articles")
        
        if add_url:
            st.session_state.url_inputs.append("")
            st.rerun()
            
        if process_button:
            with st.spinner("Processing articles..."):
                process_all_urls()

# Chat interface
st.divider()
st.markdown("""
    <div class='custom-info-container'>
        <h3 style='margin-bottom: 15px;'>💬 Research Assistant</h3>
    </div>
""", unsafe_allow_html=True)

if st.session_state.vector_store is not None:
    # Initialize session states if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "quick_action" not in st.session_state:
        st.session_state.quick_action = None
        
    # Start chat container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Quick action buttons section
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 Summarize Articles", key="summarize_btn", help="Get a concise bullet-point summary of all loaded articles"):
            st.session_state.quick_action = """Provide a concise bullet-point summary of the main points from these articles. Please:
            - Extract the key news, events, and announcements
            - Include article titles and dates
            - Focus on facts and verified information
            - Present information in a clear, organized manner
            - Highlight any significant changes or developments"""
    with col2:
        if st.button("💡 Key Insights", key="insights_btn", help="Extract key business insights and analysis"):
            st.session_state.quick_action = """Analyze the articles and provide key insights in bullet points. Include:
            - Main business implications and market impacts
            - Strategic developments and their significance
            - Industry trends and market movements
            - Stakeholder implications
            - Future outlook and potential impacts"""
    with col3:
        if st.button("📊 Main Facts", key="facts_btn", help="List key facts, figures, and statistics"):
            st.session_state.quick_action = """Extract and list all specific facts and figures from the articles. Include:
            - All numerical data and statistics
            - Dates and timeframes
            - Financial figures and percentages
            - Market data and metrics
            - Quantitative comparisons and changes"""
    
    # Chat input
    user_input = st.chat_input("Ask me anything about the articles...", key="query")
    
    # Process query (either from quick action or chat input)
    current_query = None
    
    if st.session_state.quick_action:
        current_query = st.session_state.quick_action
        st.session_state.quick_action = None
    elif user_input:
        current_query = user_input
    
    # Process if we have a query
    if current_query:
        with st.chat_message("user", avatar="🧑"):
            st.write(current_query)
        
        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner("Analyzing financial data..."):
                    answer = process_query(current_query)
                    if answer:
                        formatted_answer = format_response(answer)
                        if formatted_answer:
                            sections = formatted_answer.split('\n\n**')
                            
                            if sections:
                                st.markdown(sections[0])
                                if len(sections) > 1:
                                    with st.expander("📊 Detailed Analysis"):
                                        st.markdown('**' + '\n\n**'.join(sections[1:]))
                            
                            st.session_state.chat_history.append((current_query, formatted_answer))
                        else:
                            st.error("Error formatting the response. Please try again.")
                    else:
                        st.warning("No relevant information found. Please try a different question.")
            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
                st.warning("Please try again or rephrase your question.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### 📜 Previous Conversations")
        
        for past_query, past_answer in reversed(st.session_state.chat_history):
            with st.chat_message("user", avatar="🧑"):
                st.write(past_query)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(past_answer)
    
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("👈 Please add and process some news articles first")
    
    # Quick action buttons with improved styling
    st.markdown("""
        <div style='background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #1a237e; margin-bottom: 15px; font-size: 1.2em;'>Quick Actions</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 Summarize Articles", key="summarize_btn", help="Get a concise summary of all loaded articles"):
            st.session_state.quick_action = "Summarize these articles"
    with col2:
        if st.button("💡 Key Insights", key="insights_btn", help="Extract key insights and analysis"):
            st.session_state.quick_action = "What are the key insights from these articles?"
    with col3:
        if st.button("📊 Main Facts", key="facts_btn", help="List main facts and figures"):
            st.session_state.quick_action = "What are the main facts and figures mentioned?"
    
    # Chat input at the top
    user_input = st.chat_input("Ask me anything about the articles...", key="query")
    
    # Process query (either from quick action or chat input)
    current_query = None
    
    # Handle quick actions and user input
    if st.session_state.quick_action:
        current_query = st.session_state.quick_action
        # Clear quick action after using it
        st.session_state.quick_action = None
    elif user_input:
        current_query = user_input
    
    # Process if we have a query
    if current_query:
        # Show user message
        with st.chat_message("user", avatar="🧑"):
            st.write(current_query)
        
        # Show assistant message
        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner("Analyzing articles..."):
                    # Get response
                    answer = process_query(current_query)
                    
                    if not answer:
                        st.warning("No response generated. Please try again.")
                    else:
                        # Format the response
                        formatted_answer = format_response(answer)
                        
                        if formatted_answer:
                            # Display the response
                            try:
                                # Split into sections if response has sections
                                if '**' in formatted_answer:
                                    sections = formatted_answer.split('\n\n**')
                                    if sections:
                                        # Display first section (overview)
                                        st.markdown(sections[0])
                                        
                                        # Display remaining sections in expander
                                        if len(sections) > 1:
                                            with st.expander("📊 Detailed Analysis"):
                                                for section in sections[1:]:
                                                    st.markdown(f"**{section}")
                                else:
                                    # Display as a single section if no sections detected
                                    st.markdown(formatted_answer)
                                
                                # Save to chat history
                                st.session_state.chat_history.append((current_query, formatted_answer))
                                
                            except Exception as e:
                                st.error(f"Error displaying response: {str(e)}")
                                st.markdown(formatted_answer)  # Fallback to direct display
                        else:
                            st.error("Error formatting the response. Please try again.")
                        
                    # Display the response
                    try:
                        # Split into sections if response has sections
                        if '**' in formatted_answer:
                            sections = formatted_answer.split('\n\n**')
                            if sections:
                                # Display first section (overview)
                                st.markdown(sections[0])
                                
                                # Display remaining sections in expander
                                if len(sections) > 1:
                                    with st.expander("📊 Detailed Analysis"):
                                        for section in sections[1:]:
                                            st.markdown(f"**{section}")
                        else:
                            # Display as a single section if no sections detected
                            st.markdown(formatted_answer)
                        
                        # Save to chat history
                        st.session_state.chat_history.append((current_query, formatted_answer))
                        
                    except Exception as e:
                        st.error(f"Error displaying response: {str(e)}")
                        st.markdown(formatted_answer)  # Fallback to direct display
                    else:
                        st.warning("No relevant information found. Please try a different question.")
            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
                st.warning("Please try again or rephrase your question.")
    
    # Display chat history if there are previous conversations
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### 📜 Previous Conversations")
        
        # Display chat history in reverse order (newest first)
        for past_query, past_answer in reversed(st.session_state.chat_history):
            with st.chat_message("user", avatar="🧑"):
                st.write(past_query)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(past_answer)
            
    st.markdown("</div>", unsafe_allow_html=True)

