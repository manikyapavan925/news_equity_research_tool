import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import os
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Import advanced search with error handling
try:
    from app.advanced_search import AdvancedSearchEngine
except ImportError:
    st.error("Advanced Search module not found. Please ensure the module is properly installed.")
    AdvancedSearchEngine = None

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="AI News Research Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .main-header { text-align: center; padding: 1rem 0; }
    .response-box { 
        background-color: #f0f2f6; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'current_urls' not in st.session_state:
    st.session_state.current_urls = []

# Header
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🤖 AI News Research Assistant")
st.markdown("*Intelligent chatbot with web search capabilities*")
st.markdown("</div>", unsafe_allow_html=True)

# LLM Functions (from your existing code)
def get_advanced_llm():
    """Load and return advanced LLM model pipeline"""
    try:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        qa_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=400,
            temperature=0.7,
            do_sample=True
        )
        
        return qa_pipeline, "google/flan-t5-base"
        
    except Exception as e:
        print(f"Failed to load Flan-T5: {e}")
        return None, None

def generate_llm_response(question, context, model_pipeline, model_name):
    """Generate response using LLM with timeout protection"""
    try:
        import signal
        from contextlib import contextmanager

        @contextmanager
        def timeout_context(seconds):
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM generation timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)

        # Create prompt
        if context:
            prompt = f"Answer this question based on the context: {question}\n\nContext: {context[:1000]}...\n\nAnswer:"
        else:
            prompt = f"Answer this question: {question}\n\nAnswer:"

        # Generate with timeout
        with timeout_context(15):
            response = model_pipeline(
                prompt,
                max_length=150,
                num_return_sequences=1,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.2,
                early_stopping=True,
                pad_token_id=model_pipeline.tokenizer.eos_token_id
            )

        # Extract response
        if isinstance(response, list) and len(response) > 0:
            if 'generated_text' in response[0]:
                answer = response[0]['generated_text'].strip()
            else:
                answer = str(response[0]).strip()
        else:
            answer = str(response).strip()

        return f"**Answer:** {answer}"

    except TimeoutError:
        return "**⚡ Fast Analysis:** The AI model took too long to respond. Please try a shorter question."
    except Exception as e:
        return f"**Analysis Result:** The AI model encountered an error: {str(e)}"

def get_intelligent_response(question, context=""):
    """
    Intelligent response system with quality-based fallback to web search.
    Flow: AI Response → Quality Check → If Poor → Advanced Search → Return Best Result
    """
    try:
        # Step 1: Try AI-powered response first
        llm_result = get_advanced_llm()
        if llm_result and llm_result[0] is not None:
            model_pipeline, model_name = llm_result
            ai_response = generate_llm_response(question, context, model_pipeline, model_name)
            
            # Step 2: Simple quality check
            if ai_response and len(ai_response) > 50 and "error" not in ai_response.lower():
                return ai_response
        
        # Step 3: Fallback to Advanced Search if AI response is poor/failed
        print("AI response inadequate, falling back to Advanced Search...")
        return get_advanced_search_response(question)
        
    except Exception as e:
        print(f"Error in AI response: {e}")
        return get_advanced_search_response(question)

def get_advanced_search_response(question):
    """Advanced search with Tavily -> DuckDuckGo fallback"""
    try:
        if AdvancedSearchEngine is None:
            return "🔍 **Web Search Results:** Search functionality is currently unavailable. Please try again later."
        
        search_engine = AdvancedSearchEngine()
        search_results = search_engine.search_comprehensive(question)
        
        if search_results and search_results.get('results'):
            return format_search_results(question, search_results)
        else:
            return f"🔍 **Search Results:** I searched for information about '{question}' but couldn't find comprehensive results. Please try rephrasing your question."
            
    except Exception as e:
        return f"🔍 **Search Error:** Unable to search for information: {str(e)}"

def format_search_results(question, search_results):
    """Format search results into a readable response"""
    results = search_results.get('results', [])
    if not results:
        return f"🔍 **No Results Found:** I couldn't find information about '{question}'."
    
    response = f"🔍 **Search Results for:** {question}\n\n"
    
    # Add Tavily's advanced answer if available
    if 'answer' in search_results:
        response += f"**📋 Summary:** {search_results['answer']}\n\n"
    
    # Add top results
    response += "**📰 Sources:**\n"
    for i, result in enumerate(results[:5], 1):
        title = result.get('title', 'No Title')
        url = result.get('url', '')
        snippet = result.get('content', result.get('snippet', ''))[:200]
        
        response += f"{i}. **{title}**\n"
        if snippet:
            response += f"   {snippet}...\n"
        if url:
            response += f"   🔗 [Read more]({url})\n\n"
    
    return response

# Sidebar for article management
with st.sidebar:
    st.header("📰 Article Management")
    
    # URL input
    urls_input = st.text_area(
        "📎 Add Article URLs:",
        placeholder="https://example.com/article1\nhttps://example.com/article2",
        height=100
    )
    
    if st.button("📥 Load Articles"):
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            valid_urls = [url for url in urls if url.startswith('http')]
            
            if valid_urls:
                with st.spinner(f"Loading {len(valid_urls)} articles..."):
                    for url in valid_urls:
                        if url not in st.session_state.current_urls:
                            # Simple article loading (you can enhance this)
                            st.session_state.articles.append({
                                'url': url,
                                'title': f'Article from {url.split("/")[2]}',
                                'content': f'Content from {url}',
                                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                            })
                            st.session_state.current_urls.append(url)
                
                st.success(f"✅ Loaded {len(valid_urls)} articles!")
                st.rerun()
            else:
                st.error("❌ Please provide valid URLs starting with http:// or https://")
    
    # Display loaded articles
    if st.session_state.articles:
        st.markdown("---")
        st.markdown(f"**📊 Loaded Articles: {len(st.session_state.articles)}**")
        for i, article in enumerate(st.session_state.articles[-3:], 1):  # Show last 3
            st.markdown(f"• {article['title'][:30]}...")
    
    # Clear button
    if st.button("🗑️ Clear All"):
        st.session_state.articles = []
        st.session_state.current_urls = []
        st.success("All articles cleared!")
        st.rerun()

# Main chat interface
st.header("💬 Chat with AI Assistant")

# Question input
question = st.text_input(
    "💬 Ask me anything:",
    placeholder="e.g., What are Microsoft's AI plans? Latest news on Tesla? Summarize the loaded articles..."
)

st.info("🤖 **Intelligent Assistant** - I'll analyze your loaded articles first, then search the web if needed for comprehensive answers.")

# Chat functionality
if question and st.button("💬 Ask", type="primary"):
    with st.spinner("🤖 Thinking..."):
        try:
            start_time = time.time()
            
            # Prepare context from loaded articles
            context = ""
            if st.session_state.articles:
                context_parts = []
                for article in st.session_state.articles[-5:]:  # Use last 5 articles
                    content = article.get('content', '')[:500]  # Limit content
                    context_parts.append(f"Article: {article.get('title', 'No Title')}\nContent: {content}")
                context = "\n---\n".join(context_parts)
            
            # Get intelligent response
            answer = get_intelligent_response(question, context)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Display response
            st.success(f"🤖 Response ready in {duration:.1f}s")
            
            # Response in a nice box
            st.markdown("<div class='response-box'>", unsafe_allow_html=True)
            st.markdown("### 💬 Answer")
            st.markdown(answer)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show source indication
            if context and any(keyword in answer.lower() for keyword in ["based on", "according to", "article"]):
                st.info("📄 Answer includes information from your loaded articles")
            elif "🔍" in answer:
                st.info("🌐 Answer includes web search results")
        
        except Exception as e:
            st.error(f"❌ Failed to generate response: {str(e)}")
            st.info("💡 Please try rephrasing your question or check your connection.")

# Footer
st.markdown("---")
st.markdown("*Powered by AI with advanced web search capabilities*")
