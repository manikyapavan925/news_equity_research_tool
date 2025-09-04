# News Equity Research Tool

A tool that helps analyze financial news articles and provides a Q&A interface powered by LangChain and OpenAI.

## Features

- URL processing for financial news articles
- Vector storage using FAISS for efficient retrieval
- Q&A functionality powered by OpenAI
- Interactive UI using Streamlit

## Setup

1. Clone the repository
2. Install dependencies:
```bash
poetry install
```

3. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

4. Run the application:
```bash
poetry run streamlit run src/app.py
```

## Usage

1. Enter a financial news URL (MoneyControl, Yahoo Finance, etc.)
2. Click "Process URL" to analyze the content
3. Ask questions about the article in the chat interface

## Technologies Used

- Streamlit for UI
- LangChain for document processing and Q&A
- FAISS for vector storage
- OpenAI for language model
