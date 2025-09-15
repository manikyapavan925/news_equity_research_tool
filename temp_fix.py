        else:
            st.info("👆 Please load some articles first to start asking questions!")
            
            # Sample articles for demonstration
            st.markdown("**📚 Try these sample financial news URLs:**")
            sample_urls = [
                "https://finance.yahoo.com/news/",
                "https://www.cnbc.com/world/?region=world",
                "https://www.reuters.com/business/finance/"
            ]
            for url in sample_urls:
                st.code(url)
