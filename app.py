import streamlit as st
import chromadb
import torch
import sys
import clip
from cache import SearchCache
from database_manager import database_management_interface
from search_interface import search_interface
from performance_metrics import performance_metrics_interface

# Initialize models and ChromaDB client
@st.cache_resource
def load_models():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()  # Set model to evaluation mode
        
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Always create or get collections
        images_collection = chroma_client.get_or_create_collection("images")
        captions_collection = chroma_client.get_or_create_collection("captions")
        
        return model, preprocess, chroma_client, images_collection, captions_collection
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise e

def main():
    st.set_page_config(
        page_title="Multimedia Search Engine",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Multimedia Search Engine")
    
    # Initialize performance monitor in session state if not exists
    if 'search_cache' not in st.session_state:
        st.session_state.search_cache = SearchCache(max_size=100)
    
    try:
        # Load models and collections
        model, preprocess, chroma_client, images_collection, captions_collection = load_models()
        
        # Add tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["Search", "Database Management", "Performance Metrics"])
        
        with tab1:
            search_interface(model, preprocess, images_collection, captions_collection)
        
        with tab2:
            database_management_interface(model, preprocess, images_collection, captions_collection)
            
        with tab3:
            performance_metrics_interface()
            
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page and try again. If the error persists, check the logs for more details.")

if __name__ == "__main__":
    main()