# File: search_interface.py
import streamlit as st
import torch
import clip
import time
from PIL import Image, UnidentifiedImageError
from cache import SearchCache

def search_interface(model, preprocess, images_collection, captions_collection):
    st.header("Search Interface")

    # Add cache control to sidebar
    with st.sidebar:
        st.subheader("Cache Controls")
        if st.button("Clear Cache"):
            st.session_state.search_cache.clear()
            st.success("Cache cleared!")

    # Create tabs for different search types
    search_type = st.radio("Search Type", ["Text", "Image"])

    if search_type == "Text":
        query = st.text_input("Enter text query:")
        num_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
        
        if query and st.button("Search"):
            try:
                start_time = time.time()
                
                # Generate cache key for text query
                cache_key = st.session_state.search_cache.get_query_hash(query)
                
                # Try to get results from cache
                cached_results = st.session_state.search_cache.get(cache_key)
                
                if cached_results:
                    st.info("Results retrieved from cache")
                    results = cached_results
                    embedding_time = 0
                else:
                    # Measure embedding time
                    embed_start = time.time()
                    query_tokenized = clip.tokenize([query]).to(next(model.parameters()).device)
                    with torch.no_grad():
                        query_embedding = model.encode_text(query_tokenized).cpu().numpy()
                    embedding_time = time.time() - embed_start
                    
                    # Perform search
                    results = captions_collection.query(
                        query_embeddings=[query_embedding[0].tolist()],
                        n_results=num_results
                    )
                    
                    # Store in cache
                    st.session_state.search_cache.set(cache_key, results)
                    st.info("New search performed and cached")

                # Record performance metrics
                search_time = time.time() - start_time
                st.session_state.performance_monitor.record_search(
                    search_time=search_time,
                    embedding_time=embedding_time,
                    cache_hit=cached_results is not None,
                    query_type="text"
                )

                # Display results
                if results and 'metadatas' in results and 'documents' in results:
                    st.write(f"Found {len(results['metadatas'][0])} results in {search_time:.3f} seconds")
                    
                    # Create columns for grid layout
                    cols = st.columns(3)
                    for idx, (metadata, document) in enumerate(zip(results['metadatas'][0], results['documents'][0])):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            try:
                                st.image(
                                    metadata['image_path'],
                                    caption=f"Score: {results['distances'][0][idx]:.3f}",
                                    use_column_width=True
                                )
                                st.write(f"Caption: {document}")
                                st.write(f"Path: {metadata['image_path']}")
                                st.divider()
                            except FileNotFoundError:
                                st.warning(f"Image not found: {metadata['image_path']}")
                            except UnidentifiedImageError:
                                st.warning(f"Error loading image: {metadata['image_path']}")
                else:
                    st.write("No results found or invalid query.")

            except Exception as e:
                st.error(f"Error during text search: {str(e)}")

    else:  # Image search
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        num_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Search"):
                try:
                    start_time = time.time()
                    
                    # Process image and generate embedding
                    embed_start = time.time()
                    image = Image.open(uploaded_file)
                    input_tensor = preprocess(image).unsqueeze(0).to(next(model.parameters()).device)
                    
                    with torch.no_grad():
                        image_embedding = model.encode_image(input_tensor).cpu().numpy()
                    embedding_time = time.time() - embed_start
                    
                    # Generate cache key for image embedding
                    cache_key = st.session_state.search_cache.get_query_hash(image_embedding)
                    
                    # Try to get results from cache
                    cached_results = st.session_state.search_cache.get(cache_key)
                    
                    if cached_results:
                        st.info("Results retrieved from cache")
                        results = cached_results
                    else:
                        # Perform search
                        results = images_collection.query(
                            query_embeddings=[image_embedding[0].tolist()],
                            n_results=num_results
                        )
                        
                        # Store in cache
                        st.session_state.search_cache.set(cache_key, results)
                        st.info("New search performed and cached")

                    # Record performance metrics
                    search_time = time.time() - start_time
                    st.session_state.performance_monitor.record_search(
                        search_time=search_time,
                        embedding_time=embedding_time,
                        cache_hit=cached_results is not None,
                        query_type="image"
                    )

                    # Display results
                    if results and 'metadatas' in results:
                        st.write(f"Found {len(results['metadatas'][0])} results in {search_time:.3f} seconds")
                        
                        # Create columns for grid layout
                        cols = st.columns(3)
                        for idx, metadata in enumerate(results['metadatas'][0]):
                            col_idx = idx % 3
                            with cols[col_idx]:
                                try:
                                    st.image(
                                        metadata['image_path'],
                                        caption=f"Score: {results['distances'][0][idx]:.3f}",
                                        use_column_width=True
                                    )
                                    st.write(f"Caption: {metadata['caption']}")
                                    st.write(f"Path: {metadata['image_path']}")
                                    st.divider()
                                except FileNotFoundError:
                                    st.warning(f"Image not found: {metadata['image_path']}")
                                except UnidentifiedImageError:
                                    st.warning(f"Error loading image: {metadata['image_path']}")
                    else:
                        st.write("No results found.")

                except Exception as e:
                    st.error(f"Error during image search: {str(e)}")