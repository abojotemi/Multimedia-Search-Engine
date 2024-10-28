# File: database_manager.py
import streamlit as st
from pathlib import Path
from PIL import Image
import clip
import torch
import pandas as pd
import os
from typing import List, Dict, Optional
import time
# Add these imports at the top of database_manager.py
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def get_available_datasets():
    """Returns a list of available torchvision datasets"""
    return [
        "CIFAR10",
        "CIFAR100",
        "FashionMNIST",
        "STL10",
        "SVHN",
        "Places365",
    ]

def load_torchvision_dataset(dataset_name: str, root_dir: str = "./data", split: str = "train", max_samples: int = 1000):
    """
    Load a dataset from torchvision
    
    Args:
        dataset_name: Name of the dataset to load
        root_dir: Directory to store the dataset
        split: 'train' or 'test' split
        max_samples: Maximum number of samples to load
    
    Returns:
        dataset object
    """
    # Define basic transform
    transform = transforms.Compose([
        transforms.Resize(224),  # CLIP expects 224x224 images
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.ToPILImage(),  # Convert back to PIL for saving
    ])
    
    # Get dataset class from torchvision
    dataset_class = getattr(torchvision.datasets, dataset_name)
    
    # Load dataset
    try:
        dataset = dataset_class(
            root=root_dir,
            train=(split == "train"),
            download=True,
            transform=transform
        )
        return dataset
    except Exception as e:
        raise Exception(f"Error loading dataset {dataset_name}: {str(e)}")

def import_torchvision_dataset(
    dataset_name: str,
    model,
    preprocess,
    images_collection,
    captions_collection,
    max_samples: int = 1000,
    storage_dir: Path = Path("stored_images")
):
    """
    Import images from a torchvision dataset into the database
    
    Args:
        dataset_name: Name of the torchvision dataset
        model: CLIP model
        preprocess: CLIP preprocessing function
        images_collection: ChromaDB collection for images
        captions_collection: ChromaDB collection for captions
        max_samples: Maximum number of samples to import
        storage_dir: Directory to store the images
    """
    try:
        # Load dataset
        dataset = load_torchvision_dataset(dataset_name, max_samples=max_samples)
        
        # Create dataset directory
        dataset_dir = storage_dir / dataset_name.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        successful = []
        failed = []
        
        # Process samples
        for idx in range(min(len(dataset), max_samples)):
            try:
                # Update progress
                progress = (idx + 1) / min(len(dataset), max_samples)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {idx + 1}/{min(len(dataset), max_samples)}...")
                
                # Get image and label
                image, label = dataset[idx]
                class_name = dataset.classes[label] if hasattr(dataset, 'classes') else f"class_{label}"
                
                # Save image
                image_path = dataset_dir / f"{dataset_name.lower()}_{idx}_{class_name}.jpg"
                image.save(image_path)
                
                # Create caption
                caption = f"{dataset_name} image of {class_name}"
                
                # Add to database
                if add_image_to_database(image_path, caption, model, preprocess, 
                                       images_collection, captions_collection):
                    successful.append(str(image_path))
                else:
                    failed.append(str(image_path))
                    
            except Exception as e:
                failed.append(f"Image {idx} ({str(e)})")
                
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return {"successful": successful, "failed": failed}
        
    except Exception as e:
        raise Exception(f"Error importing dataset {dataset_name}: {str(e)}")

    # [Rest of the code remains the same...]


def setup_image_storage():
    """Create a directory to store uploaded images if it doesn't exist"""
    storage_dir = Path("stored_images")
    storage_dir.mkdir(exist_ok=True)
    return storage_dir

def validate_image_file(file_path: str) -> bool:
    """Validate if file is a supported image format"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def add_image_to_database(image_path, caption, model, preprocess, images_collection, captions_collection):
    """Add a single image and its caption to the database"""
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        input_tensor = preprocess(image).unsqueeze(0).to(next(model.parameters()).device)
        
        # Generate embeddings
        with torch.no_grad():
            image_embedding = model.encode_image(input_tensor).cpu().numpy()[0]
            caption_tokenized = clip.tokenize([caption]).to(next(model.parameters()).device)
            caption_embedding = model.encode_text(caption_tokenized).cpu().numpy()[0]
        
        # Add to collections
        image_id = str(Path(image_path).stem)
        
        images_collection.add(
            embeddings=[image_embedding.tolist()],
            metadatas=[{"image_path": str(image_path), "caption": caption}],
            ids=[f"img_{image_id}"]
        )
        
        captions_collection.add(
            embeddings=[caption_embedding.tolist()],
            metadatas=[{"image_path": str(image_path), "caption": caption}],
            documents=[caption],
            ids=[f"cap_{image_id}"]
        )
        
        return True
    except Exception as e:
        st.error(f"Error adding image to database: {str(e)}")
        return False

def database_management_interface(model, preprocess, images_collection, captions_collection):
    """Interface for managing the database"""
    st.header("Database Management")
    
    # Setup storage
    storage_dir = setup_image_storage()
    
    # Create tabs for different import methods
    tabs = st.tabs(["Single Upload", "Batch Import", "Torchvision Datasets", "View Database"])
    
    # Single Upload Tab
    with tabs[0]:
        st.subheader("Single Image Upload")
        st.write("Upload individual images with custom captions")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            key="single_upload"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(uploaded_file, caption="Preview", use_column_width=True)
            
            with col2:
                caption = st.text_area("Enter image caption", key="single_caption")
                
                if st.button("Add to Database", key="single_add"):
                    # Save image to storage
                    image_path = storage_dir / uploaded_file.name
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if add_image_to_database(
                        image_path, 
                        caption, 
                        model, 
                        preprocess, 
                        images_collection, 
                        captions_collection
                    ):
                        st.success(f"Successfully added {uploaded_file.name} to database!")
    
    # Batch Import Tab
    with tabs[1]:
        st.subheader("Batch Import")
        st.write("Import multiple images from a directory")
        
        directory_path = st.text_input("Enter directory path:")
        csv_file = st.file_uploader(
            "Upload CSV with captions (optional)", 
            type=["csv"],
            key="batch_csv"
        )
        
        if directory_path:
            if not os.path.exists(directory_path):
                st.error("Directory not found!")
            else:
                st.write(f"Found directory: {directory_path}")
                image_files = list(Path(directory_path).glob("*.[jp][pn][gf]"))
                st.write(f"Number of images found: {len(image_files)}")
                
                if st.button("Start Batch Import"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process CSV if provided
                    captions_dict = {}
                    if csv_file:
                        df = pd.read_csv(csv_file)
                        captions_dict = dict(zip(df['image_filename'], df['caption']))
                    
                    # Process images
                    successful = []
                    failed = []
                    
                    for idx, image_path in enumerate(image_files):
                        try:
                            # Update progress
                            progress = (idx + 1) / len(image_files)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {image_path.name}...")
                            
                            caption = captions_dict.get(image_path.name, image_path.stem)
                            
                            if add_image_to_database(
                                image_path, 
                                caption, 
                                model, 
                                preprocess, 
                                images_collection, 
                                captions_collection
                            ):
                                successful.append(image_path.name)
                            else:
                                failed.append(image_path.name)
                                
                        except Exception as e:
                            failed.append(f"{image_path.name} ({str(e)})")
                    
                    # Show results
                    if successful:
                        st.success(f"Successfully imported {len(successful)} images")
                    if failed:
                        st.error(f"Failed to import {len(failed)} images")
    
    # Torchvision Datasets Tab
    with tabs[2]:
        st.subheader("Import Torchvision Dataset")
        st.write("Import images from standard computer vision datasets")
        
        dataset_name = st.selectbox(
            "Select Dataset:",
            ["CIFAR10", "CIFAR100", "FashionMNIST"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_samples = st.number_input(
                "Maximum samples:", 
                min_value=1,
                max_value=10000,
                value=100
            )
        with col2:
            split = st.radio("Dataset split:", ["train", "test"])
        
        if st.button("Import Dataset"):
            try:
                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load dataset
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.ToPILImage()
                ])
                
                dataset_class = getattr(torchvision.datasets, dataset_name)
                dataset = dataset_class(
                    root="./data",
                    train=(split == "train"),
                    download=True,
                    transform=transform
                )
                
                # Process samples
                successful = []
                failed = []
                
                for idx in range(min(len(dataset), max_samples)):
                    try:
                        progress = (idx + 1) / min(len(dataset), max_samples)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing image {idx + 1}/{min(len(dataset), max_samples)}...")
                        
                        image, label = dataset[idx]
                        class_name = dataset.classes[label] if hasattr(dataset, 'classes') else f"class_{label}"
                        
                        # Save image
                        image_path = storage_dir / f"{dataset_name.lower()}_{idx}_{class_name}.jpg"
                        image.save(image_path)
                        
                        caption = f"{dataset_name} image of {class_name}"
                        
                        if add_image_to_database(
                            image_path,
                            caption,
                            model,
                            preprocess,
                            images_collection,
                            captions_collection
                        ):
                            successful.append(str(image_path))
                        else:
                            failed.append(str(image_path))
                            
                    except Exception as e:
                        failed.append(f"Image {idx} ({str(e)})")
                
                # Show results
                if successful:
                    st.success(f"Successfully imported {len(successful)} images")
                if failed:
                    st.error(f"Failed to import {len(failed)} images")
                    
            except Exception as e:
                st.error(f"Error importing dataset: {str(e)}")
    
    # View Database Tab
    with tabs[3]:
        st.subheader("View Database Contents")
        number = st.number_input("Number of images to display:", min_value=1, max_value=1000)
        randomize = st.checkbox("Randomize")
        if st.button("Load Database Contents"):
            try:
                # Get all entries from the database
                results = images_collection.get()
                
                if results and results['metadatas']:
                    total_entries = len(results['metadatas'])
                    st.write(f"Total entries: {total_entries}")
                    
                        
                    # Display entries
                    for idx, metadata in enumerate(results['metadatas'][:number]):
                        if randomize:
                            idx = np.random.choice(total_entries)
                            metadata = results['metadatas'][idx]
                            
                        with st.container():
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                try:
                                    st.image(
                                        metadata['image_path'],
                                        caption=f"Image {idx + 1}",
                                        use_column_width=True
                                    )
                                except Exception as e:
                                    st.error(f"Error loading image: {str(e)}")
                            
                            with col2:
                                st.write("Caption:", metadata['caption'])
                                st.write("Path:", metadata['image_path'])
                            
                            st.divider()
                else:
                    st.info("No entries found in the database.")
                    
            except Exception as e:
                st.error(f"Error loading database contents: {str(e)}")
