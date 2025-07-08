import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from Dataset_preparation import CalligraphyDataset, get_transforms
from model_definitions import ImprovedCalligraphyFeatureExtractor, extract_all_features, find_similar_images
from sklearn.decomposition import PCA

def setup_and_run_feature_extraction(dataset_path, num_classes=None, batch_size=4):
    """
    Main function to setup model and extract features
    Optimized for your hardware constraints
    """
    
    # 1️⃣ Setup device (CPU for Intel HD Graphics 520)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2️⃣ Get proper transforms for ResNet18
    train_transform, val_transform = get_transforms()
    
    # 3️⃣ Create Dataset
    dataset = CalligraphyDataset(dataset_path, transform=val_transform)
    print(f"Dataset loaded: {len(dataset)} images")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Determine number of classes
    if num_classes is None:
        num_classes = len(dataset.class_to_idx) if hasattr(dataset, 'class_to_idx') else 10
    
    # 4️⃣ Create DataLoader (small batch size for your RAM)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0  # Use 0 for Windows/compatibility
    )
    
    # 5️⃣ Instantiate Model
    model = ImprovedCalligraphyFeatureExtractor(
        num_classes=num_classes,
        embedding_dim=256,
        freeze_backbone=True  # Essential for small datasets
    )
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model initialized with {num_classes} classes")
    print(f"Embedding dimension: {model.embedding_dim}")
    
    # 6️⃣ Extract features for all images
    print("Extracting features...")
    all_features, all_labels, all_paths = extract_all_features(model, dataloader, device)
    print(f"Extracted features shape: {all_features.shape}")
    
    # 7️⃣ Optional: Apply PCA for dimensionality reduction
    pca = None
    reduced_features = all_features
    
    if all_features.shape[0] > 50:  # Only if we have enough samples
        n_components = min(32, all_features.shape[0]-1)
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(all_features)
        print(f"PCA reduced features shape: {reduced_features.shape}")
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:5]}")
    
    return model, reduced_features, all_labels, all_paths, dataset, pca

def demo_similarity_search(features, paths, top_k=5):
    """
    Demonstrate similarity search functionality
    """
    if len(paths) == 0:
        print("No images to search")
        return
    
    print(f"\n=== Similarity Search Demo ===")
    
    # Use first image as query
    query_idx = 0
    query_features = features[query_idx]
    query_path = paths[query_idx]
    
    print(f"Query image: {query_path}")
    
    # Find similar images
    similar_paths, scores = find_similar_images(
        query_features, features, paths, top_k=top_k
    )
    
    print("Most similar images:")
    for i, (path, score) in enumerate(zip(similar_paths, scores)):
        print(f"  {i+1}. {path}: similarity = {score:.3f}")

def save_results(model, features, labels, paths, pca=None):
    """
    Save model and extracted features
    """
    # Save model state
    torch.save(model.state_dict(), 'calligraphy_resnet_model.pth')
    
    # Save features and metadata  
    np.save('calligraphy_features.npy', features)
    np.save('calligraphy_labels.npy', labels)
    
    # Save paths
    with open('image_paths.pkl', 'wb') as f:
        pickle.dump(paths, f)
    
    # Save PCA if used
    if pca is not None:
        with open('pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)
    
    print("\n=== Files Saved ===")
    print("- calligraphy_resnet_model.pth (model weights)")
    print("- calligraphy_features.npy (extracted features)")
    print("- calligraphy_labels.npy (labels)")
    print("- image_paths.pkl (image paths)")
    if pca:
        print("- pca_model.pkl (PCA model)")

# Main execution
if __name__ == "__main__":
    # Configuration
    dataset_path = '../output_images'  # Adjust this path
    batch_size = 4  # Small batch size for your hardware
    
    try:
        # Run feature extraction
        model, features, labels, paths, dataset, pca = setup_and_run_feature_extraction(
            dataset_path=dataset_path,
            batch_size=batch_size
        )
        
        # Demonstrate similarity search
        demo_similarity_search(features, paths)
        
        # Save results
        save_results(model, features, labels, paths, pca)
        
        print(f"\n=== Success! ===")
        print(f"Processed {len(paths)} images")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Ready for Stable Diffusion integration!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()