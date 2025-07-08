import torch
import numpy as np
import pickle
from model_definitions import ImprovedCalligraphyFeatureExtractor
from PIL import Image
import torchvision.transforms as transforms

def load_saved_model_and_features():
    """
    Example of how to load and use all the generated files
    """
    
    # 1️⃣ Load the trained model
    model = ImprovedCalligraphyFeatureExtractor(num_classes=11, embedding_dim=256)
    model.load_state_dict(torch.load('calligraphy_resnet_model.pth', map_location='cpu'))
    model.eval()
    print("✅ Model loaded successfully")
    
    # 2️⃣ Load extracted features
    features = np.load('calligraphy_features.npy')
    labels = np.load('calligraphy_labels.npy')
    print(f"✅ Features loaded: {features.shape}")
    print(f"✅ Labels loaded: {labels.shape}")
    
    # 3️⃣ Load image paths
    with open('image_paths.pkl', 'rb') as f:
        image_paths = pickle.load(f)
    print(f"✅ Image paths loaded: {len(image_paths)} images")
    
    # 4️⃣ Load PCA model (if it exists)
    try:
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
        print(f"✅ PCA model loaded: reduces to {pca.n_components_} dimensions")
    except FileNotFoundError:
        pca = None
        print("ℹ️ No PCA model found (features not reduced)")
    
    return model, features, labels, image_paths, pca

def extract_features_from_new_image(model, image_path, pca=None):
    """
    Use the saved model to extract features from a new image
    """
    # Setup transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Extract features
    with torch.no_grad():
        features = model.extract_features(image_tensor)
        features = features.cpu().numpy().flatten()
    
    # Apply PCA if available
    if pca is not None:
        features = pca.transform([features])[0]
    
    return features

def find_most_similar_images(new_image_path, model, all_features, image_paths, pca=None, top_k=5):
    """
    Find images most similar to a new input image
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Extract features from new image
    new_features = extract_features_from_new_image(model, new_image_path, pca)
    
    # Calculate similarities
    similarities = cosine_similarity([new_features], all_features)[0]
    
    # Get top matches
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n🔍 Most similar images to: {new_image_path}")
    for i, idx in enumerate(top_indices):
        similarity_score = similarities[idx]
        similar_image_path = image_paths[idx]
        print(f"  {i+1}. {similar_image_path} (similarity: {similarity_score:.3f})")
    
    return top_indices, similarities[top_indices]

def inspect_dataset_statistics(features, labels, image_paths):
    """
    Analyze your dataset using the saved files
    """
    print("\n📊 Dataset Statistics:")
    print(f"Total images: {len(image_paths)}")
    print(f"Feature dimensions: {features.shape[1]}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} images")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {np.mean(features):.3f}")
    print(f"  Std: {np.std(features):.3f}")
    print(f"  Min: {np.min(features):.3f}")
    print(f"  Max: {np.max(features):.3f}")

def prepare_features_for_stable_diffusion(features, image_paths, output_file='sd_conditioning.npy'):
    """
    Prepare features in format suitable for Stable Diffusion conditioning
    """
    # Normalize features for stable diffusion
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Save for Stable Diffusion use
    conditioning_data = {
        'features': normalized_features,
        'image_paths': image_paths,
        'feature_dim': normalized_features.shape[1]
    }
    
    np.save(output_file, conditioning_data)
    print(f"✅ Stable Diffusion conditioning data saved to: {output_file}")
    
    return normalized_features

# Example usage
if __name__ == "__main__":
    # Load everything
    model, features, labels, image_paths, pca = load_saved_model_and_features()
    
    # Analyze dataset
    inspect_dataset_statistics(features, labels, image_paths)
    
    # Example: Find similar images to the first image in dataset
    if len(image_paths) > 0:
        query_image = image_paths[38]
        find_most_similar_images(query_image, model, features, image_paths, pca)
    
    # Prepare for Stable Diffusion
    sd_features = prepare_features_for_stable_diffusion(features, image_paths)
    
    print("\n🎉 All files loaded and processed successfully!")
    print("\nNext steps:")
    print("1. Use features for similarity search")
    print("2. Feed normalized features to Stable Diffusion")
    print("3. Use model to extract features from new images")