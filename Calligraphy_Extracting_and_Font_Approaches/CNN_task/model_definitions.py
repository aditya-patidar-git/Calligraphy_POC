import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class ImprovedCalligraphyFeatureExtractor(nn.Module):
    """
    Lightweight feature extractor optimized for small calligraphy datasets
    Uses pre-trained ResNet18 backbone with frozen features + small classifier
    """
    
    def __init__(self, num_classes=10, embedding_dim=256, freeze_backbone=True):
        super().__init__()
        
        # Use pre-trained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze backbone if specified (recommended for small datasets)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the feature dimension from ResNet18
        backbone_output_dim = self.backbone.fc.in_features
        
        # Replace the final classifier with our custom head
        self.backbone.fc = nn.Identity()  # Remove original classifier
        
        # Custom feature projection head
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_output_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim // 2, num_classes)
        
        # Store dimensions
        self.embedding_dim = embedding_dim // 2
        self.num_classes = num_classes
    
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone(x)  # [batch_size, 512]
        
        # Project to embedding space
        embeddings = self.feature_head(backbone_features)  # [batch_size, embedding_dim//2]
        
        # Classification logits
        logits = self.classifier(embeddings)  # [batch_size, num_classes]
        
        return embeddings, logits
    
    def extract_features(self, x):
        """Extract only embeddings without classification"""
        with torch.no_grad():
            embeddings, _ = self.forward(x)
        return embeddings

# Keep your original traditional feature extractor as backup
class LightweightCalligraphyFeatureExtractor:
    """Traditional feature extractor (your original approach - good for CPU-only)"""
    
    def __init__(self, use_traditional_features=True, use_tiny_cnn=False):
        self.use_traditional_features = use_traditional_features
        self.use_tiny_cnn = use_tiny_cnn
        self.pca = None
        self.feature_scaler = None
        
        if use_tiny_cnn:
            self.cnn_model = self._build_tiny_cnn()
    
    def _build_tiny_cnn(self):
        """Ultra-lightweight CNN for CPU inference."""
        return nn.Sequential(
            # First conv block - reduce to 64x64
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 4x4
            
            # Flatten and reduce
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)  # Final embedding size
        )
    
    # ... rest of your original traditional feature extraction methods ...

def extract_all_features(model, dataloader, device):
    """
    Extract features for all images in the dataset
    """
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for batch_images, batch_labels, batch_paths in dataloader:
            batch_images = batch_images.to(device)
            
            # Extract features
            embeddings = model.extract_features(batch_images)
            
            all_features.append(embeddings.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            all_paths.extend(batch_paths)
    
    return np.vstack(all_features), np.array(all_labels), all_paths

def find_similar_images(query_features, all_features, all_paths, top_k=5):
    """Find most similar images using cosine similarity."""
    if len(query_features.shape) == 1:
        query_features = query_features.reshape(1, -1)
    
    similarities = cosine_similarity(query_features, all_features)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    similar_paths = [all_paths[i] for i in top_indices]
    similar_scores = similarities[top_indices]
    
    return similar_paths, similar_scores