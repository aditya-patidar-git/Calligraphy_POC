from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from collections import defaultdict
import torchvision.transforms as transforms

class CalligraphyDataset(Dataset):
    def __init__(self, root_dir, transform=None, create_labels_from_dirs=True):
        """
        Args:
            root_dir: Path to dataset
            transform: Image transformations
            create_labels_from_dirs: If True, creates labels from directory names
        
        Directory structure:
            output_images/
                class_A/  (or image_0/)
                    shape_ellipse_0.png
                class_B/  (or image_1/)
                    shape_ellipse_0.png
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        if create_labels_from_dirs:
            self._create_labels_from_directories()
        else:
            self._load_all_images_single_class()
    
    def _create_labels_from_directories(self):
        """Create labels based on directory names."""
        class_names = sorted([d for d in os.listdir(self.root_dir) 
                             if os.path.isdir(os.path.join(self.root_dir, d))])
        
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        for class_name in class_names:
            class_path = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)
    
    def _load_all_images_single_class(self):
        """Load all images with dummy label (for feature extraction only)."""
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(0)  # Dummy label

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        
        try:
            # IMPORTANT: Convert to RGB for ResNet18 (needs 3 channels)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label, img_path  # Include path for debugging

    def __len__(self):
        return len(self.image_paths)
    
    def get_class_distribution(self):
        """Return distribution of classes."""
        class_counts = defaultdict(int)
        for label in self.labels:
            if hasattr(self, 'idx_to_class'):
                class_name = self.idx_to_class.get(label, f"class_{label}")
            else:
                class_name = f"class_{label}"
            class_counts[class_name] += 1
        return dict(class_counts)

def get_transforms(input_size=224):
    """
    Get transforms optimized for calligraphy images with ResNet18
    """
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(5),  # Small rotation for data augmentation
        transforms.RandomHorizontalFlip(p=0.3),  # Less likely for text
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform