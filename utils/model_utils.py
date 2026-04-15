import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import timm
import torch.nn as nn

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        # Load EfficientNet-B0 feature extractor
        self.effnet = models.efficientnet_b0(weights=None)
        # EfficientNet-B0 pooling outputs 1280 features. Remove classifier.
        self.effnet.classifier = nn.Identity()
        
        # Load MobileViTv2 feature extractor from timm
        # num_classes=0 returns pooled features (512 features)
        self.mobilevit = timm.create_model('mobilevitv2_100', pretrained=False, num_classes=0)
        
        self.fc = nn.Linear(1280 + 512, num_classes)
        
    def forward(self, x):
        f1 = self.effnet(x)
        f2 = self.mobilevit(x)
        features = torch.cat((f1, f2), dim=1)
        return self.fc(features)

def create_model(model_name, num_classes):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet_b7":
        model = models.efficientnet_b7(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilevitv2":
        model = timm.create_model('mobilevitv2_100', pretrained=False, num_classes=num_classes)
    elif model_name == "hybrid":
        model = HybridModel(num_classes)
    else:
        # Default fallback
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

HARDCODED_CLASS_NAMES = ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf', 'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf early blight', 'Potato leaf late blight', 'Raspberry leaf', 'Soyabean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf', 'grape leaf', 'grape leaf black rot']

def load_model(model_path):
    """
    Loads the trained model weights and class names dynamically, resolving raw PyTorch saves safely.
    """
    import os
    try:
        # Load the saved state dict
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Ensure we can load both structured dictionary checkpoints or bare OrderDict state_dicts
        if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            model_name = checkpoint.get('model_name', 'hybrid')
            state_dict = checkpoint['state_dict']
        else:
            class_names = HARDCODED_CLASS_NAMES
            filename = os.path.basename(model_path).lower()
            if 'efficientnet_b0' in filename:
                model_name = 'efficientnet_b0'
            elif 'efficientnet_b7' in filename:
                model_name = 'efficientnet_b7'
            elif 'mobilevit' in filename:
                model_name = 'mobilevitv2'
            else:
                model_name = 'hybrid'
            
            state_dict = checkpoint
            
        print(f"Instantiating {model_name} architecture...")
        # Recreate the model structure
        model = create_model(model_name, len(class_names))
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def transform_image(image_bytes):
    """
    Transforms the input image for the model.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension

def get_prediction(model, class_names, image_bytes):
    """
    Returns the predicted class name and confidence score.
    """
    tensor = transform_image(image_bytes).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = class_names[predicted_idx.item()]
        
        return predicted_class, confidence.item()
