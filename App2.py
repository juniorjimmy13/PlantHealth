# App.py - Complete Streamlit Plant Disease Classification App
import time
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

# =============================================================================
# MODEL DEFINITION - MUST BE EXACTLY THE SAME AS TRAINING
# =============================================================================

class OptimizedCNN(nn.Module):
    def __init__(self, K, dropout_rate=0.5):
        super(OptimizedCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1 - Reduced channels for memory efficiency
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # inplace=True saves memory
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 4 - Reduced from 256 to 192 for memory efficiency
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Global Average Pooling instead of large fully connected layers
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Smaller fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(192, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, K)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =============================================================================
# CONFIGURATION
# =============================================================================

# Update these with your actual values
MODEL_PATH = "tip.pt"  # Your model file path
NUM_CLASSES = 38  # Update with your actual number of classes

# Update with your actual class names
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# =============================================================================
# MODEL LOADING WITH MULTIPLE STRATEGIES
# =============================================================================


@st.cache_resource
def load_model():
    """Load the trained model with multiple fallback strategies"""
    # Strategy 1: Try with safe_globals
    try:
        model = OptimizedCNN(NUM_CLASSES)
        with torch.serialization.safe_globals([OptimizedCNN]):
            if MODEL_PATH.endswith('tip.pt'):
                # If it's a full model file
                loaded_model = torch.load(MODEL_PATH, map_location='cpu',weights_only=False)
                if isinstance(loaded_model, nn.Module):
                    model = loaded_model
                else:
                    model.load_state_dict(loaded_model,)
            else:
                # If it's just state dict
                state_dict = torch.load(MODEL_PATH, map_location='cpu')
                model.load_state_dict(state_dict)
        
        model.eval()       
        return model
        
    except Exception as e1:
        st.warning(f"⚠️ Safe globals failed: {str(e1)[:100]}...")
        
        # Strategy 2: Try with weights_only=False (less secure but often works)
        try:
            model = OptimizedCNN(NUM_CLASSES)
            if MODEL_PATH.endswith('_3050ti.pt'):
                loaded_model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
                if isinstance(loaded_model, nn.Module):
                    model = loaded_model
                else:
                    model.load_state_dict(loaded_model)
            else:
                state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
                model.load_state_dict(state_dict)
            
            model.eval()
            global sv2
            sv2 = True
            st.success("✅ Model loaded with weights_only=False!")
            st.warning("⚠️ Using less secure loading method. This is OK if you trust the model source.")
            return model
            
        except Exception as e2:
            st.error(f"❌ Both loading strategies failed!")
            st.error(f"Error 1 (safe_globals): {e1}")
            st.error(f"Error 2 (weights_only=False): {e2}")
            
            # Show debugging info
            with st.expander("🔍 Debugging Information"):
                st.write("**Model path exists:**", MODEL_PATH)
                try:
                    import os
                    st.write("**File exists:**", os.path.exists(MODEL_PATH))
                    if os.path.exists(MODEL_PATH):
                        st.write("**File size:**", f"{os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
                except:
                    pass
                
                st.write("**PyTorch version:**", torch.__version__)
                st.write("**Expected classes:**", NUM_CLASSES)
                st.write("**Class names length:**", len(CLASS_NAMES))
            
            return None

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image):
    """Apply the same preprocessing used during training"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_disease(model, image_tensor, class_names):
    """Generate prediction with confidence scores"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_prob, top_class = torch.topk(probabilities, min(5, len(class_names)))
    
    return {
        'predicted_class': class_names[top_class[0][0]],
        'confidence': top_prob[0][0].item(),
        'top_predictions': [(class_names[top_class[0][i]], 
                           top_prob[0][i].item()) for i in range(len(top_class[0]))]
    }

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Plant Disease Classifier",
        page_icon="🌿",
        layout="wide"
    )
    
    # Header
    st.title("🌿 Plant Disease Classification System")
    st.markdown("Upload an image of a plant leaf to detect potential diseases using our AI model")
    
    # Sidebar info
    sidebar = st.sidebar
    with sidebar:
        st.header("ℹ️ About")
        st.write(f"**Model Classes:** {NUM_CLASSES}")
        st.write(f"**PyTorch Version:** {torch.__version__}")
        st.write("**Model Architecture:** Optimized CNN")
<<<<<<< HEAD
=======
        st.markdown("### Supported Plants")
        st.markdown("""
        - Tomato
        - Potato  
        - Corn
        - Apple
        - Bell Pepper
        - Cherry
        - Grape
        - Peach
        - Strawberry
        """)
        
>>>>>>> ab57c46f648e3e16bcd9c157a3277d0b3e7631e5
        if st.button("🔄 Reload Model"):
            st.cache_resource.clear()
            st.rerun()
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Could not load the model. Please check the troubleshooting steps below.")
        
        with st.expander("🛠️ Troubleshooting Steps"):
            st.markdown("""
            **Common fixes:**
            
            1. **Check model path**: Make sure `MODEL_PATH` points to your actual model file
            
            2. **Verify model architecture**: The `OptimizedCNN` class must match exactly what you used during training
            
            3. **Check number of classes**: Update `NUM_CLASSES` to match your trained model
            
            4. **Model file format**: 
               - If you saved with `torch.save(model.state_dict(), ...)` → use the .pt file
               - If you saved with `torch.save(model, ...)` → use the _full.pt file
            
            5. **PyTorch version**: This app works with PyTorch 2.0+
            
            **Quick test:**
            ```python
            # Test if you can load the model manually
            import torch
            model = OptimizedCNN(38)  # Your number of classes
            state_dict = torch.load('your_model.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
            ```
            """)
        
        st.stop()
    
    # Model loaded successfully - show the app

    with sidebar:
        status = st.empty()
        status.info("Loading the model... Please wait.",icon="⏳")
        time.sleep(3)  # Simulate loading time
        status.empty()
        
    st.success(f"🎯 Model loaded! Ready to classify {NUM_CLASSES} different plant diseases.")
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption="Your plant leaf image", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner('🧠 AI is analyzing the image...'):
                try:
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    results = predict_disease(model, processed_image, CLASS_NAMES)
                    
                    # Display main prediction
                    predicted_disease = results['predicted_class'].replace('_', ' ').title()
                    confidence = results['confidence']
                    
                    if confidence > 0.8:
                        st.success(f"🎯 **Predicted Disease:** {predicted_disease}")
                    elif confidence > 0.5:
                        st.warning(f"⚠️ **Likely Disease:** {predicted_disease}")
                    else:
                        st.info(f"🤔 **Possible Disease:** {predicted_disease}")
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Show top predictions
                    st.subheader("📊 Top Predictions")
                    for i, (disease, conf) in enumerate(results['top_predictions'][:5], 1):
                        disease_clean = disease.replace('_', ' ').title()
                        st.write(f"{i}. **{disease_clean}**: {conf:.1%}")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.write("Please try uploading a different image or check the model configuration.")
        
        # Additional visualizations
        if len(results['top_predictions']) > 1:
            st.subheader("📈 Confidence Distribution")
            
            try:
                # Create confidence chart
                chart_data = pd.DataFrame({
                    'Disease': [pred[0].replace('_', ' ')[:30] for pred in results['top_predictions'][:5]],
                    'Confidence': [pred[1] for pred in results['top_predictions'][:5]]
                })
                
                st.bar_chart(chart_data.set_index('Disease')['Confidence'])
                
            except Exception as e:
                st.write("Chart visualization not available")
        
        # Disease information (you can expand this with actual disease info)
        if results['confidence'] > 0.3:
            with st.expander("ℹ️ Disease Information"):
                disease_name = results['predicted_class']
                st.write(f"**Detected:** {disease_name.replace('_', ' ')}")
                
                # You can add specific disease information here
                if 'healthy' in disease_name.lower():
                    st.success("✅ This plant appears to be healthy!")
                else:
                    st.warning("⚠️ Disease detected. Consider consulting with an agricultural expert.")
    
    else:
        st.info("👆 Please upload a plant leaf image to get started")
        
        # Show example of what to upload
        st.subheader("💡 Tips for Best Results")
        st.markdown("""
        - Upload clear, well-lit images of plant leaves
        - Ensure the leaf fills most of the image frame
        - Avoid blurry or very dark images
        - Images should be in JPG, JPEG, or PNG format
        """)

if __name__ == "__main__":
    main()