import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

# Configure page
st.set_page_config(
    page_title="Plant Health Scanner",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for farmer-friendly design with dark mode support
st.markdown("""
<style>
    /* Auto dark mode detection */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        
        .main-header {
            background: linear-gradient(90deg, #2d5a2d, #4a7c59);
        }
        
        .result-card {
            background-color: #262730;
            color: #fafafa;
        }
        
        .healthy-card {
            background-color: #1a2e1a;
            border-left-color: #4CAF50;
        }
        
        .diseased-card {
            background-color: #2d2416;
            border-left-color: #FF9800;
        }
        
        .tips-section {
            background-color: #262730;
            color: #fafafa;
        }
        
        .stSelectbox > div > div {
            background-color: #262730;
        }
        
        .stTextInput > div > div > input {
            background-color: #262730;
            color: #fafafa;
        }
    }
    
    /* Light mode (default) */
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #E8F5E8;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .healthy-card {
        background-color: #E8F5E8;
        border-left-color: #4CAF50;
    }
    .diseased-card {
        background-color: #FFF3E0;
        border-left-color: #FF9800;
    }
    .confidence-bar {
        background-color: #ddd;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
        font-size: 12px;
    }
    .tips-section {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    
    /* Dark mode confidence bar */
    @media (prefers-color-scheme: dark) {
        .confidence-bar {
            background-color: #404040;
        }
    }
    
    /* Smooth transitions for mode changes */
    .stApp, .result-card, .tips-section {
        transition: background-color 0.3s ease, color 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Your CNN Model Architecture
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        # Flatten
        out = out.view(-1, 50176)
        # Fully connected
        out = self.dense_layers(out)
        return out

@st.cache_resource
def load_model(model_path="PM.pt"):
    """Load the trained model with state dict"""
    try:
        # Create model architecture first (K=76 due to duplication in original training code)
        model = CNN(K=76)
        
        # Load the state dictionary (weights)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please check the file path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure your model architecture matches the saved weights")
        st.stop()

def preprocess_image(image):
    """Preprocess image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_disease(model, image_tensor):
    """Make prediction on the image and handle duplicated classes"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        original_predicted = predicted.item()
        
        # Since we don't know the exact mapping, let's try different approaches
        # Option 1: Direct mapping (if classes 0-38 are correct)
        if original_predicted < 39:
            final_class = original_predicted
        else:
            # Option 2: If classes 39-76 map to 0-37
            final_class = original_predicted - 39
            
        # Debug info (you can remove this later)
        print(f"Original prediction: {original_predicted}, Mapped to: {final_class}")
        
    return final_class, confidence.item(), original_predicted  # Return original too for debugging

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Plant Health Scanner</h1>
        <p>AI-powered tool to help farmers identify plant diseases quickly and accurately</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add model configuration section
    with st.sidebar:
        st.markdown("### Model Configuration")
        model_path = st.text_input(
            "Model Path", 
            value="PM.pt",
            help="Path to your trained PyTorch model file (.pt or .pth)"
        )
        
        # Store model path in session state
        st.session_state.model_path = model_path
        
        # Model info
        if st.button("Reload Model"):
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("### How to Use")
        st.markdown("""
        1. **Take a clear photo** of your plant leaf
        2. **Upload the image** using the button below
        3. **Get instant results** about plant health
        4. **Follow recommendations** for treatment
        """)
        
        st.markdown("### Photo Tips")
        st.markdown("""
        - Use good lighting (natural light preferred)
        - Focus on a single leaf
        - Keep the leaf flat and visible
        - Avoid blurry or dark images
        """)
        
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
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Plant Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add analyze button
            if st.button("Analyze Plant Health", type="primary", use_container_width=True):
                with st.spinner("Analyzing plant health..."):
                    # Get model path from session state
                    current_model_path = st.session_state.get('model_path', 'plant_disease_model.pt')
                    
                    # Load model and make prediction
                    model = load_model(current_model_path)
                    image_tensor = preprocess_image(image)
                    
                    # Make actual prediction with your trained model
                    predicted_class, confidence, original_pred = predict_disease(model, image_tensor)
                    
                    # Store results in session state for display in col2
                    st.session_state.prediction_result = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'original_prediction': original_pred,  # For debugging
                        'image': image
                    }
    
    with col2:
        st.markdown("### Analysis Results")
        
        if hasattr(st.session_state, 'prediction_result'):
            result = st.session_state.prediction_result
            predicted_class = result['class']
            confidence = result['confidence']
            original_pred = result.get('original_prediction', 'N/A')
            
            # Debug info - remove this later
            st.write(f"Debug: Original model output: {original_pred}, Mapped to class: {predicted_class}")
            
            # PyTorch ImageFolder sorts classes alphabetically by folder name
            # This should match the order your model was trained on
            class_names = [
                "Apple_black_rot",           # 0
                "Apple_cedar_apple_rust",    # 1  
                "Apple_healthy",             # 2
                "Apple_scab",                # 3
                "Background_without_leaves", # 4
                "Blueberry_healthy",         # 5
                "Cherry_healthy",            # 6
                "Cherry_powdery_mildew",     # 7
                "Corn_common_rust",          # 8
                "Corn_gray_leaf_spot",       # 9
                "Corn_healthy",              # 10
                "Corn_northern_leaf_blight", # 11
                "Grape_black_measles",       # 12
                "Grape_black_rot",           # 13
                "Grape_healthy",             # 14
                "Grape_leaf_blight",         # 15
                "Orange_haunglongbing",      # 16
                "Peach_bacterial_spot",      # 17
                "Peach_healthy",             # 18
                "Pepper_bacterial_spot",     # 19
                "Pepper_healthy",            # 20
                "Potato_early_blight",       # 21
                "Potato_healthy",            # 22
                "Potato_late_blight",        # 23
                "Raspberry_healthy",         # 24
                "Soybean_healthy",           # 25
                "Squash_powdery_mildew",     # 26
                "Strawberry_healthy",        # 27
                "Strawberry_leaf_scorch",    # 28
                "Tomato_bacterial_spot",     # 29
                "Tomato_early_blight",       # 30
                "Tomato_healthy",            # 31
                "Tomato_late_blight",        # 32
                "Tomato_leaf_mold",          # 33
                "Tomato_mosaic_virus",       # 34
                "Tomato_septoria_leaf_spot", # 35
                "Tomato_spider_mites_two-spotted_spider_mite", # 36
                "Tomato_target_spot",        # 37
                "Tomato_yellow_leaf_curl_virus" # 38
            ]
            
            # Safety check
            if predicted_class >= len(class_names):
                st.error(f"Class index {predicted_class} is out of range. Max index should be {len(class_names)-1}")
                predicted_label = f"Unknown_Class_{predicted_class}"
            else:
                predicted_label = class_names[predicted_class]
            
            # Determine if it's healthy or diseased based on class name
            is_healthy = "healthy" in predicted_label.lower()
            
            # Handle special cases
            if predicted_label == "Background_without_leaves":
                st.markdown(f"""
                <div class="result-card" style="background-color: #FFF8E1; border-left-color: #FFC107;">
                    <h3>Detection Result: NO PLANT DETECTED</h3>
                    <p><strong>Detected:</strong> {predicted_label}</p>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%; background-color: #FFC107;">
                            {confidence:.1%}
                        </div>
                    </div>
                    <p><strong>Recommendation:</strong> Please upload an image containing plant leaves for analysis.</p>
                </div>
                """, unsafe_allow_html=True)
                
            elif is_healthy:
                st.markdown(f"""
                <div class="result-card healthy-card">
                    <h3>Plant Status: HEALTHY</h3>
                    <p><strong>Detected Class:</strong> {predicted_label}</p>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%; background-color: #4CAF50;">
                            {confidence:.1%}
                        </div>
                    </div>
                    <p><strong>Recommendation:</strong> Your plant looks healthy! Continue with regular care and monitoring.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Healthy plant care tips
                st.markdown("#### Maintenance Tips")
                st.success("Continue regular watering schedule")
                st.success("Ensure adequate sunlight")
                st.success("Monitor for any changes")
                st.success("Apply fertilizer as needed")
                
            else:  # Diseased
                st.markdown(f"""
                <div class="result-card diseased-card">
                    <h3>Plant Status: DISEASE DETECTED</h3>
                    <p><strong>Disease Type:</strong> {predicted_label}</p>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%; background-color: #FF9800;">
                            {confidence:.1%}
                        </div>
                    </div>
                    <p><strong>Recommendation:</strong> Specific disease detected. Take appropriate action based on disease type.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Disease treatment recommendations
                st.markdown("#### Immediate Actions")
                st.warning("Isolate affected plants")
                st.warning("Remove diseased leaves/parts")
                st.warning("Adjust watering practices")
                st.warning("Consider appropriate treatment for this specific disease")
                
                # Contact information
                st.markdown("#### Need Help?")
                st.info("Contact your local agricultural extension office for specific treatment advice.")
        
        else:
            st.info("Upload an image to get started with plant health analysis")
    
    # Additional information section
    st.markdown("""
    <div class="tips-section">
        <h3>About This Tool</h3>
        <p>This AI-powered plant disease classifier helps farmers:</p>
        <ul>
            <li><strong>Quick Detection:</strong> Get instant results on plant health</li>
            <li><strong>Early Prevention:</strong> Catch diseases before they spread</li>
            <li><strong>Cost Saving:</strong> Reduce crop losses through early intervention</li>
            <li><strong>Easy to Use:</strong> No technical expertise required</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Important Notes")
    st.info("This tool provides preliminary analysis - consult experts for severe cases")
    st.info("Results accuracy depends on image quality and lighting") 
    st.info("For best results, capture images during daylight hours")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with AI & Deep Learning for farmers")

if __name__ == "__main__":
    main()