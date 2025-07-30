import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import traceback

# Set up Streamlit page
st.set_page_config(page_title="Plant Disease Detector 🌿", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.write("Upload a plant leaf image to detect its disease.")

# Debug information
st.sidebar.header("🔧 Debug Info")
st.sidebar.write(f"Python version: {st.__version__}")
st.sidebar.write(f"PyTorch version: {torch.__version__}")
st.sidebar.write(f"CUDA available: {torch.cuda.is_available()}")

# PlantVillage 38-class labels
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.device = None
    st.session_state.error_message = None

# Define the CNN model architecture (must match training code exactly)
class CNN(torch.nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.conv_layers(dummy)
            self.flatten_size = out.view(1, -1).shape[1]

        self.dense_layers = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(self.flatten_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(1024, K)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return x

def load_model_safe():
    """Safely load the PyTorch model with comprehensive error handling"""
    try:
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.sidebar.write(f"Using device: {device}")
        
        # Check if model file exists - try both PM.pt and PM2.pt
        model_paths = ["PM2.pt", "PM.pt"]  # Try PM2.pt first (from your training code)
        model_path = None
        
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            current_dir = os.getcwd()
            files_in_dir = os.listdir(current_dir)
            error_msg = f"""
            ❌ Model file not found! Looked for: {model_paths}
            
            Current directory: {current_dir}
            Files in directory: {files_in_dir}
            
            Please ensure:
            1. The model file is in the same folder as this script
            2. You have read permissions for the file
            3. The model was saved correctly from training
            """
            return None, None, error_msg
        
        # Check file size and permissions
        file_size = os.path.getsize(model_path)
        st.sidebar.write(f"Model file: {model_path}")
        st.sidebar.write(f"Model file size: {file_size / (1024*1024):.1f} MB")
        
        if file_size == 0:
            return None, None, "❌ Model file is empty (0 bytes)"
        
        # Attempt to load model
        st.info(f"🔄 Loading model from {model_path}... This may take a moment.")
        
        # Load the state dict
        try:
            state_dict = torch.load(model_path, map_location=device)
            st.sidebar.write(f"Loaded object type: {type(state_dict)}")
            
            # Check if it's a state_dict (OrderedDict) or full model
            if isinstance(state_dict, dict) and not hasattr(state_dict, 'eval'):
                # It's a state_dict, need to create model architecture first
                st.info("📦 Detected state_dict format, creating model architecture...")
                
                # Create model with correct number of classes
                num_classes = len(class_names)
                model = CNN(num_classes)
                
                # Load the state dict into the model
                model.load_state_dict(state_dict)
                st.success("✅ State dict loaded successfully!")
                
            else:
                # It's a full model
                model = state_dict
                st.success("✅ Full model loaded successfully!")
        
        except Exception as e:
            return None, None, f"❌ Error loading model file:\n{str(e)}\n\nThis usually means the file is corrupted or incompatible."
        
        # Verify model structure
        if model is None:
            return None, None, "❌ Model loaded but is None"
        
        # Set model to evaluation mode
        model.eval()
        
        # Move model to device
        model = model.to(device)
        
        # Test model with dummy input
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                output = model(dummy_input)
                
                # Check output shape
                if output.shape[1] != len(class_names):
                    return None, None, f"❌ Model output classes ({output.shape[1]}) don't match expected classes ({len(class_names)})"
                
                st.sidebar.write(f"✅ Model test passed - Output shape: {output.shape}")
                
        except Exception as e:
            return None, None, f"❌ Model test failed: {str(e)}"
        
        return model, device, None
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        return None, None, f"❌ Unexpected error loading model:\n{str(e)}\n\nFull traceback:\n{error_traceback}"

# Load model if not already loaded
if not st.session_state.model_loaded:
    with st.spinner("Loading model..."):
        model, device, error = load_model_safe()
        
        if error:
            st.session_state.error_message = error
        else:
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.session_state.error_message = None

# Display error if model loading failed
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    
    # Provide troubleshooting steps
    st.markdown("### 🛠️ Troubleshooting Steps:")
    st.markdown("""
    1. **Check model file location**: Ensure `PM.pt` is in the same directory as this script
    2. **Verify file integrity**: Re-download or re-save your model file
    3. **Check PyTorch version**: Your model might be saved with a different PyTorch version
    4. **Try loading in Python directly**:
       ```python
       import torch
       model = torch.load('PM.pt', map_location='cpu')
       print(type(model))
       print(model)
       ```
    5. **Check model architecture**: Ensure the model has 38 output classes
    """)
    
    if st.button("🔄 Retry Loading Model"):
        st.session_state.model_loaded = False
        st.rerun()

# Image preprocessing function
def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        # Define transforms (removed ColorJitter for inference)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms and add batch dimension
        tensor = transform(image).unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

# Main application logic
if st.session_state.model_loaded and st.session_state.model is not None:
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            
            # Display image info
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            st.write(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Image mode:** {image.mode}")
            
            # Preprocess image
            with st.spinner("Preprocessing image..."):
                input_tensor = preprocess_image(image)
            
            if input_tensor is not None:
                # Make prediction
                with st.spinner("Making prediction..."):
                    try:
                        input_tensor = input_tensor.to(st.session_state.device)
                        
                        with torch.no_grad():
                            outputs = st.session_state.model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            
                            predicted_class = class_names[predicted.item()]
                            confidence_score = confidence.item()
                        
                        # Display results
                        st.success("🎯 Prediction Complete!")
                        
                        # Format prediction
                        formatted_prediction = predicted_class.replace('___', ' → ').replace('_', ' ')
                        
                        st.markdown(f"### **Prediction:** {formatted_prediction}")
                        st.markdown(f"### **Confidence:** {confidence_score:.1%}")
                        
                        # Confidence assessment
                        if confidence_score >= 0.8:
                            st.success("🟢 High confidence prediction")
                        elif confidence_score >= 0.6:
                            st.warning("🟡 Medium confidence prediction")
                        else:
                            st.error("🔴 Low confidence prediction - consider retaking the photo")
                        
                        # Show top 3 predictions
                        st.markdown("### Top 3 Predictions:")
                        top3_prob, top3_idx = torch.topk(probabilities, 3)
                        
                        for i in range(3):
                            class_name = class_names[top3_idx[0][i].item()]
                            prob = top3_prob[0][i].item()
                            formatted_name = class_name.replace('___', ' → ').replace('_', ' ')
                            st.write(f"{i+1}. {formatted_name}: {prob:.1%}")
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        st.write("**Error details:**")
                        st.code(traceback.format_exc())
            
        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {str(e)}")
            st.write("Please try uploading a different image file.")
    
    else:
        st.info("👆 Upload an image to start disease detection!")

else:
    st.warning("⏳ Model not loaded. Please check the error messages above.")

# Additional information
st.markdown("---")
with st.expander("📖 About This Classifier"):
    st.markdown(f"""
    **Classes Supported:** {len(class_names)} plant disease categories
    
    **Plants Included:**
    - Fruits: Apple, Blueberry, Cherry, Grape, Orange, Peach, Raspberry, Strawberry  
    - Vegetables: Corn, Bell Pepper, Potato, Tomato
    - Other: Soybean, Squash
    
    **Model Requirements:**
    - Input: 224x224 RGB images
    - Output: {len(class_names)} classes
    - Preprocessing: ImageNet normalization
    """)

with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    - Use well-lit, clear images of leaves
    - Fill the frame with the leaf
    - Avoid blurry or heavily shadowed photos  
    - Best results with leaves showing clear symptoms
    - Multiple angles can help confirm diagnosis
    """)