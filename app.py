# app.py - COMPLETE INTEGRATED VERSION
import streamlit as st
import os
import tempfile
import time
from datetime import datetime
from PIL import Image
import cv2
import pandas as pd

# ====================== IMPORT YOUR FUNCTIONS ======================
# Import all your utility functions
from utils.model_loader import load_model
from utils.file_utils import process_media
# We'll use these directly when needed
from utils.image_utils import predict_and_save_image
from utils.video_utils import predict_and_plot_video

# ====================== PAGE SETUP ======================
st.set_page_config(
    page_title="License Plate Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== GLOBAL STATE ======================
# Initialize all session states

if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_files': 0,
        'total_plates': 0,
        'success_rate': 0.0,
        'avg_confidence': 0.0,
        'total_processing_time': 0.0,
        'successful_files': 0
    }
if 'current_detection' not in st.session_state:
    st.session_state.current_detection = None

# ====================== HELPER FUNCTIONS ======================
def analyze_image_results(image_path):
    """
    Analyze the processed image to count plates and get confidence scores
    This reads the processed image and extracts info
    """
    try:
        # In your current code, boxes are drawn but not saved separately
        # For now, we'll return placeholder values
        # You can modify your image_utils.py to return detection info
        return {
            'num_plates': 2,  # This should come from your detection results
            'confidences': [0.95, 0.88],  # Placeholder
            'avg_confidence': 0.915,
            'boxes': []
        }
    except Exception as e:
        st.warning(f"Could not analyze image results: {e}")
        return {'num_plates': 0, 'confidences': [], 'avg_confidence': 0.0, 'boxes': []}

def update_statistics(detection_stats):
    """Update global statistics with new detection"""
    if detection_stats and detection_stats.get('num_plates', 0) > 0:
        # Update file count
        st.session_state.stats['total_files'] += 1
        
        # Update plate count
        st.session_state.stats['total_plates'] += detection_stats['num_plates']
        
        # Update success rate
        st.session_state.stats['successful_files'] += 1
        st.session_state.stats['success_rate'] = (
            st.session_state.stats['successful_files'] / 
            st.session_state.stats['total_files']
        ) * 100
        
        # Update average confidence (simple average)
        if detection_stats.get('avg_confidence'):
            old_avg = st.session_state.stats['avg_confidence']
            old_total = st.session_state.stats['total_files'] - 1
            new_avg = (old_avg * old_total + detection_stats['avg_confidence']) / st.session_state.stats['total_files']
            st.session_state.stats['avg_confidence'] = new_avg

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("🚗 LPR System")
    st.markdown("---")
    
    # Load Model Button
    if st.button("🔄 Load/Reload Model", use_container_width=True, type="primary"):
        with st.spinner("Loading YOLO model..."):
            try:
                # Load your model using your function
                st.session_state.model = load_model()
                if st.session_state.model:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    
                    # Store model info for display
                    try:
                        # Try to get model info
                        model_info = {
                            'framework': 'YOLOv8',
                            'input_size': 640,  # YOLO default
                            'device': 'cpu',  # Your code uses device='cpu'
                        }
                        
                        # Try to get class names
                        if hasattr(st.session_state.model, 'names'):
                            model_info['classes'] = list(st.session_state.model.names.values())
                            model_info['num_classes'] = len(st.session_state.model.names)
                        
                        st.session_state.model_info = model_info
                        
                    except:
                        st.session_state.model_info = {'framework': 'YOLOv8'}
                        
                else:
                    st.error("❌ Failed to load model")
                    
            except Exception as e:
                st.error(f"Error loading model: {e}")
    
    st.markdown("---")

    CONF_THRESH = st.select_slider(
        "Confidence Threshold",
        options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    st.session_state.conf_thresh = CONF_THRESH
    
    # Model Information
    st.markdown("### 🤖 Model Information")
    
    if st.session_state.model_loaded:
        if hasattr(st.session_state, 'model_info'):
            info = st.session_state.model_info
            
            st.markdown(f"**Framework:** {info.get('framework', 'YOLOv8')}")
            st.markdown(f"**Input Size:** {info.get('input_size', 640)}px")
            st.markdown(f"**Device:** {info.get('device', 'CPU').upper()}")
            current_thresh = st.session_state.conf_thresh
            st.markdown(f"**Confidence Threshold:** {current_thresh}")


            if 'num_classes' in info:
                st.markdown(f"**Classes:** {info['num_classes']}")
                if 'classes' in info:
                    with st.expander("View Class Names"):
                        for cls in info['classes']:
                            st.markdown(f"- {cls}")
        else:
            st.info("YOLOv8 Model Loaded")
    else:
        st.info("Model not loaded yet")
    
    st.markdown("---")
    
    # Real-time Statistics
    st.markdown("### 📊 Live Statistics")
    
    stats = st.session_state.stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Files Processed", stats['total_files'])
        st.metric("Total Plates", stats['total_plates'])
    
    with col2:
        st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
        if stats['total_files'] > 0:
            st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0)*100:.1f}%")
    
    if stats['total_files'] > 0:
        avg_time = stats['total_processing_time'] / stats['total_files']
        st.markdown(f"**Avg Processing Time:** {avg_time:.2f}s")
    
    st.markdown("---")
    
    # File Support
    st.markdown("### 📁 Supported Formats")
    st.markdown("**Images:** JPG, PNG, BMP")
    st.markdown("**Videos:** MP4, AVI, MOV, MKV")
    st.markdown("**Max Size:** 200MB")

# ====================== MAIN CONTENT ======================
st.title("License Plate Detection System")
st.markdown("### Automatic license plate recognition using YOLOv8")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload & Detect", "📊 Results", "📈 Analytics"])

# TAB 1: Upload and Detection
with tab1:
    st.header("Upload Media for Detection")
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model from the sidebar first.")
        st.info("Click the 'Load/Reload Model' button in the sidebar to load your YOLO model.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
        help="Maximum file size: 200MB"
    )
    
    if uploaded_file and st.session_state.model_loaded:
        # Display file info in columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("File Preview")
            
            # Preview based on file type
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)
                
                # Get image dimensions
                width, height = image.size
                st.caption(f"Dimensions: {width} × {height} pixels")
                
            elif uploaded_file.type.startswith('video'):
                st.video(uploaded_file)
                st.info("Video file uploaded. Processing may take longer than images.")
        
        with col2:
            st.subheader("File Details")
            st.metric("File Name", uploaded_file.name)
            st.metric("File Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
            st.metric("File Type", uploaded_file.type.split('/')[-1].upper())
            
            # Show file type icon
            if uploaded_file.type.startswith('image'):
                st.markdown("**Type:** 📷 Image")
            else:
                st.markdown("**Type:** 🎥 Video")
        
        st.markdown("---")
        
        # Detection Controls
        st.subheader("Detection Settings")
        
    
        col1, col2 = st.columns(2)
        
        with col1:
            # Save option
            save_results = st.checkbox("Save Results to Output Folder", value=True)
            
        with col2:
            # Device option (your code uses CPU)
            device = st.selectbox(
                "Processing Device",
                ["CPU", "GPU (if available)"],
                index=0,
                disabled=True,  # Your code is set to CPU
                help="Your code is configured to use CPU only"
            )
        
        # Process button
        st.markdown("---")
        if st.button("🚀 Start Detection", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    # Create temporary directory
                    os.makedirs("temp", exist_ok=True)
                    
                
                    input_path = os.path.join("temp", uploaded_file.name)
                

                    # Save uploaded file to temp location
                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Create output path
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"detected_{uploaded_file.name}"
                    
                    # Create output directory if it doesn't exist
                    os.makedirs("output", exist_ok=True)
                    output_path = os.path.join("output", output_filename)
                    
                    # Start timer
                    start_time = time.time()
                    
                    # ========== USE YOUR process_media FUNCTION ==========
                    # This automatically chooses image or video processing
                    result_path, detection_info = process_media(input_path, output_path , st.session_state.conf_thresh)
                    # =====================================================
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    if result_path and os.path.exists(result_path):
                        # Analyze results (you need to modify your functions to return detection info)
                        # For now, we'll use placeholders
                        detection_stats = {
                            'num_plates': detection_info['plate_count'],  
                            'confidences': detection_info['confidences'],  
                            'avg_confidence': detection_info['avg_confidence'],  
                            'processing_time': processing_time,
                            'input_file': uploaded_file.name,
                            'output_file': result_path,
                            'timestamp': timestamp,
                            'file_type': uploaded_file.type
                        }
                        
                        # Store current detection
                        st.session_state.current_detection = detection_stats
                        
                        # Update global statistics
                        update_statistics(detection_stats)
                        
                        # Add to history
                        history_entry = {
                            'timestamp': timestamp,
                            'filename': uploaded_file.name,
                            'num_plates': detection_stats['num_plates'],
                            'avg_confidence': detection_stats['avg_confidence'],
                            'processing_time': processing_time,
                            'file_type': uploaded_file.type.split('/')[0]  # 'image' or 'video'
                        }
                        st.session_state.detection_history.append(history_entry)
                        
                        # Show success
                        st.success(f"✅ Detection complete! Processing time: {processing_time:.2f}s")
                        st.balloons()
                        
                        # Auto-switch to Results tab
                        st.markdown("**👉 Switch to 'Results' tab to see the output**")
                        
                    else:
                        st.error("❌ Processing failed. No output file created.")
                        
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
    
    elif uploaded_file and not st.session_state.model_loaded:
        st.error("⚠️ Model not loaded. Please load the model from sidebar first.")

# TAB 2: Results
with tab2:
    st.header("Detection Results")
    
    if st.session_state.current_detection:
        detection = st.session_state.current_detection
        
        # Display results in columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 Processed Output")
            
            if detection.get('output_file') and os.path.exists(detection['output_file']):
                output_path = detection['output_file']
                
                # Check if it's video or image
                if output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    st.video(output_path)
                    st.caption("Processed video with license plate detections")
                else:
                    st.image(output_path, caption="Detected License Plates", use_container_width=True)
                    st.caption("Green boxes show detected license plates with confidence scores")
            else:
                st.warning("Output file not found. It may have been deleted or moved.")
        
        with col2:
            st.subheader("📊 Detection Statistics")
            
            # Show metrics
            st.metric("Plates Detected", detection['num_plates'])
            st.metric("Average Confidence", f"{detection['avg_confidence']*100:.1f}%")
            st.metric("Processing Time", f"{detection['processing_time']:.2f}s")
            st.metric("File Type", detection.get('file_type', 'Unknown').split('/')[-1].upper())
            
            # Confidence scores (if available)
            if detection.get('confidences'):
                with st.expander("View Confidence Scores"):
                    for i, conf in enumerate(detection['confidences']):
                        st.progress(conf, text=f"Plate {i+1}: {conf:.1%}")
            
            # Download button
            if detection.get('output_file') and os.path.exists(detection['output_file']):
                with open(detection['output_file'], "rb") as f:
                    file_bytes = f.read()
                
                # Determine MIME type
                if detection['output_file'].lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    mime_type = "video/mp4"
                else:
                    mime_type = "image/jpeg"
                
                st.download_button(
                    label="📥 Download Result",
                    data=file_bytes,
                    file_name=os.path.basename(detection['output_file']),
                    mime=mime_type,
                    use_container_width=True,
                    type="primary"
                )
            
            # New detection button
            if st.button("🔄 New Detection", use_container_width=True):
                st.session_state.current_detection = None
                st.rerun()
        
        # Show file info
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.write(f"**Input File:** {detection['input_file']}")
            st.write(f"**Output File:** {os.path.basename(detection['output_file'])}")
        with col4:
            st.write(f"**Processed At:** {detection['timestamp']}")
            st.write(f"**Saved In:** {os.path.dirname(detection['output_file'])}")
    
    else:
        st.info("👈 No results yet. Upload a file and run detection in Tab 1.")
        st.markdown("""
        **What to do:**
        1. Go to "Upload & Detect" tab
        2. Load model from sidebar (if not loaded)
        3. Upload an image or video
        4. Click "Start Detection"
        5. Come back here to see results
        """)

# TAB 3: Analytics
with tab3:
    st.header("Analytics Dashboard")
    
    if st.session_state.detection_history:
        history = st.session_state.detection_history
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_plates = sum(h['num_plates'] for h in history)
            st.metric("Total Plates", total_plates)
        
        with col2:
            total_files = len(history)
            st.metric("Files Processed", total_files)
        
        with col3:
            avg_plates = total_plates / total_files if total_files > 0 else 0
            st.metric("Avg Plates/File", f"{avg_plates:.1f}")
        
        with col4:
            successful = sum(1 for h in history if h['num_plates'] > 0)
            success_rate = (successful / total_files) * 100 if total_files > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Detection History Table
        st.subheader("📋 Detection History")
        
        # Create DataFrame
        df_history = pd.DataFrame(history)
        
        # Format for display
        display_df = df_history.copy()
        display_df['avg_confidence'] = display_df['avg_confidence'].apply(lambda x: f"{x*100:.1f}%")
        display_df['processing_time'] = display_df['processing_time'].apply(lambda x: f"{x:.2f}s")
        display_df = display_df[['filename', 'file_type', 'num_plates', 'avg_confidence', 'processing_time', 'timestamp']]
        
        # Rename columns for better display
        display_df.columns = ['Filename', 'Type', 'Plates', 'Confidence', 'Time', 'Timestamp']
        
        st.dataframe(display_df, use_container_width=True, height=300)
        
        # File type distribution
        st.subheader("📊 File Type Distribution")
        
        if 'file_type' in df_history.columns:
            file_counts = df_history['file_type'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**By Count:**")
                for file_type, count in file_counts.items():
                    st.write(f"{file_type.capitalize()}: {count}")
            
            with col2:
                st.write("**By Plates Found:**")
                image_plates = df_history[df_history['file_type'] == 'image']['num_plates'].sum()
                video_plates = df_history[df_history['file_type'] == 'video']['num_plates'].sum()
                st.write(f"Images: {image_plates} plates")
                st.write(f"Videos: {video_plates} plates")
        
        # Clear history button
        st.markdown("---")
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.detection_history = []
            st.session_state.stats = {
                'total_files': 0,
                'total_plates': 0,
                'success_rate': 0.0,
                'avg_confidence': 0.0,
                'total_processing_time': 0.0,
                'successful_files': 0
            }
            st.success("History cleared!")
            st.rerun()
    
    else:
        st.info("No detection history yet. Process some files to see analytics here.")
        st.markdown("""
        **Analytics will show:**
        - Total files processed
        - Total plates detected
        - Success rate
        - File type distribution
        - Processing time statistics
        """)

# ====================== FOOTER ======================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p><strong>🚗 License Plate Detection System</strong> | Powered by YOLOv8</p>
    <p><small>Built with Streamlit | Last updated: {}</small></p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")),
    unsafe_allow_html=True
)