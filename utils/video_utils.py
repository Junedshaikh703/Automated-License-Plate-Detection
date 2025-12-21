import cv2 
import streamlit as st  
from utils.model_loader import load_model

model = load_model()

def predict_and_plot_video(video_path, output_path):
    """
    Predicts and saves the bounding boxes on the given test video using the trained YOLO model.

    Parameters:
    video_path (str): Path to the test video file.
    output_path (str): Path to save the output video file.

    Returns:
    tuple: (output_path, detection_info) where detection_info is a dictionary containing:
        - total_plates: Total number of plates detected in the video
        - confidences: List of all confidence scores
        - avg_confidence: Average confidence score
        - frames_processed: Number of frames processed
        - fps: Frames per second of the video
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None, None
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize counters for statistics
        total_plates = 0
        all_confidences = []
        frames_processed = 0
        
        # Progress tracking (for Streamlit)
        progress_bar = None
        if total_frames > 0:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_processed += 1
            
            # Update progress
            if progress_bar and total_frames > 0:
                progress = frames_processed / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frames_processed}/{total_frames}")
            
            # Convert frame for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run prediction
            results = model.predict(rgb_frame, device='cpu', verbose=False)
            
            # Process detections in this frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])  # Convert to Python float
                    
                    # Only count if confidence meets threshold (0.4 from your image_utils)
                    if confidence >= 0.4:
                        total_plates += 1
                        all_confidences.append(confidence)
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{confidence*100:.2f}%', 
                                  (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Write processed frame
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Clear progress bar
        if progress_bar:
            progress_bar.empty()
            status_text.empty()
        
        # Calculate statistics
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # Prepare detection info dictionary
        detection_info = {
            'total_plates': total_plates,
            'confidences': all_confidences,
            'avg_confidence': avg_confidence,
            'frames_processed': frames_processed,
            'video_fps': fps,
            'video_resolution': f"{frame_width}x{frame_height}",
            'total_frames': total_frames if total_frames > 0 else frames_processed
        }
        
        return output_path, detection_info
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None