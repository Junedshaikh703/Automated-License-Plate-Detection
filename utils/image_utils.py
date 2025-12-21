import cv2
import streamlit as st
from utils.model_loader import load_model
model = load_model()

  

def predict_and_save_image(path_test_car, output_image_path , conf_thresh):
    """
    Predicts and saves the bounding boxes on the given test image using the trained YOLO model.
    
    Returns:
    tuple: (output_image_path, detection_info) where detection_info contains plate count and confidences
    """
    try:
        results = model.predict(path_test_car, conf=float(conf_thresh), device='cpu')
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plate_count = 0
        confidences = []
        boxes = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                if confidence >= conf_thresh:
                    plate_count += 1
                    confidences.append(confidence)
                    boxes.append([x1, y1, x2, y2])
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        detection_info = {
            'plate_count': plate_count,
            'confidences': confidences,
            'avg_confidence': avg_confidence,
            'boxes': boxes
        }
        
        return output_image_path, detection_info
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None