import os
import json
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
from datetime import datetime
import time
import cv2
from PIL import Image
import io

from utils.model_loader import load_model
from utils.file_utils import process_media
from utils.image_utils import predict_and_save_image
from utils.video_utils import predict_and_plot_video

app = Flask(__name__)
app.secret_key = 'license_plate_detection_secret_key_2026'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['OUTPUT_FOLDER'] = 'output'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global model variable
model = None

# Initialize session stats
def init_session_stats():
    if 'stats' not in session:
        session['stats'] = {
            'total_files': 0,
            'total_plates': 0,
            'success_rate': 0.0,
            'avg_confidence': 0.0,
            'total_processing_time': 0.0,
            'successful_files': 0
        }
    if 'history' not in session:
        session['history'] = []
    if 'current_detection' not in session:
        session['current_detection'] = None

@app.before_request
def before_request():
    init_session_stats()

@app.route('/')
def index():
    global model
    init_session_stats()
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded, stats=session.get('stats', {}))

@app.route('/load-model', methods=['POST'])
def load_model_route():
    global model
    try:
        if model is None:
            model = load_model()
        return jsonify({'success': True, 'message': 'Model loaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    init_session_stats()
    global model
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 400
        
        file = request.files['file']
        conf_threshold = float(request.form.get('conf_threshold', 0.5))
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        try:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # Create output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"detected_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Start timer
            start_time = time.time()
            
            # Process media
            result_path, detection_info = process_media(input_path, output_path, conf_threshold)
            processing_time = time.time() - start_time
            
            if result_path and os.path.exists(result_path):
                # Prepare detection stats
                detection_stats = {
                    'num_plates': detection_info['plate_count'] if 'plate_count' in detection_info else detection_info.get('total_plates', 0),
                    'confidences': detection_info.get('confidences', []),
                    'avg_confidence': detection_info.get('avg_confidence', 0),
                    'processing_time': processing_time,
                    'input_file': filename,
                    'output_file': result_path,
                    'timestamp': timestamp,
                    'file_type': file.content_type
                }
                
                # Update session
                session['current_detection'] = detection_stats
                
                # Update stats
                stats = session.get('stats', {})
                stats['total_files'] = stats.get('total_files', 0) + 1
                stats['total_plates'] = stats.get('total_plates', 0) + detection_stats['num_plates']
                if detection_stats['num_plates'] > 0:
                    stats['successful_files'] = stats.get('successful_files', 0) + 1
                stats['success_rate'] = (stats['successful_files'] / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
                
                if stats['total_files'] == 1:
                    stats['avg_confidence'] = detection_stats['avg_confidence']
                else:
                    old_avg = stats['avg_confidence']
                    new_avg = (old_avg * (stats['total_files'] - 1) + detection_stats['avg_confidence']) / stats['total_files']
                    stats['avg_confidence'] = new_avg
                
                stats['total_processing_time'] = stats.get('total_processing_time', 0) + processing_time
                session['stats'] = stats
                
                # Add to history
                history = session.get('history', [])
                history_entry = {
                    'timestamp': timestamp,
                    'filename': filename,
                    'num_plates': detection_stats['num_plates'],
                    'avg_confidence': detection_stats['avg_confidence'],
                    'processing_time': processing_time,
                    'file_type': 'image' if file.content_type.startswith('image') else 'video'
                }
                history.append(history_entry)
                session['history'] = history
                session.modified = True
                
                return jsonify({
                    'success': True,
                    'detection': detection_stats,
                    'message': f'Detection complete! Processing time: {processing_time:.2f}s'
                })
            else:
                return jsonify({'success': False, 'error': 'Processing failed'}), 500
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template('upload.html', model_loaded=model is not None)

@app.route('/results')
def results():
    init_session_stats()
    detection = session.get('current_detection')
    return render_template('results.html', detection=detection)

@app.route('/analytics')
def analytics():
    init_session_stats()
    history = session.get('history', [])
    stats = session.get('stats', {})
    return render_template('analytics.html', history=history, stats=stats)

@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/clear-history', methods=['POST'])
def clear_history():
    session['history'] = []
    session['stats'] = {
        'total_files': 0,
        'total_plates': 0,
        'success_rate': 0.0,
        'avg_confidence': 0.0,
        'total_processing_time': 0.0,
        'successful_files': 0
    }
    session['current_detection'] = None
    session.modified = True
    return jsonify({'success': True, 'message': 'History cleared'})

@app.route('/get-model-info')
def get_model_info():
    global model
    if model is None:
        return jsonify({'loaded': False})
    
    try:
        model_info = {
            'loaded': True,
            'framework': 'YOLOv8',
            'input_size': 640,
            'device': 'cpu'
        }
        if hasattr(model, 'names'):
            model_info['classes'] = list(model.names.values())
            model_info['num_classes'] = len(model.names)
        return jsonify(model_info)
    except:
        return jsonify({'loaded': True, 'framework': 'YOLOv8'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)