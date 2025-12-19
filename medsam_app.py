from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import json
from werkzeug.utils import secure_filename
from utils.annotation_helpers import AnnotationManager
from utils.medsam_integration import MedSAMIntegrator
from utils.supabase_client import SupabaseManager
from config import IMAGE_ID
import numpy as np
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
medsam_integrator = MedSAMIntegrator()
supabase_manager = SupabaseManager()

# Global storage for mask data (in production, use Redis or similar)
mask_storage = {}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Redirect to MedSAM interface"""
    return render_template('medsam.html')

@app.route('/medsam')
def medsam_interface():
    """MedSAM interactive segmentation interface"""
    return render_template('medsam.html')

@app.route('/medsam/load_from_supabase', methods=['POST'])
def load_from_supabase():
    """Load image and masks from Supabase database"""
    try:
        tick1 = time.perf_counter()
        print(f"Loading image and masks for image_id: {IMAGE_ID}")
        
        # Get image and masks from Supabase
        tick2 = time.perf_counter()
        data = supabase_manager.get_image_and_masks(IMAGE_ID)
        print(f"Time to get image and masks: {time.perf_counter() - tick2}")
        
        # Load image for MedSAM processing
        tick3 = time.perf_counter()
        success = medsam_integrator.load_image(data['temp_image_path'], precomputed_embedding=data.get('embedding'))
        print(f"Time to load image: {time.perf_counter() - tick3}")
        if not success:
            return jsonify({'error': 'Failed to load image for MedSAM processing'}), 500
        
        # Send only essential data initially (without large mask arrays)
        mask_metadata = []
        for i, mask_data in enumerate(data['masks']):
            try:
                mask_id = f'mask_{i}'
                # Explicitly check for mask or segmentation key
                if 'mask' in mask_data and mask_data['mask'] is not None:
                    mask_array = mask_data['mask']
                elif 'segmentation' in mask_data and mask_data['segmentation'] is not None:
                    mask_array = mask_data['segmentation']
                else:
                    print(f"Warning: No mask data found for mask {i}")
                    continue
                mask_storage[mask_id] = {
                    'mask': mask_array
                }
                mask_metadata.append({
                    'id': mask_id,
                    'bbox': mask_data['bbox'],
                    'area': mask_data['area'],
                    'predicted_iou': mask_data.get('predicted_iou'),
                    'stability_score': mask_data.get('stability_score'),
                    'point_coords': mask_data.get('point_coords', []),
                    'crop_box': mask_data.get('crop_box', []),
                    'mask_available': True  # Indicate that mask data is available
                })
            except Exception as e:
                print(f"Error processing mask {i}: {e}")
                continue
        
        response_data = {
            'success': True,
            'image_url': data['signed_url'],
            'image_data': data['image_data'],
            'mask_metadata': mask_metadata,  # Send metadata instead of full masks
            'temp_image_path': data['temp_image_path']
        }
        
        print(f"Successfully loaded {len(mask_metadata)} mask metadata from Supabase")
        print(f"Time to load image and masks: {time.perf_counter() - tick1}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error loading from Supabase: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load from Supabase: {str(e)}'}), 500

@app.route('/medsam/segment_points', methods=['POST'])
def medsam_segment_points():
    """Segment using point prompts"""
    try:
        data = request.get_json()
        points = data.get('points', [])  # [[x1, y1], [x2, y2], ...]
        labels = data.get('labels', [])  # [1, 0, 1, ...] (1=foreground, 0=background)
        
        if not points or not labels:
            return jsonify({'error': 'Points and labels are required'}), 400
        
        # Perform segmentation
        result = medsam_integrator.segment_with_points(points, labels)
        
        if result:
            # Convert mask to list for JSON serialization
            if isinstance(result['mask'], np.ndarray):
                result['mask'] = result['mask'].tolist()
            
            return jsonify({
                'success': True,
                'mask': result['mask'],
                'confidence': result['confidence']
            })
        else:
            return jsonify({'error': 'Segmentation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500

@app.route('/medsam/segment_box', methods=['POST'])
def medsam_segment_box():
    """Segment using bounding box"""
    try:
        tick1 = time.perf_counter()
        data = request.get_json()
        bbox = data.get('bbox', [])  # [x1, y1, x2, y2]
        
        if not bbox or len(bbox) != 4:
            return jsonify({'error': 'Valid bounding box is required'}), 400
        
        # Perform segmentation
        result = medsam_integrator.segment_with_box(bbox)
        print(f"Segmentation took {time.perf_counter() - tick1}")
        
        if result:
            # Convert mask to list for JSON serialization
            tick2 = time.perf_counter()
            if isinstance(result['mask'], np.ndarray):
                result['mask'] = result['mask'].tolist()
            print(f"Conversion took {time.perf_counter() - tick2}")
            
            return jsonify({
                'success': True,
                'mask': result['mask'],
                'confidence': result['confidence']
            })
        else:
            return jsonify({'error': 'Segmentation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500

@app.route('/medsam/segment_combined', methods=['POST'])
def medsam_segment_combined():
    """Segment using both points and bounding box"""
    try:
        data = request.get_json()
        points = data.get('points', [])
        labels = data.get('labels', [])
        bbox = data.get('bbox', [])
        
        if not points or not labels or not bbox:
            return jsonify({'error': 'Points, labels, and bounding box are required'}), 400
        
        # Perform segmentation
        result = medsam_integrator.segment_with_points_and_box(points, labels, bbox)
        
        if result:
            # Convert mask to list for JSON serialization
            if isinstance(result['mask'], np.ndarray):
                result['mask'] = result['mask'].tolist()
            
            return jsonify({
                'success': True,
                'mask': result['mask'],
                'confidence': result['confidence']
            })
        else:
            return jsonify({'error': 'Segmentation failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500

@app.route('/medsam/status')
def medsam_status():
    """Check MedSAM availability"""
    return jsonify({
        'available': medsam_integrator.is_available(),
        'device': str(medsam_integrator.device) if medsam_integrator.device else None
    })

@app.route('/medsam/get_mask/<mask_id>', methods=['GET'])
def get_mask(mask_id):
    """Get individual mask data by ID"""
    try:
        if mask_id not in mask_storage:
            return jsonify({'error': f'Mask {mask_id} not found'}), 404
        
        mask_data = mask_storage[mask_id]
        mask_array = mask_data['mask']
        
        # Convert to list format only when sending to frontend
        if hasattr(mask_array, 'tolist'):
            mask_list = mask_array.tolist()
        else:
            mask_list = mask_array
        
        return jsonify({
            'success': True,
            'mask': mask_list
        })
        
    except Exception as e:
        print(f"Error getting mask {mask_id}: {str(e)}")
        return jsonify({'error': f'Failed to get mask: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 