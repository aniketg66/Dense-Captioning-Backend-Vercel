import json
import os
import uuid
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import io
import base64
import pickle
import threading
import time

class AnnotationManager:
    def __init__(self, sessions_dir='sessions'):
        """Initialize the annotation manager with fast, lazy saving"""
        self.sessions_dir = sessions_dir
        self.sessions = {}  # In-memory session storage
        self.sessions_file = os.path.join(sessions_dir, 'sessions.json')
        self.session_files = {}  # Individual session files for fast access
        self.save_lock = threading.Lock()  # Thread safety
        self.pending_saves = set()  # Track sessions that need saving
        
        # Create sessions directory if it doesn't exist
        os.makedirs(sessions_dir, exist_ok=True)
        
        # Load existing sessions (metadata only)
        self._load_sessions_metadata()
    
    def create_session(self, image_id, image_path, segments):
        """Create a new annotation session"""
        session_id = str(uuid.uuid4())
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_segments = []
        for segment in segments:
            serializable_segment = segment.copy()
            if isinstance(segment['mask'], np.ndarray):
                serializable_segment['mask'] = segment['mask'].tolist()
            serializable_segments.append(serializable_segment)
        
        session_data = {
            'session_id': session_id,
            'image_id': image_id,
            'image_path': image_path,
            'segments': serializable_segments,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'labels': {},
            'annotations': []
        }
        
        # Store in memory
        self.sessions[session_id] = session_data
        
        # Save session immediately (fast, individual file)
        self._save_session_fast(session_id)
        
        return session_id
    
    def get_session(self, session_id):
        """Get session data by ID - loads from individual file if needed"""
        if session_id not in self.sessions:
            return None
        
        # If session is in memory, return it
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Load from individual file if not in memory
        return self._load_session_fast(session_id)
    
    def update_label(self, session_id, segment_id, label):
        """Update label for a specific segment"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session['labels'][segment_id] = label
        session['updated_at'] = datetime.now().isoformat()
        
        # Update segment label
        for segment in session['segments']:
            if segment['id'] == segment_id:
                segment['label'] = label
                break
        
        # Mark for lazy save
        self.pending_saves.add(session_id)
        self._lazy_save_session(session_id)
    
    def add_segment(self, session_id, mask, bbox):
        """Add a new segment to the session - FAST VERSION"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Generate new segment ID
        segment_id = f"segment_{len(session['segments'])}"
        
        # Create new segment - keep mask as numpy array for efficiency
        new_segment = {
            'id': segment_id,
            'mask': mask if isinstance(mask, np.ndarray) else np.array(mask),
            'bbox': bbox,
            'color': self._get_next_color(session['segments']),
            'label': f"Region {len(session['segments']) + 1}",
            'confidence': 0.9,
            'method': 'manual'
        }
        
        session['segments'].append(new_segment)
        session['updated_at'] = datetime.now().isoformat()
        
        # Mark for lazy save - this is the key optimization
        self.pending_saves.add(session_id)
        self._lazy_save_session(session_id)
        
        return segment_id
    
    def _lazy_save_session(self, session_id):
        """Lazy save a session - only saves if it's been pending for a while"""
        def delayed_save():
            time.sleep(0.1)  # Wait 100ms before saving
            if session_id in self.pending_saves:
                self.pending_saves.remove(session_id)
                self._save_session_fast(session_id)
        
        # Start background save thread
        thread = threading.Thread(target=delayed_save, daemon=True)
        thread.start()
    
    def _save_session_fast(self, session_id):
        """Save a single session to its own file - FAST"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session_file = os.path.join(self.sessions_dir, f"{session_id}.pkl")
        
        try:
            with self.save_lock:
                # Use pickle for fast serialization of numpy arrays
                with open(session_file, 'wb') as f:
                    pickle.dump(session, f)
                
                self.session_files[session_id] = session_file
        except Exception as e:
            print(f"Warning: Could not save session {session_id}: {e}")
    
    def _load_session_fast(self, session_id):
        """Load a single session from its file - FAST"""
        session_file = os.path.join(self.sessions_dir, f"{session_id}.pkl")
        
        if not os.path.exists(session_file):
            return None
        
        try:
            with open(session_file, 'rb') as f:
                session = pickle.load(f)
            
            self.sessions[session_id] = session
            self.session_files[session_id] = session_file
            return session
        except Exception as e:
            print(f"Warning: Could not load session {session_id}: {e}")
            return None
    
    def _load_sessions_metadata(self):
        """Load only session metadata, not full data"""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r') as f:
                    sessions_metadata = json.load(f)
                
                # Only load metadata, not full session data
                for session_id, metadata in sessions_metadata.items():
                    if 'session_id' in metadata:
                        # This is a full session, load it properly
                        self._load_session_fast(session_id)
                    else:
                        # This is just metadata
                        self.sessions[session_id] = metadata
            except Exception as e:
                print(f"Warning: Could not load sessions metadata: {e}")
    
    def _save_sessions_metadata(self):
        """Save only session metadata, not full data"""
        try:
            # Create metadata-only version
            metadata = {}
            for session_id, session_data in self.sessions.items():
                if isinstance(session_data, dict) and 'session_id' in session_data:
                    # Full session - create metadata
                    metadata[session_id] = {
                        'session_id': session_id,
                        'image_id': session_data.get('image_id'),
                        'image_path': session_data.get('image_path'),
                        'num_segments': len(session_data.get('segments', [])),
                        'created_at': session_data.get('created_at'),
                        'updated_at': session_data.get('updated_at')
                    }
                else:
                    # Already metadata
                    metadata[session_id] = session_data
            
            with open(self.sessions_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save sessions metadata: {e}")
    
    def remove_segment(self, session_id, segment_id):
        """Remove a segment from the session"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Remove segment
        session['segments'] = [s for s in session['segments'] if s['id'] != segment_id]
        
        # Remove associated label
        if segment_id in session['labels']:
            del session['labels'][segment_id]
        
        session['updated_at'] = datetime.now().isoformat()
        
        # Mark for lazy save
        self.pending_saves.add(session_id)
        self._lazy_save_session(session_id)
    
    def add_annotation(self, session_id, annotation_data):
        """Add a new annotation to the session"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        annotation = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'data': annotation_data
        }
        
        session['annotations'].append(annotation)
        session['updated_at'] = datetime.now().isoformat()
        
        # Mark for lazy save
        self.pending_saves.add(session_id)
        self._lazy_save_session(session_id)
        
        return annotation['id']
    
    def export_session(self, session_id):
        """Export session data for download"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Create export data - convert numpy arrays to lists for JSON
        export_data = {
            'session_id': session_id,
            'image_id': session['image_id'],
            'image_path': session['image_path'],
            'segments': [],
            'labels': session['labels'],
            'annotations': session['annotations'],
            'created_at': session['created_at'],
            'updated_at': session['updated_at'],
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Convert segments for JSON export
        for segment in session['segments']:
            export_segment = segment.copy()
            if isinstance(segment['mask'], np.ndarray):
                export_segment['mask'] = segment['mask'].tolist()
            export_data['segments'].append(export_segment)
        
        return export_data
    
    def save_session_to_file(self, session_id, filepath):
        """Save session data to a JSON file"""
        export_data = self.export_session(session_id)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath
    
    def load_session_from_file(self, filepath):
        """Load session data from a JSON file"""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        session_id = session_data['session_id']
        
        # Convert lists back to numpy arrays
        for segment in session_data['segments']:
            if isinstance(segment['mask'], list):
                segment['mask'] = np.array(segment['mask'])
        
        self.sessions[session_id] = session_data
        self._save_session_fast(session_id)
        
        return session_id
    
    def get_session_summary(self, session_id):
        """Get a summary of the session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        summary = {
            'session_id': session_id,
            'image_id': session['image_id'],
            'num_segments': len(session['segments']),
            'num_labels': len(session['labels']),
            'num_annotations': len(session['annotations']),
            'created_at': session['created_at'],
            'updated_at': session['updated_at']
        }
        
        return summary
    
    def list_sessions(self):
        """List all sessions"""
        summaries = []
        for session_id in self.sessions.keys():
            summary = self.get_session_summary(session_id)
            if summary:
                summaries.append(summary)
        return summaries
    
    def delete_session(self, session_id):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            # Delete individual session file
            session_file = os.path.join(self.sessions_dir, f"{session_id}.pkl")
            if os.path.exists(session_file):
                os.remove(session_file)
            
            self._save_sessions_metadata()
    
    def _get_next_color(self, segments):
        """Get the next color for a new segment"""
        color_palette = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],    # Dark Red
            [0, 128, 0],    # Dark Green
            [0, 0, 128],    # Dark Blue
            [128, 128, 0],  # Olive
        ]
        
        used_colors = [segment['color'] for segment in segments]
        
        for color in color_palette:
            if color not in used_colors:
                return color
        
        # If all colors are used, generate a random one
        return [np.random.randint(0, 255) for _ in range(3)]
    
    def create_mask_image(self, mask, color=(255, 0, 0)):
        """Create a colored mask image"""
        if isinstance(mask, list):
            mask = np.array(mask)
        
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[mask] = color
        
        return colored_mask
    
    def mask_to_base64(self, mask, color=(255, 0, 0)):
        """Convert mask to base64 encoded image"""
        mask_image = self.create_mask_image(mask, color)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(mask_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def get_segment_statistics(self, session_id):
        """Get statistics about segments in a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        segments = session['segments']
        
        if not segments:
            return {}
        
        # Calculate statistics
        areas = []
        confidences = []
        methods = {}
        
        for segment in segments:
            mask = segment['mask'] if isinstance(segment['mask'], np.ndarray) else np.array(segment['mask'])
            area = np.sum(mask)
            areas.append(area)
            
            confidences.append(segment.get('confidence', 0))
            
            method = segment.get('method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        stats = {
            'total_segments': len(segments),
            'total_area': sum(areas),
            'average_area': np.mean(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'average_confidence': np.mean(confidences),
            'methods_used': methods,
            'labeled_segments': len(session['labels'])
        }
        
        return stats 