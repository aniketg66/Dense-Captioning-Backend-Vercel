from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import requests
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        # Get the audio data from the request
        data = request.get_json()
        
        if not data or 'audio' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        audio_base64 = data['audio']
        
        # Decode base64 audio
        audio_binary = base64.b64decode(audio_base64)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            temp_file.write(audio_binary)
            temp_file_path = temp_file.name
        
        try:
            # Prepare the request to OpenAI Whisper API
            with open(temp_file_path, 'rb') as audio_file:
                files = {
                    'file': ('audio.webm', audio_file, 'audio/webm')
                }
                data = {
                    'model': 'whisper-1'
                }
                
                headers = {
                    'Authorization': f'Bearer {OPENAI_API_KEY}'
                }
                
                # Make request to OpenAI
                response = requests.post(
                    'https://api.openai.com/v1/audio/transcriptions',
                    files=files,
                    data=data,
                    headers=headers
                )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if response.status_code == 200:
                result = response.json()
                return jsonify({
                    'transcript': result['text'],
                    'success': True
                })
            else:
                print(f"OpenAI API error: {response.status_code} - {response.text}")
                return jsonify({
                    'error': f'OpenAI API error: {response.status_code}',
                    'success': False
                }), 400
                
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({
            'error': f'Error processing audio: {str(e)}',
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Whisper service is running'})

if __name__ == '__main__':
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY environment variable not set!")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 