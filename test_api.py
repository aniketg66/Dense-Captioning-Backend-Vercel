"""
Test script for Railway-deployed Flask API
Tests MedSAM HuggingFace endpoints and Transcribe API

Usage:
    python test_api.py

Make sure to update BASE_URL with your Railway deployment URL
"""

import requests
import json
import base64
import os
from pathlib import Path

# Update this with your Railway deployment URL
BASE_URL = "https://web-production-49608.up.railway.app"

def test_medsam_status():
    """Test MedSAM status endpoint"""
    print("\n" + "="*60)
    print("TEST 1: MedSAM Status Check")
    print("="*60)
    
    url = f"{BASE_URL}/api/medsam/status"
    response = requests.get(url)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def load_image_for_medsam(image_id):
    """
    Load an image for MedSAM segmentation from Supabase
    Requires: image_id from your Supabase database
    
    Args:
        image_id: The image ID from Supabase database
    """
    print("\n" + "="*60)
    print("TEST 2: Load Image for MedSAM from Supabase")
    print("="*60)
    
    url = f"{BASE_URL}/api/medsam/load_from_supabase"
    
    data = {
        "image_id": image_id
    }
    
    response = requests.post(url, json=data)
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if response.status_code == 200 and result.get('success'):
        print("✅ Image loaded successfully!")
        print(f"   Image URL: {result.get('image_url', 'N/A')}")
        print(f"   Mask count: {result.get('mask_count', 0)}")
        return True
    else:
        print("❌ Failed to load image")
        return False

def test_medsam_segment_points():
    """Test MedSAM segmentation with points"""
    print("\n" + "="*60)
    print("TEST 3: MedSAM Segment with Points")
    print("="*60)
    
    url = f"{BASE_URL}/api/medsam/segment_points"
    
    # Example points: [[x1, y1], [x2, y2], ...]
    # Labels: 1 = foreground, 0 = background
    data = {
        "points": [[100, 100], [200, 200]],  # Two points
        "labels": [1, 1]  # Both are foreground points
    }
    
    response = requests.post(url, json=data)
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if response.status_code == 200 and result.get('success'):
        print("✅ Segmentation successful!")
        print(f"   Method: {result.get('method')}")
        print(f"   Number of masks: {len(result.get('masks', []))}")
        return True
    else:
        print("❌ Segmentation failed")
        return False

def test_medsam_segment_box():
    """Test MedSAM segmentation with bounding box"""
    print("\n" + "="*60)
    print("TEST 4: MedSAM Segment with Bounding Box")
    print("="*60)
    
    url = f"{BASE_URL}/api/medsam/segment_box"
    
    # Bounding box: [x1, y1, x2, y2]
    data = {
        "bbox": [50, 50, 250, 250]  # [x1, y1, x2, y2]
    }
    
    response = requests.post(url, json=data)
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if response.status_code == 200 and result.get('success'):
        print("✅ Segmentation successful!")
        print(f"   Method: {result.get('method')}")
        print(f"   Confidence: {result.get('confidence')}")
        return True
    else:
        print("❌ Segmentation failed")
        return False

def test_medsam_segment_combined():
    """Test MedSAM segmentation with both points and box"""
    print("\n" + "="*60)
    print("TEST 5: MedSAM Segment Combined (Points + Box)")
    print("="*60)
    
    url = f"{BASE_URL}/api/medsam/segment_combined"
    
    data = {
        "points": [[150, 150]],
        "labels": [1],
        "bbox": [50, 50, 250, 250]
    }
    
    response = requests.post(url, json=data)
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if response.status_code == 200 and result.get('success'):
        print("✅ Combined segmentation successful!")
        return True
    else:
        print("❌ Combined segmentation failed")
        return False

def test_transcribe_audio(audio_file_path=None):
    """Test transcription API"""
    print("\n" + "="*60)
    print("TEST 6: Transcribe Audio")
    print("="*60)
    
    url = f"{BASE_URL}/api/transcribe"
    
    if audio_file_path and os.path.exists(audio_file_path):
        # Test with actual audio file
        with open(audio_file_path, 'rb') as f:
            files = {
                'audio': ('audio.wav', f, 'audio/wav')
            }
            
            # Optional: Add image filename and click timestamps
            data = {
                'image_filename': 'test_image.jpg',
                'click_timestamps': json.dumps([])  # Empty array = transcribe entire audio
            }
            
            response = requests.post(url, files=files, data=data)
    else:
        # Create a simple test audio file using a dummy approach
        # Note: This won't work without an actual audio file
        print("⚠️  No audio file provided. Creating dummy request...")
        print("   Please provide a valid audio file path for full testing")
        
        # For testing purposes, you can create a minimal WAV file
        # or use an existing audio file
        return False
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if response.status_code == 200:
        print("✅ Transcription successful!")
        if 'segmented_transcriptions' in result:
            print(f"   Number of segments: {len(result['segmented_transcriptions'])}")
        return True
    else:
        print("❌ Transcription failed")
        return False

def test_transcribe_with_segments(audio_file_path, click_timestamps):
    """Test transcription with click timestamps (segmented)"""
    print("\n" + "="*60)
    print("TEST 7: Transcribe Audio with Segments")
    print("="*60)
    
    url = f"{BASE_URL}/api/transcribe"
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    with open(audio_file_path, 'rb') as f:
        files = {
            'audio': ('audio.wav', f, 'audio/wav')
        }
        
        data = {
            'image_filename': 'test_image.jpg',
            'click_timestamps': json.dumps(click_timestamps)  # [1000, 3000, 5000] in milliseconds
        }
        
        response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if response.status_code == 200:
        print("✅ Segmented transcription successful!")
        if 'segmented_transcriptions' in result:
            print(f"   Number of segments: {len(result['segmented_transcriptions'])}")
            for i, seg in enumerate(result['segmented_transcriptions']):
                print(f"   Segment {i+1}: {seg.get('start_time_ms')}ms - {seg.get('end_time_ms')}ms")
        return True
    else:
        print("❌ Segmented transcription failed")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RAILWAY API TEST SUITE")
    print(f"Testing: {BASE_URL}")
    print("="*60)
    
    results = {}
    
    # Test MedSAM endpoints
    print("\n📋 Testing MedSAM HuggingFace Endpoints...")
    results['medsam_status'] = test_medsam_status()
    
    # Load image from Supabase (required for segmentation tests)
    # Update this with an actual image_id from your Supabase database
    SUPABASE_IMAGE_ID = None  # Set this to your image_id, e.g., "123e4567-e89b-12d3-a456-426614174000"
    
    if SUPABASE_IMAGE_ID:
        results['load_image'] = load_image_for_medsam(SUPABASE_IMAGE_ID)
        
        # Only test segmentation if image was loaded successfully
        if results.get('load_image'):
            results['segment_points'] = test_medsam_segment_points()
            results['segment_box'] = test_medsam_segment_box()
            results['segment_combined'] = test_medsam_segment_combined()
        else:
            print("\n⚠️  Skipping segmentation tests - image not loaded")
            results['segment_points'] = None
            results['segment_box'] = None
            results['segment_combined'] = None
    else:
        print("\n⚠️  SUPABASE_IMAGE_ID not set. Skipping image load and segmentation tests.")
        print("   To test segmentation:")
        print("   1. Upload an image to Supabase")
        print("   2. Get the image_id from Supabase")
        print("   3. Set SUPABASE_IMAGE_ID in this script")
        results['load_image'] = None
        results['segment_points'] = None
        results['segment_box'] = None
        results['segment_combined'] = None
    
    # Test Transcription endpoints
    print("\n📋 Testing Transcription Endpoints...")
    
    # Test with audio file if provided
    # Update this path to your test audio file
    audio_file = "test_audio.wav"  # Update this path
    
    if os.path.exists(audio_file):
        results['transcribe'] = test_transcribe_audio(audio_file)
        results['transcribe_segmented'] = test_transcribe_with_segments(
            audio_file, 
            [1000, 3000, 5000]  # Example click timestamps in milliseconds
        )
    else:
        print(f"\n⚠️  Audio file not found: {audio_file}")
        print("   Skipping transcription tests. Provide a valid audio file to test.")
        results['transcribe'] = None
        results['transcribe_segmented'] = None
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Configuration
    BASE_URL = "https://web-production-49608.up.railway.app"
    
    # Optional: Override BASE_URL from command line
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
        print(f"Using custom BASE_URL: {BASE_URL}")
    
    print(f"\n🔗 Testing API at: {BASE_URL}")
    print("📝 Make sure to update SUPABASE_IMAGE_ID in main() if testing MedSAM segmentation")
    print("📝 Make sure to provide a valid audio file path for transcription tests\n")
    
    main()

