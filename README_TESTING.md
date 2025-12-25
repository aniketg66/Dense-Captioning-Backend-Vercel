# API Testing Guide

## Test Script: `test_api.py`

A comprehensive test script for testing the Railway-deployed Flask API, focusing on MedSAM HuggingFace endpoints and Transcription API.

## Setup

1. Install required dependencies:
```bash
pip install requests
```

2. Update configuration in `test_api.py`:
   - `BASE_URL`: Your Railway deployment URL (default: https://web-production-49608.up.railway.app)
   - `SUPABASE_IMAGE_ID`: Your image ID from Supabase (for MedSAM tests)
   - `audio_file`: Path to test audio file (for transcription tests)

## Running Tests

### Basic Test Run
```bash
python test_api.py
```

### Custom Base URL
```bash
python test_api.py https://your-custom-url.railway.app
```

## Test Coverage

### MedSAM HuggingFace Endpoints

1. **Status Check** (`/api/medsam/status`)
   - Tests if the HuggingFace Space is available
   - No prerequisites needed

2. **Load Image** (`/api/medsam/load_from_supabase`)
   - Loads an image from Supabase for segmentation
   - Requires: Valid `image_id` from Supabase database

3. **Segment with Points** (`/api/medsam/segment_points`)
   - Segments image using point prompts
   - Requires: Image loaded first
   - Parameters:
     - `points`: [[x1, y1], [x2, y2], ...]
     - `labels`: [1, 0, 1, ...] (1=foreground, 0=background)

4. **Segment with Bounding Box** (`/api/medsam/segment_box`)
   - Segments image using bounding box
   - Requires: Image loaded first
   - Parameters:
     - `bbox`: [x1, y1, x2, y2]

5. **Segment Combined** (`/api/medsam/segment_combined`)
   - Segments using both points and bounding box
   - Requires: Image loaded first

### Transcription Endpoints

1. **Transcribe Full Audio** (`/api/transcribe`)
   - Transcribes entire audio file
   - Requires: Audio file (WAV format)
   - Optional: `image_filename`, `click_timestamps`

2. **Transcribe Segmented** (`/api/transcribe`)
   - Transcribes audio segments based on click timestamps
   - Requires: Audio file + click timestamps array
   - Parameters:
     - `click_timestamps`: [1000, 3000, 5000] (milliseconds)

## Example Usage

### Testing MedSAM Segmentation

1. First, upload an image to Supabase and get the `image_id`
2. Update `SUPABASE_IMAGE_ID` in `test_api.py`
3. Run the tests:
```bash
python test_api.py
```

### Testing Transcription

1. Prepare a WAV audio file
2. Update `audio_file` path in `test_api.py`
3. Run the tests:
```bash
python test_api.py
```

## Expected Output

The script will output:
- ✅ PASSED: Test succeeded
- ❌ FAILED: Test failed
- ⏭️ SKIPPED: Test skipped (missing prerequisites)

## Troubleshooting

### MedSAM Tests Fail
- Ensure image is loaded first via `/api/medsam/load_from_supabase`
- Check that Supabase credentials are configured in Railway
- Verify HuggingFace Space is accessible

### Transcription Tests Fail
- Ensure audio file is in WAV format
- Check file path is correct
- Verify Whisper model is loaded (check Railway logs)

### Connection Errors
- Verify BASE_URL is correct
- Check Railway deployment is running
- Ensure no firewall blocking requests
