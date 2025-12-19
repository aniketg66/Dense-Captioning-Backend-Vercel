"""
Vercel Serverless Function Wrapper for Flask App
This file wraps the Flask app to work with Vercel's serverless functions
"""
import sys
import os

# Add parent directory to path to import app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the Flask app
from app import app

# Vercel Python runtime expects the Flask app to be exported
# The Flask app is a WSGI application that Vercel can call directly

