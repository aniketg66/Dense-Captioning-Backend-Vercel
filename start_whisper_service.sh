#!/bin/bash

# Load OpenAI API key from environment variable or .env file
# DO NOT hardcode API keys in this file!
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable not set!"
    echo "Please set it using: export OPENAI_API_KEY='your-key-here'"
    echo "Or create a .env file with: OPENAI_API_KEY=your-key-here"
fi

# Install dependencies if needed
pip3 install -r requirements.txt

# Start the Flask service
python3 whisper_service.py 