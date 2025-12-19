import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def check_database():
    print("Checking database tables...")
    
    # Check chart_elements table
    try:
        chart_response = supabase.table('chart_elements').select('image_id').execute()
        print(f"Chart elements: {len(chart_response.data)} records")
        
        # Get unique image IDs
        unique_image_ids = set()
        for record in chart_response.data:
            unique_image_ids.add(record['image_id'])
        
        print(f"Unique image IDs in chart_elements: {len(unique_image_ids)}")
        print("Image IDs:", list(unique_image_ids))
        
        if chart_response.data:
            print("Sample chart element:", chart_response.data[0])
    except Exception as e:
        print(f"Error fetching chart_elements: {e}")
    
    # Check text_elements table
    try:
        text_response = supabase.table('text_elements').select('image_id').execute()
        print(f"Text elements: {len(text_response.data)} records")
        
        # Get unique image IDs
        unique_text_image_ids = set()
        for record in text_response.data:
            unique_text_image_ids.add(record['image_id'])
        
        print(f"Unique image IDs in text_elements: {len(unique_text_image_ids)}")
        print("Image IDs:", list(unique_text_image_ids))
        
        if text_response.data:
            print("Sample text element:", text_response.data[0])
    except Exception as e:
        print(f"Error fetching text_elements: {e}")
    
    # Check images table
    try:
        images_response = supabase.table('images').select('id, storage_link, task_id').execute()
        print(f"Images: {len(images_response.data)} records")
        if images_response.data:
            print("Sample image:", images_response.data[0])
    except Exception as e:
        print(f"Error fetching images: {e}")

def check_tasks_and_annotations():
    print("\nChecking tasks and annotations...")
    
    # Check tasks table
    try:
        tasks_response = supabase.table('tasks').select('id, name, category_id, isReady').execute()
        print(f"Tasks: {len(tasks_response.data)} records")
        for task in tasks_response.data:
            print(f"Task: {task['id']} - {task['name']} - Ready: {task.get('isReady', False)}")
    except Exception as e:
        print(f"Error fetching tasks: {e}")
    
    # Check annotation_sessions table
    try:
        annotations_response = supabase.table('annotation_sessions').select('id, user_id, image_id, iscompleted').execute()
        print(f"Annotation sessions: {len(annotations_response.data)} records")
        if annotations_response.data:
            print("Sample annotation:", annotations_response.data[0])
    except Exception as e:
        print(f"Error fetching annotation_sessions: {e}")

if __name__ == "__main__":
    check_database()
    check_tasks_and_annotations() 