#!/usr/bin/env python3
"""
Test script to verify MedSAM setup
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config module imported")
        print(f"  SUPABASE_URL: {'Set' if config.SUPABASE_URL else 'NOT SET'}")
        print(f"  SUPABASE_KEY: {'Set' if config.SUPABASE_KEY else 'NOT SET'}")
        print(f"  STORAGE_BUCKET: {config.STORAGE_BUCKET}")
        print(f"  MASKS_BUCKET: {config.MASKS_BUCKET}")
        print(f"  EMBEDDINGS_BUCKET: {config.EMBEDDINGS_BUCKET}")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from utils.supabase_client import SupabaseManager
        print("✓ SupabaseManager imported")
    except Exception as e:
        print(f"✗ Failed to import SupabaseManager: {e}")
        return False
    
    try:
        from utils.medsam_integration import MedSAMIntegrator
        print("✓ MedSAMIntegrator imported")
    except Exception as e:
        print(f"✗ Failed to import MedSAMIntegrator: {e}")
        return False
    
    return True

def test_supabase_connection():
    """Test Supabase connection"""
    print("\nTesting Supabase connection...")
    
    try:
        from utils.supabase_client import SupabaseManager
        manager = SupabaseManager()
        print("✓ SupabaseManager initialized")
        
        # Try to query images table
        try:
            response = manager.supabase.table('images').select('id').limit(1).execute()
            if response.data:
                print(f"✓ Images table exists and has data (found {len(response.data)} records)")
            else:
                print("⚠ Images table exists but is empty")
        except Exception as e:
            print(f"✗ Failed to query images table: {e}")
            print("  Make sure the 'images' table exists in your Supabase database")
        
        # Try to query masks2 table
        try:
            response = manager.supabase.table('masks2').select('id').limit(1).execute()
            print(f"✓ masks2 table exists")
            if response.data:
                print(f"  Found {len(response.data)} masks")
            else:
                print("  Table is empty (this is normal if no masks have been created yet)")
        except Exception as e:
            print(f"✗ masks2 table does not exist or is not accessible: {e}")
            print("  Run the SQL from MEDSAM_QUICK_START.md to create it")
        
        # Try to query embeddings2 table
        try:
            response = manager.supabase.table('embeddings2').select('id').limit(1).execute()
            print(f"✓ embeddings2 table exists")
        except Exception as e:
            print(f"✗ embeddings2 table does not exist or is not accessible: {e}")
            print("  Run the SQL from MEDSAM_QUICK_START.md to create it")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test Supabase connection: {e}")
        return False

def test_medsam_integrator():
    """Test MedSAM integrator"""
    print("\nTesting MedSAM integrator...")
    
    try:
        from utils.medsam_integration import MedSAMIntegrator
        integrator = MedSAMIntegrator()
        print("✓ MedSAMIntegrator initialized")
        
        if integrator.is_available():
            print("✓ MedSAM model is available")
        else:
            print("⚠ MedSAM model not available (this is OK, you can still use manual segmentation)")
            print("  To enable MedSAM model:")
            print("  1. Download: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            print("  2. Create models/ directory: mkdir -p models")
            print("  3. Move checkpoint: mv sam_vit_b_01ec64.pth models/medsam_vit_b.pth")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test MedSAM integrator: {e}")
        return False

def main():
    print("=" * 60)
    print("MedSAM Setup Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Supabase Connection", test_supabase_connection()))
    results.append(("MedSAM Integrator", test_medsam_integrator()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! MedSAM is ready to use.")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nQuick fixes:")
        print("1. Create backend/.env file with your Supabase credentials")
        print("2. Run the SQL from MEDSAM_QUICK_START.md in Supabase")
        print("3. Install missing packages: pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

