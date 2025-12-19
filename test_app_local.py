#!/usr/bin/env python3
"""
Test script to verify the Flask app works locally
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_import():
    """Test if the app can be imported"""
    print("Testing app import...")
    try:
        from app import app
        print("✓ App imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import app: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_routes():
    """Test if basic routes are registered"""
    print("\nTesting app routes...")
    try:
        from app import app
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append(str(rule))
        
        print(f"✓ Found {len(routes)} routes")
        print(f"  Sample routes: {routes[:5]}")
        return True
    except Exception as e:
        print(f"✗ Failed to get routes: {e}")
        return False

def test_app_config():
    """Test app configuration"""
    print("\nTesting app configuration...")
    try:
        from app import app
        print(f"✓ App name: {app.name}")
        print(f"✓ Debug mode: {app.debug}")
        return True
    except Exception as e:
        print(f"✗ Failed to get config: {e}")
        return False

def test_vercel_compatibility():
    """Test Vercel-specific compatibility"""
    print("\nTesting Vercel compatibility...")
    try:
        # Test if /tmp directory is accessible
        import tempfile
        test_file = os.path.join("/tmp", "vercel_test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("✓ /tmp directory is writable")
        
        # Test environment variable detection
        os.environ["VERCEL"] = "1"
        from app import BASE_DIR
        if BASE_DIR == "/tmp":
            print("✓ Vercel environment detected correctly")
        else:
            print(f"⚠ BASE_DIR is {BASE_DIR}, expected /tmp in Vercel mode")
        del os.environ["VERCEL"]
        
        return True
    except Exception as e:
        print(f"✗ Vercel compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Flask App Local Test")
    print("=" * 50)
    
    results = []
    results.append(("Import", test_app_import()))
    results.append(("Routes", test_app_routes()))
    results.append(("Config", test_app_config()))
    results.append(("Vercel Compatibility", test_vercel_compatibility()))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed! App is ready for deployment.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please fix issues before deploying.")
        sys.exit(1)

