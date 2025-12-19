#!/usr/bin/env python3
"""
Test script to verify the updated inference mechanism with chart_label+.pth
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, '.')

from science_analyzer import ScientificImageAnalyzer

def test_updated_inference():
    """Test the updated inference mechanism"""
    print("\n🧪 TESTING UPDATED INFERENCE MECHANISM")
    print("="*60)
    
    # Test images to try
    test_images = [
        '../science2.jpg',
        '../science3.png', 
        '../static/images/science3.png'
    ]
    
    # Create analyzer
    print("🔧 Creating ScientificImageAnalyzer...")
    analyzer = ScientificImageAnalyzer()
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n🔍 Testing with: {img_path}")
            print("-" * 40)
            
            try:
                # Test full analysis
                result = analyzer.analyze_image(img_path)
                
                if 'error' not in result:
                    print("✅ Analysis successful!")
                    print(f"📊 Chart Type: {result.get('chart_type', 'None')}")
                    print(f"🎯 Confidence: {result.get('chart_type_confidence', 0):.3f}")
                    print(f"🔢 Chart Elements: {result.get('total_detections', 0)}")
                    print(f"📝 Text Elements: {len(result.get('text_elements', []))}")
                    
                    # Show top 5 chart elements
                    chart_elements = result.get('chart_elements', [])
                    if chart_elements:
                        print("\n🎯 Top 5 Chart Elements:")
                        sorted_elements = sorted(chart_elements, key=lambda x: x['confidence'], reverse=True)
                        for i, elem in enumerate(sorted_elements[:5]):
                            print(f"  {i+1}. {elem['element_type']}: {elem['confidence']:.3f}")
                    
                    break  # Test successful, exit
                else:
                    print(f"❌ Analysis failed: {result['error']}")
                    
            except Exception as e:
                print(f"❌ Exception during analysis: {e}")
                import traceback
                traceback.print_exc()
                
    else:
        print("⚠️  No test images found!")
        print("Available images:")
        for root, dirs, files in os.walk('..'):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print(f"   - {os.path.join(root, file)}")

if __name__ == '__main__':
    test_updated_inference() 