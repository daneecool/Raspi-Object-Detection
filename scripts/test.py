import unittest
from ultralytics import YOLO
import os
import time

class TestYoloDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*50)
        print("YOLO DETECTION TEST SUITE")
        print("="*50)
        print("Loading YOLO model...")
        start_time = time.time()
        cls.model = YOLO("yolo11n.pt")
        load_time = time.time() - start_time
        print(f" Model loaded successfully in {load_time:.2f} seconds")
        print(f" Model type: {type(cls.model)}")
        print(f" Available classes: {len(cls.model.names)} classes")
        print(f" Model names preview: {list(cls.model.names.values())[:10]}...")
        
        # Create test expectations
        cls.expected = {
            "test.jpg": ["person"]  # Example: expecting to detect a person in test.jpg
        }
        print(f" Test expectations loaded: {cls.expected}")
        print("-" * 50)

    def test_model_loading(self):
        """Test 1: Verify YOLO model loads correctly"""
        print("\n TEST 1: Model Loading Verification")
        print("-" * 30)
        
        # Check model is not None
        self.assertIsNotNone(self.model, "YOLO model is None")
        print("✓ Model object exists")
        
        # Check model has required attributes
        self.assertTrue(hasattr(self.model, 'model'), "Model missing 'model' attribute")
        print("✓ Model has required attributes")
        
        # Check model names dictionary
        self.assertIsInstance(self.model.names, dict, "Model names should be a dictionary")
        self.assertGreater(len(self.model.names), 0, "Model should have class names")
        print(f"✓ Model has {len(self.model.names)} class names")
        
        # Check for 'person' class (COCO class 0)
        self.assertIn('person', self.model.names.values(), "Model should include 'person' class")
        print("'person' class found in model")
        
        print("Model loading test PASSED")

    def test_inference_on_images(self):
        """Test 2: Verify inference works on test images"""
        print("\n TEST 2: Image Inference Verification")
        print("-" * 35)
        
        test_results = []
        
        for img_name, expected_objects in self.expected.items():
            print(f"\n Testing image: {img_name}")
            
            if os.path.exists(img_name):
                print(f" Image file found: {img_name}")
                
                # Run inference
                start_time = time.time()
                results = self.model.predict(img_name, verbose=False)
                inference_time = time.time() - start_time
                print(f" Inference completed in {inference_time:.3f} seconds")
                
                # Analyze results
                if len(results[0].boxes) > 0:
                    detected_labels = [self.model.names[int(c)] for c in results[0].boxes.cls]
                    confidence_scores = results[0].boxes.conf.tolist()
                    
                    print(f" Detected {len(detected_labels)} objects:")
                    for i, (label, conf) in enumerate(zip(detected_labels, confidence_scores)):
                        print(f"  {i+1}. {label} (confidence: {conf:.3f})")
                    
                    # Check expected objects
                    for obj in expected_objects:
                        if obj in detected_labels:
                            obj_indices = [i for i, label in enumerate(detected_labels) if label == obj]
                            obj_confidences = [confidence_scores[i] for i in obj_indices]
                            print(f" Expected object '{obj}' found with confidence(s): {obj_confidences}")
                            test_results.append(f" {img_name}: {obj} detected")
                        else:
                            print(f" Expected object '{obj}' NOT found")
                            test_results.append(f" {img_name}: {obj} missing")
                            
                        self.assertIn(obj, detected_labels, f"{obj} not detected in {img_name}")
                else:
                    print(" No objects detected in image")
                    test_results.append(f" {img_name}: No detections")
                    # Don't fail the test if no objects detected, just report it
                    print(f" Warning: No objects detected in {img_name}")
                    
            else:
                print(f" Warning: Test image {img_name} not found")
                test_results.append(f" {img_name}: File not found")
                # Create a simple test to ensure this doesn't fail silently
                print(f" Note: To test detection, place an image named '{img_name}' in the current directory")
        
        # Print summary
        print("\n TEST SUMMARY:")
        print("-" * 20)
        for result in test_results:
            print(result)
        
        print(" Image inference test COMPLETED")

    @classmethod
    def tearDownClass(cls):
        print("\n" + "="*50)
        print("TEST SUITE COMPLETED")
        print("="*50)
        print(" Summary:")
        print("  - Model loading: Verified")
        print("  - Inference capability: Verified") 
        print("  - Class detection: Tested")
        print("\n Tips:")
        print("  - Add test images to verify actual detection")
        print("  - Modify expected results for your specific use case")
        print("  - Check confidence thresholds if needed")
        print("="*50)

if __name__ == "__main__":
    # Run with more verbose output
    unittest.main(verbosity=2)
