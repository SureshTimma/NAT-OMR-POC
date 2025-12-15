import os
import shutil
import time
import subprocess
import glob

def run_cross_tests():
    keys = [
        "answer_key_all_a.json", 
        "answer_key_all_b.json", 
        "answer_key_all_c.json", 
        "answer_key_all_d.json"
    ]
    
    images = [
        "omr_sheet_filled_a.jpg", 
        "omr_sheet_filled_b.jpg", 
        "omr_sheet_filled_c.jpg", 
        "omr_sheet_filled_d.jpg"
    ]
    
    timestamp = int(time.time())
    base_test_dir = os.path.join(os.getcwd(), "test")
    cross_test_dir = os.path.join(base_test_dir, f"cross_test_{timestamp}")
    
    if not os.path.exists(cross_test_dir):
        os.makedirs(cross_test_dir)
        
    print(f"Starting Cross-Test Evaluation Matrix (4x4)")
    print(f"Results will be saved in: {cross_test_dir}\n")
    
    summary_report = []
    
    for img_name in images:
        for key_name in keys:
            # Extract identifiers (e.g., 'a' from 'omr_sheet_filled_a.jpg')
            img_id = img_name.split('_')[-1].split('.')[0].upper()
            key_id = key_name.split('_')[-1].split('.')[0].upper()
            
            test_name = f"IMG_{img_id}_vs_KEY_{key_id}"
            test_dir = os.path.join(cross_test_dir, test_name)
            os.makedirs(test_dir)
            
            print(f"Running: {test_name}...")
            
            # Paths
            src_img = os.path.join(os.getcwd(), img_name)
            src_key = os.path.join(os.getcwd(), key_name)
            dst_img = os.path.join(test_dir, "filled.jpg")
            dst_key = os.path.join(test_dir, "expected_key.json")
            
            # Setup test files
            shutil.copy(src_img, dst_img)
            shutil.copy(src_key, dst_key)
            
            # Run evaluation
            cmd = ["python", "evaluate.py", "--image", dst_img, "--key", dst_key]
            
            # Capture output
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            # Parse score from output
            score_line = [line for line in result.stdout.split('\n') if "OVERALL SCORE:" in line]
            score = score_line[0].strip() if score_line else "Error"
            
            summary_report.append(f"{test_name}: {score}")
            
            # Move debug output to test dir (evaluate.py saves to cwd by default/design currently?)
            # Wait, evaluate.py saves to CWD or logic? 
            # Looking at evaluate.py: 
            # It saves debug_bubbles.jpg to current directory.
            # We ran it in CWD. So we need to move the outputs.
            
            if os.path.exists("debug_bubbles.jpg"):
                shutil.move("debug_bubbles.jpg", os.path.join(test_dir, "debug_bubbles.jpg"))
            if os.path.exists("detected_answers.json"):
                shutil.move("detected_answers.json", os.path.join(test_dir, "detected_answers.json"))
            
            # Save individual report
            with open(os.path.join(test_dir, "report.txt"), "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nERRORS:\n")
                    f.write(result.stderr)
            
            print(f"  Result: {score}")

    # Write summary
    summary_path = os.path.join(cross_test_dir, "summary_matrix.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_report))
        
    print("\n" + "="*50)
    print("CROSS TEST COMPLETED")
    print("="*50)
    for line in summary_report:
        print(line)

if __name__ == "__main__":
    run_cross_tests()
