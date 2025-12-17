import cv2
import numpy as np
import imutils
import json
import os
from datetime import datetime

# Configuration for new OMR format (4 columns per section)
SECTIONS_CONFIG = {
    1: {
        "name": "Section 1 (Psychometric)",
        "questions": list(range(1, 26)),  # Q1-25
        "num_options": 4,  # A, B, C, D
        "columns": 4,
        "questions_per_col": [7, 6, 6, 6]
    },
    2: {
        "name": "Section 2 (Aptitude)",
        "questions": list(range(26, 44)),  # Q26-43
        "num_options": 4,  # A, B, C, D
        "columns": 4,
        "questions_per_col": [5, 5, 4, 4]
    },
    3: {
        "name": "Section 3 (Math)",
        "questions": list(range(44, 61)),  # Q44-60
        "num_options": 4,  # A, B, C, D
        "columns": 4,
        "questions_per_col": [5, 4, 4, 4]
    }
}

class OMRDebugger:
    """Class to handle all debug visualization"""
    
    def __init__(self, output_dir="debug_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
    def save_image(self, image, name):
        """Save debug image"""
        path = os.path.join(self.session_dir, f"{name}.jpg")
        cv2.imwrite(path, image)
        print(f"  [DEBUG] Saved: {name}.jpg")
        return path
    
    def draw_contours_debug(self, image, contours, color=(0, 255, 0), thickness=2):
        """Draw contours on image for debugging"""
        debug_img = image.copy()
        cv2.drawContours(debug_img, contours, -1, color, thickness)
        return debug_img

def detect_alignment_markers(image):
    """Detect the 4 corner alignment markers (black squares)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold for black markers
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    markers = []
    img_area = image.shape[0] * image.shape[1]
    
    for c in cnts:
        area = cv2.contourArea(c)
        # Markers should be small (< 1% of image)
        if img_area * 0.0001 < area < img_area * 0.001:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h) if h > 0 else 0
            # Should be roughly square
            if 0.8 <= ar <= 1.2:
                markers.append((x, y, w, h))
    
    # Sort markers: TL, TR, BL, BR
    if len(markers) >= 4:
        markers = sorted(markers, key=lambda m: (m[1], m[0]))
        return markers
    
    return None

def detect_red_boxes(image, debugger=None):
    """Detect red-bordered boxes (question sections only, not phone fields)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Dark red/maroon color range
    lower_red1 = np.array([0, 50, 30])
    upper_red1 = np.array([10, 255, 200])
    lower_red2 = np.array([170, 50, 30])
    upper_red2 = np.array([180, 255, 200])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    if debugger:
        debugger.save_image(red_mask, "01_red_mask")
    
    # Morphological operations to connect borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    if debugger:
        debugger.save_image(red_mask, "02_red_mask_processed")
    
    # Find contours
    cnts = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    all_boxes = []
    boxes = []
    img_area = image.shape[0] * image.shape[1]
    img_height = image.shape[0]
    
    for c in cnts:
        area = cv2.contourArea(c)
        # All red boxes should be at least 3% of image
        if area > img_area * 0.03:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h) if h > 0 else 0
            
            # Store all potential boxes for debugging
            all_boxes.append({
                'x': x, 'y': y, 'w': w, 'h': h, 
                'area': area, 'ar': ar, 'contour': c
            })
            
            # Question sections are:
            # 1. Larger (> 8% of image area)
            # 2. Wider aspect ratio (width/height > 2.0)
            # 3. Located in lower 70% of image (y > 30% of height)
            
            is_question_section = (
                area > img_area * 0.08 and  # Large enough
                ar > 2.0 and                 # Wide enough
                y > img_height * 0.3         # Below phone fields
            )
            
            if is_question_section:
                boxes.append((x, y, w, h, c))
    
    # Debug: Print all detected boxes
    print(f"\nDetected {len(all_boxes)} total red boxes:")
    for i, box in enumerate(all_boxes):
        box_type = "QUESTION SECTION" if any(
            b[0] == box['x'] and b[1] == box['y'] for b in boxes
        ) else "PHONE FIELD (ignored)"
        print(f"  Box {i+1}: Area={box['area']:.0f}, AR={box['ar']:.2f}, "
              f"Y={box['y']}, Size={box['w']}x{box['h']} -> {box_type}")
    
    print(f"\nFiltered to {len(boxes)} question sections")
    
    # Sort by Y coordinate (top to bottom)
    boxes = sorted(boxes, key=lambda b: b[1])
    
    return boxes

def process_section(section_img, section_config, section_num, debugger=None):
    """
    Process a single section and detect bubble answers
    Uses techniques from evaluate.py
    """
    print(f"\n{'='*60}")
    print(f"Processing {section_config['name']}")
    print(f"{'='*60}")
    
    num_options = section_config["num_options"]
    questions = section_config["questions"]
    num_cols = section_config["columns"]
    
    # Convert to grayscale
    gray = cv2.cvtColor(section_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold using OTSU
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    if debugger:
        debugger.save_image(thresh, f"03_section{section_num}_thresh")
    
    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    bubble_cnts = []
    img_h, img_w = section_img.shape[:2]
    
    # Calculate expected bubble size
    min_bubble_size = int(img_w * 0.015)
    max_bubble_size = int(img_w * 0.06)
    
    print(f"Image size: {img_w}x{img_h}")
    print(f"Bubble size range: {min_bubble_size} - {max_bubble_size}")
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h) if h > 0 else 0
        area = cv2.contourArea(c)
        
        # Calculate circularity
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Filter for circular bubbles
        if (min_bubble_size <= w <= max_bubble_size and 
            min_bubble_size <= h <= max_bubble_size and 
            0.7 <= ar <= 1.3 and
            circularity > 0.4):
            
            # Exclude edge artifacts
            margin = 10
            if (x > margin and y > margin and 
                (x + w) < (img_w - margin) and 
                (y + h) < (img_h - margin)):
                bubble_cnts.append(c)
    
    print(f"Found {len(bubble_cnts)} bubble candidates")
    
    if not bubble_cnts:
        print("WARNING: No bubbles detected!")
        return {}
    
    # Debug: Draw detected bubbles
    if debugger:
        debug_bubbles = section_img.copy()
        cv2.drawContours(debug_bubbles, bubble_cnts, -1, (0, 255, 0), 2)
        debugger.save_image(debug_bubbles, f"04_section{section_num}_bubbles")
    
    # Get bubble data
    bubble_data = []
    for c in bubble_cnts:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2  # Center X
        cy = y + h // 2  # Center Y
        bubble_data.append((x, y, w, h, cx, cy, c))
    
    # Split bubbles into columns
    all_xs = [b[4] for b in bubble_data]
    x_min, x_max = min(all_xs), max(all_xs)
    col_width = (x_max - x_min) / num_cols
    
    columns = [[] for _ in range(num_cols)]
    for data in bubble_data:
        cx = data[4]
        col_idx = min(int((cx - x_min) / col_width), num_cols - 1)
        columns[col_idx].append(data)
    
    # Sort each column by Y coordinate
    for i in range(num_cols):
        columns[i] = sorted(columns[i], key=lambda b: b[5])
        print(f"Column {i+1}: {len(columns[i])} bubbles")
    
    # Process each column and group into rows
    answers = {}
    q_idx = 0
    
    for col_idx, col_bubbles in enumerate(columns):
        if not col_bubbles:
            continue
        
        # Group by Y coordinate (same row)
        rows = []
        current_row = [col_bubbles[0]]
        
        for i in range(1, len(col_bubbles)):
            prev_cy = col_bubbles[i-1][5]
            curr_cy = col_bubbles[i][5]
            
            row_threshold = img_h * 0.03
            if abs(curr_cy - prev_cy) < row_threshold:
                current_row.append(col_bubbles[i])
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [col_bubbles[i]]
        
        if current_row:
            rows.append(current_row)
        
        print(f"Column {col_idx+1}: {len(rows)} rows detected")
        
        # Process each row
        for row in rows:
            if len(row) < num_options - 1:  # Allow some tolerance
                continue
            
            # Sort row left-to-right
            row = sorted(row, key=lambda b: b[4])
            
            if q_idx >= len(questions):
                break
            
            q_num = questions[q_idx]
            
            # Find filled bubble (most pixels)
            max_pixels = 0
            selected = None
            option_pixels = []
            
            for opt_idx, bubble in enumerate(row):
                if opt_idx >= num_options:
                    break
                contour = bubble[6]
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [contour], -1, 255, -1)
                masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(masked)
                option_pixels.append(total)
                
                if total > max_pixels:
                    max_pixels = total
                    selected = opt_idx
            
            answers[q_num] = selected
            answer_letters = ['A', 'B', 'C', 'D', 'E']
            print(f"  Q{q_num}: {answer_letters[selected] if selected is not None else '?'} (pixels: {option_pixels})")
            q_idx += 1
    
    print(f"Detected answers for {len(answers)}/{len(questions)} questions")
    return answers

def evaluate_omr_opencv(image_path, answer_key_path="answer_key.json"):
    """
    Main evaluation function using OpenCV with comprehensive debugging
    """
    print("\n" + "="*70)
    print("OMR EVALUATION WITH OPENCV - NEW FORMAT (4 COLUMNS)")
    print("="*70)
    
    # Initialize debugger
    debugger = OMRDebugger()
    print(f"\nDebug output directory: {debugger.session_dir}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return None
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Resize for consistent processing
    width = 1400
    ratio = width / float(image.shape[1])
    resized = cv2.resize(image, (width, int(image.shape[0] * ratio)))
    debugger.save_image(resized, "00_resized_input")
    
    # Load answer key
    try:
        with open(answer_key_path, 'r') as f:
            loaded_key = json.load(f)
            mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
            answer_key = {int(k): mapping.get(v, 0) for k, v in loaded_key.items()}
        print(f"Answer key loaded: {len(answer_key)} questions")
    except Exception as e:
        print(f"WARNING: Could not load answer key: {e}")
        answer_key = {}
    
    # Detect section boxes
    print("\nDetecting section boxes...")
    section_boxes = detect_red_boxes(resized, debugger)
    print(f"Found {len(section_boxes)} section boxes")
    
    # Draw detected sections
    debug_sections = resized.copy()
    for i, (x, y, w, h, _) in enumerate(section_boxes):
        cv2.rectangle(debug_sections, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(debug_sections, f"Section {i+1}", (x+10, y+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    debugger.save_image(debug_sections, "05_detected_sections")
    
    # Process each section
    all_answers = {}
    section_results = []
    
    for i, (x, y, w, h, _) in enumerate(section_boxes[:3]):  # Only process first 3 sections
        section_num = i + 1
        if section_num not in SECTIONS_CONFIG:
            continue
        
        config = SECTIONS_CONFIG[section_num]
        
        # Extract section with padding
        pad = 10
        section_img = resized[max(0, y+pad):min(resized.shape[0], y+h-pad), 
                               max(0, x+pad):min(resized.shape[1], x+w-pad)]
        
        # Process section
        answers = process_section(section_img, config, section_num, debugger)
        all_answers.update(answers)
        
        # Calculate score
        correct = 0
        total = len(config['questions'])
        
        for q in config['questions']:
            detected = answers.get(q)
            expected = answer_key.get(q)
            
            if detected is not None and detected == expected:
                correct += 1
        
        section_score = (correct / total) * 100 if total > 0 else 0
        section_results.append({
            "section": section_num,
            "name": config['name'],
            "correct": correct,
            "total": total,
            "score": section_score
        })
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    total_correct = sum(r['correct'] for r in section_results)
    total_questions = sum(r['total'] for r in section_results)
    overall_score = (total_correct / total_questions) * 100 if total_questions > 0 else 0
    
    for r in section_results:
        print(f"  {r['name']}: {r['score']:.1f}% ({r['correct']}/{r['total']})")
    
    print(f"\n  OVERALL SCORE: {overall_score:.1f}% ({total_correct}/{total_questions})")
    print("="*70)
    
    # Save results
    answer_letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    detected_output = {str(q): answer_letters.get(a, '?') for q, a in sorted(all_answers.items())}
    
    results_path = os.path.join(debugger.session_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "overall_score": overall_score,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "sections": section_results,
            "detected_answers": detected_output
        }, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")
    print(f"All debug images saved to: {debugger.session_dir}")
    
    return all_answers, section_results, debugger.session_dir

if __name__ == "__main__":
    image_path = "omr_sheet_new_updates_filled.jpg"
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
    else:
        evaluate_omr_opencv(image_path)
