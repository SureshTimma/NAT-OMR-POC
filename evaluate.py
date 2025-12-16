import cv2
import numpy as np
import imutils
import argparse
import json
import os

# Section Configuration - Updated layout
# Section 1: Pyschometric - 25 questions (4 options A-D)
# Section 2: Aptitude - 18 questions (4 options A-D)
# Section 3: Math - 17 questions (4 options A-D)
# Total: 60 questions
SECTIONS = {
    1: {
        "name": "Pyschometric",
        "questions": list(range(1, 26)),  # Q1-25
        "num_options": 4,  # A, B, C, D
        "columns": 3
    },
    2: {
        "name": "Aptitude",
        "questions": list(range(26, 44)),  # Q26-43
        "num_options": 4,  # A, B, C, D
        "columns": 3
    },
    3: {
        "name": "Math",
        "questions": list(range(44, 61)),  # Q44-60
        "num_options": 4,  # A, B, C, D
        "columns": 3
    }
}

def detect_section_boxes(image):
    """
    Detect the rectangular section boxes using color-based detection
    Returns list of bounding boxes sorted top to bottom
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Detect dark red/maroon borders
    # Dark red in HSV - adjusted for darker tones
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 200])
    lower_red2 = np.array([165, 30, 30])
    upper_red2 = np.array([180, 255, 200])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Save debug mask
    cv2.imwrite("debug_red_mask.jpg", red_mask)
    
    # Dilate to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.dilate(red_mask, kernel, iterations=3)
    
    # Close to fill gaps
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours - use RETR_TREE to get hierarchy
    cnts = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    section_boxes = []
    img_area = image.shape[0] * image.shape[1]
    
    for c in cnts:
        area = cv2.contourArea(c)
        # Filter by area (section boxes should be substantial - at least 5% of image)
        if area > img_area * 0.05:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Should be roughly rectangular
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(c)
                ar = w / float(h)
                # Sections are wider than tall typically
                if ar > 1.0:  # Width > Height
                    section_boxes.append((x, y, w, h, c))
    
    # Sort by Y coordinate (top to bottom)
    section_boxes = sorted(section_boxes, key=lambda b: b[1])
    
    return section_boxes


def detect_section_boxes_alternative(image):
    """
    Alternative method: Use edge detection to find rectangular boxes
    Useful if color detection doesn't work well
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    section_boxes = []
    img_area = image.shape[0] * image.shape[1]
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > img_area * 0.05:  # At least 5% of image
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if 4 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(c)
                section_boxes.append((x, y, w, h, c))
    
    section_boxes = sorted(section_boxes, key=lambda b: b[1])
    return section_boxes


def process_section(section_img, section_config, debug_prefix=""):
    """
    Process a single section and extract answers
    Returns dict of {question_num: detected_answer_index}
    """
    num_options = section_config["num_options"]
    questions = section_config["questions"]
    num_cols = section_config["columns"]
    
    # Convert to grayscale
    gray = cv2.cvtColor(section_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to find bubbles - use OTSU for automatic threshold
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_thresh.jpg", thresh)
    
    # Find ALL contours (not just external) to get bubbles inside section
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    bubble_cnts = []
    img_h, img_w = section_img.shape[:2]
    
    # Calculate expected bubble size based on section dimensions
    # Bubbles are typically about 2-4% of section width
    min_bubble_size = int(img_w * 0.015)
    max_bubble_size = int(img_w * 0.06)
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h) if h > 0 else 0
        area = cv2.contourArea(c)
        
        # Calculate circularity
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Filter for circular bubbles
        # - Size within expected range
        # - Aspect ratio close to 1 (circular)
        # - Circularity > 0.5 (reasonably round)
        # - Not touching edges
        if (min_bubble_size <= w <= max_bubble_size and 
            min_bubble_size <= h <= max_bubble_size and 
            0.7 <= ar <= 1.3 and
            circularity > 0.4):
            
            # Exclude edge artifacts (bubbles too close to section boundary)
            margin = 10
            if (x > margin and y > margin and 
                (x + w) < (img_w - margin) and 
                (y + h) < (img_h - margin)):
                bubble_cnts.append(c)
    
    print(f"  Found {len(bubble_cnts)} bubble candidates")
    
    if not bubble_cnts:
        print(f"  Warning: No bubbles detected in section")
        return {}
    
    # Debug: Draw detected bubbles
    if debug_prefix:
        debug_bubbles = section_img.copy()
        cv2.drawContours(debug_bubbles, bubble_cnts, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{debug_prefix}_bubbles.jpg", debug_bubbles)
    
    # Get bounding boxes for sorting
    bubble_data = []
    for c in bubble_cnts:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2  # Center X
        cy = y + h // 2  # Center Y
        bubble_data.append((x, y, w, h, cx, cy, c))
    
    # Split into columns based on center X coordinate
    # Use k-means style clustering for better column detection
    cxs = sorted(set([b[4] for b in bubble_data]))
    
    if len(cxs) < num_options:
        print(f"  Warning: Not enough distinct X positions for bubbles")
        return {}
    
    # Find column boundaries by looking for large gaps in X coordinates
    # First, group bubbles by approximate X position
    x_groups = []
    current_group = [cxs[0]]
    
    for i in range(1, len(cxs)):
        if cxs[i] - cxs[i-1] > img_w * 0.08:  # Gap > 8% of width = new column group
            x_groups.append(current_group)
            current_group = [cxs[i]]
        else:
            current_group.append(cxs[i])
    x_groups.append(current_group)
    
    # Each group represents bubbles in one column's worth of options
    # We expect num_cols * num_options groups approximately, or num_cols groups
    
    # Calculate column boundaries
    all_xs = [b[4] for b in bubble_data]
    x_min, x_max = min(all_xs), max(all_xs)
    col_width = (x_max - x_min) / num_cols
    
    # Assign bubbles to columns
    columns = [[] for _ in range(num_cols)]
    for data in bubble_data:
        cx = data[4]
        col_idx = min(int((cx - x_min) / col_width), num_cols - 1)
        columns[col_idx].append(data)
    
    # Sort each column top-to-bottom by center Y
    for i in range(num_cols):
        columns[i] = sorted(columns[i], key=lambda b: b[5])
    
    # Group bubbles into rows within each column
    answers = {}
    q_idx = 0
    
    for col_idx, col_bubbles in enumerate(columns):
        if not col_bubbles:
            continue
        
        # Group by Y coordinate (each question's bubbles are on same row)
        rows = []
        current_row = [col_bubbles[0]]
        
        for i in range(1, len(col_bubbles)):
            prev_cy = col_bubbles[i-1][5]
            curr_cy = col_bubbles[i][5]
            
            # If Y difference is small, same row (use percentage of image height)
            row_threshold = img_h * 0.03  # 3% of height
            if abs(curr_cy - prev_cy) < row_threshold:
                current_row.append(col_bubbles[i])
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [col_bubbles[i]]
        
        if current_row:
            rows.append(current_row)
        
        # Process each row
        for row in rows:
            # Allow some flexibility in row size (might miss a bubble occasionally)
            if len(row) < num_options - 1:
                continue
            
            # Sort row left-to-right by center X
            row = sorted(row, key=lambda b: b[4])
            
            if q_idx >= len(questions):
                break
            
            q_num = questions[q_idx]
            
            # Find which bubble is filled (most pixels)
            max_pixels = 0
            selected = None
            
            for opt_idx, bubble in enumerate(row):
                if opt_idx >= num_options:
                    break
                contour = bubble[6]
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [contour], -1, 255, -1)
                masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(masked)
                
                if total > max_pixels:
                    max_pixels = total
                    selected = opt_idx
            
            answers[q_num] = selected
            q_idx += 1
    
    print(f"  Detected answers for {len(answers)} questions")
    return answers


def evaluate_omr_sections(image_path, answer_key_path="answer_key.json", output_dir=None):
    """
    Main function to evaluate OMR sheet with section detection
    """
    print("=" * 60)
    print("OMR Section-Based Evaluation")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Resize for consistent processing
    width = 1200
    ratio = width / float(image.shape[1])
    resized = cv2.resize(image, (width, int(image.shape[0] * ratio)))
    
    # Load answer key
    try:
        with open(answer_key_path, 'r') as f:
            loaded_key = json.load(f)
            # Map letters to indices
            mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
            answer_key = {int(k): mapping.get(v, 0) for k, v in loaded_key.items()}
    except Exception as e:
        print(f"Warning: Could not load answer key: {e}")
        answer_key = {}
    
    # Detect section boxes
    print("\nDetecting section boxes...")
    section_boxes = detect_section_boxes(resized)
    
    if len(section_boxes) < 3:
        print(f"Warning: Only found {len(section_boxes)} sections, trying alternative method...")
        section_boxes = detect_section_boxes_alternative(resized)
    
    print(f"Found {len(section_boxes)} section boxes")
    
    # Debug: Draw detected sections
    debug_img = resized.copy()
    for i, (x, y, w, h, _) in enumerate(section_boxes):
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"Section {i+1}", (x+10, y+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    debug_sections_path = os.path.join(output_dir, "debug_sections.jpg") if output_dir else "debug_sections.jpg"
    cv2.imwrite(debug_sections_path, debug_img)
    print(f"Saved {debug_sections_path}")
    
    # Process each section
    all_answers = {}
    section_results = []
    
    for i, (x, y, w, h, _) in enumerate(section_boxes):
        section_num = i + 1
        if section_num not in SECTIONS:
            continue
        
        config = SECTIONS[section_num]
        print(f"\n--- Section {section_num}: {config['name']} ---")
        print(f"Questions: {config['questions'][0]}-{config['questions'][-1]}")
        print(f"Options: {config['num_options']}")
        
        # Extract section region with small padding
        pad = 5
        section_img = resized[max(0, y+pad):y+h-pad, max(0, x+pad):x+w-pad]
        
        debug_sec_filename = f"debug_section_{section_num}"
        debug_sec_path = os.path.join(output_dir, debug_sec_filename) if output_dir else debug_sec_filename
        
        cv2.imwrite(f"{debug_sec_path}.jpg", section_img)
        
        # Process section
        answers = process_section(section_img, config, debug_sec_path)
        all_answers.update(answers)
        
        # Calculate section score
        correct = 0
        total = len(config['questions'])
        
        for q in config['questions']:
            detected = answers.get(q)
            expected = answer_key.get(q - 1)  # Answer key is 0-indexed
            
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
        
        print(f"Section Score: {section_score:.1f}% ({correct}/{total})")
    
    # Overall results
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    
    total_correct = sum(r['correct'] for r in section_results)
    total_questions = sum(r['total'] for r in section_results)
    overall_score = (total_correct / total_questions) * 100 if total_questions > 0 else 0
    
    for r in section_results:
        print(f"  Section {r['section']} ({r['name']}): {r['score']:.1f}% ({r['correct']}/{r['total']})")
    
    print(f"\nOVERALL SCORE: {overall_score:.1f}% ({total_correct}/{total_questions})")
    
    # Save detected answers
    answer_letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    detected_output = {str(q): answer_letters.get(a, '?') for q, a in sorted(all_answers.items())}
    
    json_path = os.path.join(output_dir, "detected_answers.json") if output_dir else "detected_answers.json"
    with open(json_path, 'w') as f:
        json.dump(detected_output, f, indent=2)
    print(f"\nSaved {json_path}")
    
    return all_answers, section_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMR Evaluation with Section Detection")
    parser.add_argument("--image", required=True, help="Path to the OMR image")
    parser.add_argument("--key", default="answer_key.json", help="Path to answer key JSON")
    args = parser.parse_args()
    
    evaluate_omr_sections(args.image, args.key)

