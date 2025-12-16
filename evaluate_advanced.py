import cv2
import numpy as np
import imutils
import argparse
import json
import os

# Configuration from generate_advanced.py
SECTIONS = [
     # Phone Sections (we will detect these dynamically, but knowing order helps)
]

ANSWER_SECTIONS = {
    1: {"name": "Section: 1 (Pyschometric)", "questions": list(range(1, 26)), "num_options": 4, "columns": 3},
    2: {"name": "Section: 2 (Aptitude)", "questions": list(range(26, 44)), "num_options": 4, "columns": 3},
    3: {"name": "Section: 3 (Math)", "questions": list(range(44, 61)), "num_options": 4, "columns": 3} # Q44-60
}

def order_points(pts):
    """Order coordinates: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Perspective transform to get top-down view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_alignment_markers(image):
    """Find the 4 corner markers"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold - markers are black
    thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # We expect 4 markers, roughly square, substantial size
    markers = []
    
    h, w = image.shape[:2]
    min_area = (w * h) * 0.001 
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                markers.append(c)
                
    if len(markers) == 4:
        # Get centers
        centers = []
        for m in markers:
            M = cv2.moments(m)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
        return np.array(centers, dtype="float32")
        
    return None

def detect_red_boxes(image):
    """Detect all red boxes (Phone fields and Answer Sections)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red color range
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cnts = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000: # Filter noise
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h, c))
            
    # Sort top-to-bottom
    boxes = sorted(boxes, key=lambda b: b[1])
    return boxes

def process_phone_box(roi):
    """Read a phone number box (10 columns, 0-9 bubbles)"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    bg_h, bg_w = roi.shape[:2]
    min_size = bg_w * 0.02
    max_size = bg_w * 0.1
    
    bubbles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if 0.8 <= ar <= 1.2 and min_size < w < max_size:
            bubbles.append((x, y, w, h, c))
            
    if not bubbles:
        return "?"
        
    # Sort bubbles into 10 columns
    bubbles = sorted(bubbles, key=lambda b: b[0]) # Sort by X first
    
    # Cluster into columns
    columns = []
    if len(bubbles) > 0:
        current_col = [bubbles[0]]
        for i in range(1, len(bubbles)):
            if bubbles[i][0] - bubbles[i-1][0] > bg_w * 0.05: # Gap
                 columns.append(current_col)
                 current_col = [bubbles[i]]
            else:
                 current_col.append(bubbles[i])
        columns.append(current_col)
    
    phone_number = ""
    
    for col in columns:
        # Sort top-to-bottom (0 to 9)
        col = sorted(col, key=lambda b: b[1])
        
        # Robust Grid Matching:
        # We expect 10 bubbles. If we have missing ones, simple enumerate fails.
        # We can detect the grid range from the first and last detected bubbles (or box bounds if clean)
        # But even better: we know there are 10 rows evenly spaced.
        
        if not col:
            phone_number += "_"
            continue
            
        # Estimate grid from detected bubbles
        min_y = min(b[1] for b in col)
        max_y = max(b[1] for b in col)
        
        # If we have few bubbles, this estimate is weak.
        # Fallback: We know the 10 rows span roughly most of the ROI height minus margins.
        # Let's try to map each bubble to a digit 0-9 based on normalized Y position.
        
        # Assuming height of ROI contains the 10 rows plus some margin
        # The bubbles are roughly centered in 10 slots.
        
        filled_digit = -1
        max_pixels = 0
        
        # Use box height to define slots
        row_height = bg_h / 10.0
        
        for (bx, by, bw, bh, c) in col:
            # Determine digit by centroid Y
            cy = by + bh/2
            digit_idx = int(cy / row_height)
            digit_idx = min(max(digit_idx, 0), 9) # Clamp 0-9
            
            # Check filling
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            masked = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(masked)
            
            if total > max_pixels and total > (bw*bh)*0.5:
                max_pixels = total
                filled_digit = digit_idx
                
        if filled_digit != -1:
            phone_number += str(filled_digit)
        else:
            phone_number += "_"
            
    return phone_number

def process_section_box(roi, config):
    """Reuse existing logic to process answer sections"""
    # This is a simplified version of the logic from evaluate.py
    # Re-implementing essentially the same steps adapted for the ROI
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    h, w = roi.shape[:2]
    bubbles = []
    
    # Filter bubbles
    for c in cnts:
        (x, y, bw, bh) = cv2.boundingRect(c)
        ar = bw / float(bh)
        if 0.8 <= ar <= 1.2 and w*0.01 < bw < w*0.05:
            bubbles.append((x, y, bw, bh, c))
            
    # Sort bubbles into columns
    bubbles = sorted(bubbles, key=lambda b: b[0])
    num_cols = config["columns"]
    
    # Quick clustering
    cols = [[] for _ in range(num_cols)]
    if bubbles:
        min_x = min(b[0] for b in bubbles)
        max_x = max(b[0] for b in bubbles)
        span = max_x - min_x
        # Avoid div by zero
        if span == 0: span = 1
        
        for b in bubbles:
            # Determine column index 0..num_cols-1
            # Center X of bubble
            cx = b[0] + b[2]//2
            norm_x = (cx - min_x) / span
            col_idx = int(norm_x * num_cols)
            col_idx = min(col_idx, num_cols-1)
            cols[col_idx].append(b)
            
    answers = {}
    q_start = config["questions"][0]
    
    for col_idx, col_bubbles in enumerate(cols):
        # Sort rows
        col_bubbles = sorted(col_bubbles, key=lambda b: b[1])
        
        # Group into questions
        # Each question has num_options bubbles in a row
        rows = []
        if col_bubbles:
            curr_row = [col_bubbles[0]]
            for i in range(1, len(col_bubbles)):
                if col_bubbles[i][1] - col_bubbles[i-1][1] < h*0.015: # Same row
                    curr_row.append(col_bubbles[i])
                else:
                    rows.append(curr_row)
                    curr_row = [col_bubbles[i]]
            rows.append(curr_row)
            
        # Process rows
        for r_idx, row in enumerate(rows):
            # Sort left-to-right (A, B, C, D)
            row = sorted(row, key=lambda b: b[0])
            
            # Find filled
            best_opt = -1
            max_p = 0
            for opt_idx, (bx, by, bw, bh, bc) in enumerate(row):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [bc], -1, 255, -1)
                masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(masked)
                
                if total > max_p and total > (bw*bh)*0.5:
                    max_p = total
                    best_opt = opt_idx
            
            # Map Row Index to Question Number?
            # We must be careful here. Assuming robust detection:
            # The simplified logic assumes we found exactly the rows we expect.
            # In a real rigorous system we'd check coordinates.
            # For this MVP, we rely on the clean generation.
            
            # Calculate question number
            # We need to know how many questions were in previous columns?
            # Or assume standard distribution?
            # SECTIONS_CONFIG specificies questions_per_col.
            # Let's use a simple counter if we assume perfect detection.
            pass # We need a reliable mapping strategy.
            
            # Alternative: Detection is usually perfect on generated images.
            # We'll use a global counter approach if we process columns in order.
            
    # REWRITE: Robust Answer Extraction
    # We will return the raw detected `rows` and map them to questions outside.
    return cols

def evaluate_advanced_omr(image_path, answer_key_path="answer_key.json"):
    image = cv2.imread(image_path)
    if image is None:
        print("Error reading image")
        return

    # 1. Alignment
    markers = detect_alignment_markers(image)
    if markers is not None:
        print("Alignment markers found. Warping...")
        warped = four_point_transform(image, markers)
    else:
        print("Markers not found. Using original image.")
        warped = image
        
    # Resize for consistency
    warped = imutils.resize(warped, width=1000)
    
    # 2. Box Detection
    boxes = detect_red_boxes(warped)
    print(f"Detected {len(boxes)} red boxes.")
    
    # Filter/Sort boxes
    # Phone boxes are on the LEFT. Answer boxes are on the RIGHT.
    h, w = warped.shape[:2]
    mid_x = w // 2
    
    phone_boxes = []
    answer_boxes = []
    
    for (x, y, bw, bh, c) in boxes:
        center_x = x + bw//2
        if center_x < mid_x:
            phone_boxes.append((x, y, bw, bh, c))
        else:
            answer_boxes.append((x, y, bw, bh, c))
            
    # Sort top-to-bottom
    phone_boxes = sorted(phone_boxes, key=lambda b: b[1])
    answer_boxes = sorted(answer_boxes, key=lambda b: b[1])
    
    results = {}
    
    # 3. Process Phone Numbers
    phone_labels = ["Candidate Phone", "Parent Phone", "Alternate Phone"]
    for i, box in enumerate(phone_boxes):
        if i >= 3: break
        x, y, bw, bh, c = box
        roi = warped[y:y+bh, x:x+bw]
        number = process_phone_box(roi)
        # Verify length 10
        if len(number) > 10: number = number[:10]
        results[phone_labels[i]] = number
        print(f"{phone_labels[i]}: {number}")
        
    # 4. Process Answers
    all_answers = {}
    
    # We expect 3 answer sections
    for i, box in enumerate(answer_boxes):
        if i+1 not in ANSWER_SECTIONS: break
        
        sec_id = i + 1
        config = ANSWER_SECTIONS[sec_id]
        
        x, y, bw, bh, c = box
        roi = warped[y:y+bh, x:x+bw]
        
        cols_data = process_section_box(roi, config)
        
        # Map to questions
        # config['questions'] is range.
        # We assume columns are detected in order and rows in order.
        
        q_idx = 0
        qs = config["questions"]
        
        for col in cols_data:
            # Sort rows by Y
            col = sorted(col, key=lambda b: b[1])
            
            # Group into rows (each row is one question)
            # A row is a cluster of bubbles with similar Y
            if not col: continue
            
            rows = []
            curr = [col[0]]
            for ball in col[1:]:
                if ball[1] - curr[0][1] < bh * 0.02: # Same row
                    curr.append(ball)
                else:
                    rows.append(curr)
                    curr = [ball]
            rows.append(curr)
            
            for row in rows:
                if q_idx >= len(qs): break
                q_num = qs[q_idx]
                
                # Find filled answer
                # Sort A-D
                row = sorted(row, key=lambda b: b[0])
                
                # Simple vote
                selected = -1
                max_area = 0
                
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                
                for opt_idx, bubble in enumerate(row):
                    bx, by, bbw, bbh, bc = bubble
                    mask = np.zeros(thresh_roi.shape, dtype="uint8")
                    # We need to shift contour or use boundingRect coords relative to ROI?
                    # The contour 'bc' is from findContours on ROI directly IN process_section_box, 
                    # so coordinates are relative to ROI. Correct.
                    
                    cv2.drawContours(mask, [bc], -1, 255, -1)
                    masked = cv2.bitwise_and(thresh_roi, thresh_roi, mask=mask)
                    filled = cv2.countNonZero(masked)
                    
                    if filled > max_area and filled > (bbw*bbh)*0.4:
                        max_area = filled
                        selected = opt_idx
                
                all_answers[q_num] = selected
                q_idx += 1
                
    # 5. Scoring
    try:
        with open(answer_key_path) as f:
            key_raw = json.load(f)
            # Parse key (it might be "1":"B")
            key_map = {}
            mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
            for k, v in key_raw.items():
                key_map[int(k)] = mapping.get(v, -1)
    except:
        key_map = {}
        
    score = 0
    total = 0
    
    # Save results
    results["answers"] = {}
    
    for q in range(1, 61):
        detected = all_answers.get(q, -1)
        expected = key_map.get(q, -1)
        
        is_cor = (detected == expected) if expected != -1 else False
        if expected != -1: # Only count if in key
            total += 1
            if is_cor: score += 1
            
        # Convert detected back to char
        char_map = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
        results["answers"][q] = char_map.get(detected, "?")
        
    results["score"] = f"{score}/{total}"
    results["percentage"] = (score/total)*100 if total else 0
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print(f"Candidate Phone: {results.get('Candidate Phone')}")
    print(f"Score: {results['score']} ({results['percentage']:.2f}%)")
    print("="*40)
    
    with open("evaluation_report.json", "w") as f:
         json.dump(results, f, indent=2)
         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--key", default="answer_key.json")
    args = parser.parse_args()
    evaluate_advanced_omr(args.image, args.key)
