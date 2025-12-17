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
            
    # Filter and Sort boxes
    # New Layout:
    # Header Boxes (Names, Instructions) -> Top
    # Phone Boxes -> Left Column
    # Answer Sections -> Right Column
    
    # We can filter by location.
    h_img, w_img = image.shape[:2]
    mid_x = w_img // 2
    
    phone_boxes = []
    answer_boxes = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000: # Filter noise
            x, y, w, h = cv2.boundingRect(c)
            # Aspect ratio check can help
            ar = w / float(h)
            
            # print(f"DEBUG: Found box. Area={area}, x={x}, y={y}, w={w}, h={h}, ar={ar:.2f}")

            # Header exclusion: If Y is very high (top of page), ignore
            # With new layout, header is top 25-30%. 
            # Phone/Answers are at bottom.
            if y < h_img * 0.25:
                # print("  -> Skipped as HEADER (y < 25%)")
                continue # Skip header elements
                
            # Classify Left vs Right
            center_bx = x + w // 2
            if center_bx < mid_x:
                # print("  -> Classified as PHONE (Left)")
                phone_boxes.append((x, y, w, h, c))
            else:
                # print("  -> Classified as ANSWER (Right)")
                answer_boxes.append((x, y, w, h, c))
            
    # Sort top-to-bottom
    phone_boxes = sorted(phone_boxes, key=lambda b: b[1])
    answer_boxes = sorted(answer_boxes, key=lambda b: b[1])
    
    # Return as combined list but in expected order for main loop processing?
    # Actually, main loop logic likely iterates blindly. 
    # Let's verify how main loop works. 
    # It probably needs to know which is which.
    # So let's return a dictionary or tuple.
    return phone_boxes, answer_boxes
    
    # Only returning tuple breaks existing signature.
    # Let's see how `evaluate_advanced.py` uses this. 
    # It calls `detect_red_boxes` and then expects a list. 
    # I should update the calling code too.

def save_debug_image(img, name):
    debug_dir = "tests/debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    path = os.path.join(debug_dir, f"{name}.jpg")
    cv2.imwrite(path, img)
    # print(f"Saved debug: {path}")

def process_phone_box(roi, debug_name=None):
    """Read a phone number box (10 columns, 0-9 bubbles)"""
    # Debug: Save original ROI
    if debug_name:
        save_debug_image(roi, f"{debug_name}_orig")

    # Crop out the header (Title + Write Boxes)
    # Header is approx 28% of height. Safe to crop 30%.
    bg_h_full, bg_w_full = roi.shape[:2]
    crop_y = int(bg_h_full * 0.30)
    roi_bubbles = roi[crop_y:, :]
    
    gray = cv2.cvtColor(roi_bubbles, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    bg_h, bg_w = roi_bubbles.shape[:2] # Upldate bg_h to bubble area height
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
            if bubbles[i][0] - current_col[0][0] < bg_w * 0.05: # Same column
                current_col.append(bubbles[i])
            else:
                columns.append(current_col)
                current_col = [bubbles[i]]
        columns.append(current_col)
        
    # ROBUST ROW DETECTION
    # Collect all bubbles and cluster by Y to find the 10 rows
    all_bubbles_by_y = sorted(bubbles, key=lambda b: b[1])
    rows = []
    if all_bubbles_by_y:
         current_row = [all_bubbles_by_y[0]]
         for i in range(1, len(all_bubbles_by_y)):
             if all_bubbles_by_y[i][1] - current_row[0][1] < bg_h * 0.05: # Same row (5% threshold)
                 current_row.append(all_bubbles_by_y[i])
             else:
                 rows.append(current_row)
                 current_row = [all_bubbles_by_y[i]]
         rows.append(current_row)
    
    # Sort rows top-to-bottom and assign digits 0-9
    rows = sorted(rows, key=lambda r: r[0][1])
    
    # Create a mapping of Y-center -> Digit
    row_centers = []
    for i, row in enumerate(rows):
        # Average Y of row
        avg_y = sum(b[1] + b[3]/2 for b in row) / len(row)
        row_centers.append((avg_y, i))
        
    print(f"DEBUG: Phone Box - Found {len(columns)} columns and {len(rows)} rows.")
    print(f"DEBUG: Row Centers: {[int(rc[0]) for rc in row_centers]}")
        
    # If we strictly have 10 rows, great. If not, we might need fallback.
    # But usually with clean generation we get 10.
    
    phone_number = ""
    
    for col in columns:
        filled_digit = -1
        max_pixels = 0
        
        for (bx, by, bw, bh, c) in col:
            # Check if filled
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            masked = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(masked)
            
            # Find which row this bubble belongs to
            cy = by + bh/2
            
            # Find closest row center
            best_row_idx = -1
            min_dist = float('inf')
            
            for (ry, r_idx) in row_centers:
                dist = abs(cy - ry)
                if dist < min_dist:
                    min_dist = dist
                    best_row_idx = r_idx
            
            # Map valid row index to digit (0-9)
            # If we detected exactly 10 rows, best_row_idx IS the digit.
            # If we detected != 10, we might need to map percentage?
            # Let's trust the row index if len(rows) == 10.
            
            digit = best_row_idx
            if len(rows) == 10:
                digit = best_row_idx
            else:
                 # Fallback: simple linear map if row detection failed count
                 digit = int(cy / (bg_h/10.0))
            
            digit = min(max(digit, 0), 9)

            if total > max_pixels and total > (bw*bh)*0.5:
                max_pixels = total
                filled_digit = digit
                
        if filled_digit != -1:
            phone_number += str(filled_digit)
        else:
            phone_number += "_"
            
    # Draw debug visual
    if debug_name:
        debug_img = roi.copy()
        
        # Draw detected rows
        for ry, r_idx in row_centers:
            cv2.line(debug_img, (0, int(ry)), (bg_w, int(ry)), (0, 255, 255), 1)
        
        for col in columns:
             for (bx, by, bw, bh, c) in col:
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(masked)
                
                color = (255, 0, 0) # Blue
                thickness = 1
                if total > (bw*bh)*0.5:
                     color = (0, 255, 0) # Green
                     thickness = 2
                
                cv2.drawContours(debug_img, [c], -1, color, thickness)
                
        save_debug_image(debug_img, f"{debug_name}_visual")

    return phone_number

def process_section_box(roi, config, debug_name=None):
    """Reuse existing logic to process answer sections"""
    # Debug: Save ROI (Step 1: Raw)
    if debug_name:
        save_debug_image(roi, f"{debug_name}_1_raw")

    # This is a simplified version of the logic from evaluate.py
    # Re-implementing essentially the same steps adapted for the ROI
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Debug: Save Threshold (Step 2: Thresh)
    if debug_name:
        save_debug_image(thresh, f"{debug_name}_2_thresh")

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Debug: Save ALL Contours (Step 3: All Contours)
    if debug_name:
        debug_contours_all = roi.copy()
        cv2.drawContours(debug_contours_all, cnts, -1, (0, 0, 255), 1) # Red for raw contours
        save_debug_image(debug_contours_all, f"{debug_name}_3_contours_all")

    h, w = roi.shape[:2]
    bubbles = []
    
    # Filter bubbles
    for c in cnts:
        (x, y, bw, bh) = cv2.boundingRect(c)
        ar = bw / float(bh)
        if 0.8 <= ar <= 1.2 and w*0.01 < bw < w*0.05:
            bubbles.append((x, y, bw, bh, c))
            
    # Debug: Save Filtered Bubbles (Step 4: Filtered)
    if debug_name:
        debug_bubbles_filtered = roi.copy()
        for b in bubbles:
             cv2.drawContours(debug_bubbles_filtered, [b[4]], -1, (0, 255, 0), 1) # Green for valid bubbles
        save_debug_image(debug_bubbles_filtered, f"{debug_name}_4_bubbles_filtered")

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
            
    # Draw logic inside function
    if debug_name:
        debug_img = roi.copy()
        
        # Step 5: Visual Grid / Logic
        debug_grid = roi.copy()
        
        for col_idx, col_bubbles in enumerate(cols):
            # Sort rows
            col_bubbles = sorted(col_bubbles, key=lambda b: b[1])
            
            # Group into questions
            rows = []
            if col_bubbles:
                curr_row = [col_bubbles[0]]
                for i in range(1, len(col_bubbles)):
                    if col_bubbles[i][1] - col_bubbles[i-1][1] < h*0.02: # Same row threshold
                        curr_row.append(col_bubbles[i])
                    else:
                        rows.append(curr_row)
                        curr_row = [col_bubbles[i]]
                rows.append(curr_row)
            
            for row in rows:
                # Find filled in this row
                max_p = 0
                best_idx = -1
                
                # Sort row left-to-right
                row = sorted(row, key=lambda b: b[0])
                
                for idx, (bx, by, bw, bh, bc) in enumerate(row):
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [bc], -1, 255, -1)
                    masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                    total = cv2.countNonZero(masked)
                    if total > max_p:
                        max_p = total
                        best_idx = idx
                
                for idx, (bx, by, bw, bh, bc) in enumerate(row):
                    color = (255, 0, 0) # Blue for empty
                    thickness = 1
                    
                    # Threshold for valid fill
                    # Re-calc pixel count (inefficient but safe)
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [bc], -1, 255, -1)
                    masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                    total = cv2.countNonZero(masked)
                    
                    # Visual check logic
                    is_filled = total > (bw*bh)*0.5
                    
                    if is_filled and idx == best_idx:
                        color = (0, 255, 0) # Green for filled
                        thickness = 2
                    
                    cv2.drawContours(debug_img, [bc], -1, color, thickness)
                    # cv2.rectangle(debug_img, (bx, by), (bx+bw, by+bh), color, thickness)

        save_debug_image(debug_img, f"{debug_name}_5_visual_final")

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
    # 2. Box Detection
    phone_boxes, answer_boxes = detect_red_boxes(warped)
    print(f"Detected {len(phone_boxes)} phone boxes and {len(answer_boxes)} answer boxes.")
    
    # Filter/Sort boxes - ALREADY DONE in detect_red_boxes
    # Phone boxes are on the LEFT. Answer boxes are on the RIGHT.
    # No further sorting needed as default order returned is sorted by Y.
    
    # DEBUG: Save Master Debug Image with all boxes
    debug_master = warped.copy()
    for i, (x, y, w, h, c) in enumerate(phone_boxes):
        cv2.rectangle(debug_master, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(debug_master, f"Phone {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
    for i, (x, y, w, h, c) in enumerate(answer_boxes):
        cv2.rectangle(debug_master, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(debug_master, f"Section {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    debug_dir = "tests/debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        
    cv2.imwrite(os.path.join(debug_dir, "master_debug.jpg"), debug_master)
    print(f"Saved master debug image to {os.path.join(debug_dir, 'master_debug.jpg')}")
    
    results = {}
    
    # 3. Process Phone Numbers
    phone_labels = ["Candidate Phone", "Parent Phone", "Alternate Phone"]
    for i, box in enumerate(phone_boxes):
        if i >= 3: break
        x, y, bw, bh, c = box
        roi = warped[y:y+bh, x:x+bw]
        number = process_phone_box(roi, debug_name=f"phone_{i}")
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
        
        # DEBUG: ONLY PROCESS SECTION 1
        if sec_id != 1:
            print(f"Skipping Section {sec_id} for debug focus.")
            continue
            
        config = ANSWER_SECTIONS[sec_id]
        
        x, y, bw, bh, c = box
        roi = warped[y:y+bh, x:x+bw]
        
        cols_data = process_section_box(roi, config, debug_name=f"answer_sec_{sec_id}")
        
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
         
    return results
         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--key", default="answer_key.json")
    parser.add_argument("-o", "--output", required=False, help="Path to save JSON report")
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_advanced_omr(args.image, args.key)
    
    # Save Report
    if args.output:
        output_path = args.output
    else:
        # Default to same dir as image
        base = os.path.splitext(args.image)[0]
        output_path = f"{base}_report.json"
        
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Report saved to {output_path}")
