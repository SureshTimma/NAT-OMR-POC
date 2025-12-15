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
    Detect the rectangular section boxes using edge detection
    Returns list of bounding boxes sorted top to bottom
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection to find borders
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Close to fill gaps
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Save debug mask
    # cv2.imwrite("debug_edges.jpg", edges)
    
    # Find contours
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def detect_timing_marks(thresh, img_w, img_h, num_cols):
    """
    Detect timing marks (small squares at the start of each row)
    Returns list of Y positions for each timing mark, grouped by column
    """
    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    timing_marks = []
    
    # Timing marks are small squares (aspect ratio ~1, smaller than bubbles)
    min_size = int(img_w * 0.008)
    max_size = int(img_w * 0.025)
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h) if h > 0 else 0
        area = cv2.contourArea(c)
        
        # Calculate fill ratio (timing marks are solid squares)
        bbox_area = w * h
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
        
        # Timing marks: small, square, solid, near left edge of columns
        if (min_size <= w <= max_size and 
            min_size <= h <= max_size and 
            0.8 <= ar <= 1.2 and
            fill_ratio > 0.7):  # Solid fill
            
            cx = x + w // 2
            cy = y + h // 2
            timing_marks.append((cx, cy, x, y, w, h))
    
    if not timing_marks:
        return None
    
    # Group timing marks by column (based on X position)
    timing_marks = sorted(timing_marks, key=lambda m: m[0])  # Sort by X
    
    col_width = img_w / num_cols
    columns = [[] for _ in range(num_cols)]
    
    for mark in timing_marks:
        cx = mark[0]
        col_idx = min(int(cx / col_width), num_cols - 1)
        columns[col_idx].append(mark[1])  # Store Y position
    
    # Sort each column by Y
    for i in range(num_cols):
        columns[i] = sorted(columns[i])
    
    return columns


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
    
    # if debug_prefix:
    #     cv2.imwrite(f"debug_thresh_{debug_prefix}.jpg", thresh)
    
    # if debug_prefix:
    #     cv2.imwrite(f"{debug_prefix}_thresh.jpg", thresh)
    
    img_h, img_w = section_img.shape[:2]
    
    # Detect timing marks for precise row positioning
    timing_mark_rows = detect_timing_marks(thresh, img_w, img_h, num_cols)
    
    # Timing mark detection can be unreliable on synthetic images with perfect alignment
    # Fallback row detection (grouping by Y) works better here.
    use_timing_marks = False
    if timing_mark_rows:
         total_marks = sum(len(col) for col in timing_mark_rows)
         print(f"  Found {total_marks} timing marks (ignored)")
    
    # Find ALL contours (not just external) to get bubbles inside section
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    bubble_cnts = []
    
    # Calculate expected bubble size based on section dimensions
    # Bubbles are typically about 1.5-5% of section width
    min_bubble_size = int(img_w * 0.015)
    max_bubble_size = int(img_w * 0.055)
    
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
        # - Circularity > 0.4 (reasonably round)
        # - Not touching edges
        if (min_bubble_size <= w <= max_bubble_size and 
            min_bubble_size <= h <= max_bubble_size and 
            0.7 <= ar <= 1.3 and
            circularity > 0.4):
            
            # Exclude edge artifacts (bubbles too close to section boundary)
            # Use smaller margin for bottom/right edges to capture last row bubbles
            margin_top = 10
            margin_left = 10
            margin_right = 5
            margin_bottom = 3  # Smaller margin at bottom to capture last row
            
            if (x > margin_left and y > margin_top and 
                (x + w) < (img_w - margin_right) and 
                (y + h) < (img_h - margin_bottom)):
                bubble_cnts.append(c)
    
    print(f"  Found {len(bubble_cnts)} bubble candidates")
    
    if not bubble_cnts:
        print(f"  Warning: No bubbles detected in section")
        return {}
    
    # Debug: Draw detected bubbles
    if debug_prefix:
        debug_bubbles = section_img.copy()
        cv2.drawContours(debug_bubbles, bubble_cnts, -1, (0, 255, 0), 2)
        # cv2.imwrite(f"{debug_prefix}_bubbles.jpg", debug_bubbles)
    
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
    # Sort all bubble center X positions
    all_xs = sorted([b[4] for b in bubble_data])
    
    # Find gaps between consecutive X positions
    gaps = []
    for i in range(1, len(all_xs)):
        gap = all_xs[i] - all_xs[i-1]
        gaps.append((gap, all_xs[i-1], all_xs[i]))
    
    # Sort gaps by size (largest first) and take top (num_cols - 1) gaps
    # These represent the boundaries between question columns
    gaps.sort(reverse=True)
    
    # Get the column boundaries (the midpoints of the largest gaps)
    col_boundaries = []
    for i in range(min(num_cols - 1, len(gaps))):
        gap_size, left_x, right_x = gaps[i]
        # Only use gaps that are significant (> 5% of image width)
        if gap_size > img_w * 0.05:
            midpoint = (left_x + right_x) / 2
            col_boundaries.append(midpoint)
    
    col_boundaries.sort()
    
    # Assign bubbles to question columns based on boundaries
    columns = [[] for _ in range(num_cols)]
    for data in bubble_data:
        cx = data[4]
        col_idx = 0
        for boundary in col_boundaries:
            if cx > boundary:
                col_idx += 1
            else:
                break
        if col_idx >= num_cols:
            col_idx = num_cols - 1
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
        
        # Use timing marks if available for precise row detection
        if use_timing_marks and timing_mark_rows and col_idx < len(timing_mark_rows) and timing_mark_rows[col_idx]:
            # Use timing mark Y positions to group bubbles into rows
            timing_ys = timing_mark_rows[col_idx]
            row_tolerance = img_h * 0.025  # 2.5% tolerance for matching
            
            rows = []
            for ty in timing_ys:
                row = []
                for bubble in col_bubbles:
                    bubble_cy = bubble[5]
                    if abs(bubble_cy - ty) < row_tolerance:
                        row.append(bubble)
                if row:
                    rows.append(row)
        else:
            # Fallback: Group by Y coordinate (each question's bubbles are on same row)
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
            
            # Remove duplicate contours at similar X positions (keep the larger one)
            # This handles cases where both outer ring and inner content are detected
            deduplicated_row = []
            min_x_gap = img_w * 0.02  # Minimum 2% width gap between distinct bubbles
            
            for bubble in row:
                if not deduplicated_row:
                    deduplicated_row.append(bubble)
                else:
                    last_bubble = deduplicated_row[-1]
                    if bubble[4] - last_bubble[4] < min_x_gap:
                        # Same position - keep the one with larger area
                        last_area = last_bubble[2] * last_bubble[3]  # w * h
                        curr_area = bubble[2] * bubble[3]
                        if curr_area > last_area:
                            deduplicated_row[-1] = bubble
                    else:
                        deduplicated_row.append(bubble)
            
            row = deduplicated_row
            
            # Only take the first num_options bubbles (in case extras were detected)
            row = row[:num_options]
            
            if q_idx >= len(questions):
                break
            
            q_num = questions[q_idx]
            
            # Find which bubbles are filled
            # Use adaptive thresholding based on pixel counts in the row
            pixel_counts = []
            for opt_idx, bubble in enumerate(row):
                 if opt_idx >= num_options: break
                 contour = bubble[6]
                 mask = np.zeros(thresh.shape, dtype="uint8")
                 cv2.drawContours(mask, [contour], -1, 255, -1)
                 # Count non-zero pixels in the thresholded image (bubbles are white in thresh)
                 masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                 total = cv2.countNonZero(masked)
                 pixel_counts.append(total)
            
            if not pixel_counts:
                answers[q_num] = {'filled_indices': [], 'bubbles': []}
                q_idx += 1
                continue

            # Adaptive threshold logic
            # Bubbles are either filled (high pixels) or empty (low pixels)
            # We can use a relative threshold. 
            # If a bubble has > 50% of its area filled? Or relative to max/min in row.
            
            # Simple approach: Threshold = Min + (Max - Min) * 0.5
            # If variation is small (all empty), threshold might be too sensitive.
            # Add a minimum absolute filled area requirement (e.g. 30% of bubble area)
            
            min_val = min(pixel_counts)
            max_val = max(pixel_counts)
            avg_area = sum(b[2]*b[3] for b in row[:len(pixel_counts)]) / len(pixel_counts)
            
            # Absolute threshold: at least 40% of standard bubble area must be filled
            # (Bubbles in 'thresh' are white where filled)
            abs_threshold = avg_area * 0.4 
            
            # Relative threshold to distinguish filled vs empty circles (borders usually give some pixels)
            rel_threshold = min_val + (max_val - min_val) * 0.5
            
            final_threshold = max(abs_threshold, rel_threshold)
            
            filled_indices = []
            for i, count in enumerate(pixel_counts):
                if count > final_threshold:
                    filled_indices.append(i)
            
            answers[q_num] = {
                'filled_indices': filled_indices,
                'bubbles': row[:len(pixel_counts)] # Store bubble data for drawing
            }
            q_idx += 1
    
    print(f"  Detected answers for {len(answers)} questions")
    return answers


def evaluate_omr_sections(image_path, answer_key_path="answer_key.json", output_dir="."):
    """
    Main function to evaluate OMR sheet with section detection
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
    
    # Load Answer Key
    try:
        with open(answer_key_path, "r") as f:
            answer_key = json.load(f)
            # Convert keys to integers
            answer_key = {int(k): v for k, v in answer_key.items()}
            # Convert answers to indices (A=0, B=1, etc)
            opt_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            answer_key = {k: opt_map.get(v, -1) for k, v in answer_key.items()}
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
    # cv2.imwrite("debug_sections.jpg", debug_img)
    # print("Saved debug_sections.jpg")
    
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
        
        # cv2.imwrite(f"debug_section_{section_num}.jpg", section_img)
        
        # Process section
        answers_data = process_section(section_img, config, f"debug_section_{section_num}")
        
        # Calculate section score and draw visualization
        section_correct = 0
        total = len(config['questions'])
        
        for q_num, q_data in answers_data.items():
            filled_indices = q_data['filled_indices']
            bubbles = q_data['bubbles'] # List of (x, y, w, h, cx, cy, contour) relative to section
            
            expected_idx = answer_key.get(q_num) # Answer key is 1-indexed now
            
            # Determine correctness
            is_correct = False
            if len(filled_indices) == 1 and filled_indices[0] == expected_idx:
                is_correct = True
                section_correct += 1
            
            # Draw visualization on debug_img (Global coordinates)
            # Global offset is (x, y) (from section_boxes loop: x, y, w, h, _)
            # Need to account for section padding 'pad' used when cropping
            current_pad = 5 # Matches the pad=5 used above
            
            # Helper to draw circle in global coords
            def draw_global_circle(local_bubble, color, thickness=2):
                 bx, by, bw, bh = local_bubble[0], local_bubble[1], local_bubble[2], local_bubble[3]
                 global_x = x + current_pad + bx + bw//2
                 global_y = y + current_pad + by + bh//2
                 radius = int(max(bw, bh) // 2 * 1.2) # Slightly larger than bubble
                 cv2.circle(debug_img, (global_x, global_y), radius, color, thickness)

            if is_correct:
                # Correct Answer 1: Circle with GREEN
                if expected_idx is not None and expected_idx < len(bubbles):
                     draw_global_circle(bubbles[expected_idx], (0, 255, 0), 3) # Green
            else:
                # Wrong/Unfilled/Multi: Score 0
                
                # "Wrong answers should be circle with red" -> All filled options get RED
                for idx in filled_indices:
                    if idx < len(bubbles):
                        draw_global_circle(bubbles[idx], (0, 0, 255), 3) # Red
                
                # "Correct answer should be circle with blue"
                if expected_idx is not None and expected_idx < len(bubbles):
                    # Draw Blue. If it was also filled (multi-fill case containing correct), 
                    # this will draw Blue over Red (or we can use different radius/thickness)
                    # User request: "correct answer should be blue"
                     draw_global_circle(bubbles[expected_idx], (255, 0, 0), 2) # Blue
            
            # Store simple result for JSON output (join with | if multiple)
            answer_letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
            detected_chars = [answer_letters.get(idx, '?') for idx in filled_indices]
            all_answers[q_num] = "".join(detected_chars) if detected_chars else ""

        section_score = (section_correct / total) * 100 if total > 0 else 0
        section_results.append({
            "section": section_num,
            "name": config['name'],
            "correct": section_correct,
            "total": total,
            "score": section_score
        })
        
        print(f"Section Score: {section_score:.1f}% ({section_correct}/{total})")
    
    # Save the visualized result
    output_image_path = os.path.join(output_dir, "debug_bubbles.jpg")
    cv2.imwrite(output_image_path, debug_img)
    print(f"Saved {output_image_path}")
    
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
    # detected_output is already populated in the loop
    detected_output = {str(q): val for q, val in sorted(all_answers.items())}
    
    with open(os.path.join(output_dir, "detected_answers.json"), 'w') as f:
        json.dump(detected_output, f, indent=2)
    print("\nSaved detected_answers.json")
    
    return all_answers, section_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMR Evaluation with Section Detection")
    parser.add_argument("--image", required=True, help="Path to the OMR image")
    parser.add_argument("--key", default="answer_key.json", help="Path to answer key JSON")
    args = parser.parse_args()
    
    evaluate_omr_sections(args.image, args.key)

