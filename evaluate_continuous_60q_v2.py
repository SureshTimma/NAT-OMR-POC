import cv2
import numpy as np
import json
import os

# Configuration
INPUT_IMAGE = "omr_sheet_continuous_60q_filled.jpg"
ANSWER_KEY_FILE = "answer_key.json"
DEBUG_DIR = "debug_output_v10"

if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

def load_answer_key():
    with open(ANSWER_KEY_FILE, 'r') as f:
        return json.load(f)

def evaluate_omr():
    print(f"Processing {INPUT_IMAGE}...")
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"Error: Could not read {INPUT_IMAGE}")
        return
    
    debug_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Main Boxes (Phone x2, Question Grid) using Red Color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    box_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 5000:
            box_contours.append((x, y, w, h))
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    box_contours = sorted(box_contours, key=lambda b: b[1])
    
    img_h, img_w = img.shape[:2]
    phone_boxes = []
    question_box = None
    
    for b in box_contours:
        x, y, w, h = b
        if w > img_w * 0.8:
            question_box = b
        elif w > img_w * 0.3 and y < img_h * 0.6:
            phone_boxes.append(b)
            
    phone_boxes = sorted(phone_boxes, key=lambda b: b[0])
    
    # 2. Process Phone Numbers
    detected_phones = []
    if len(phone_boxes) >= 2:
        for i, box in enumerate(phone_boxes):
            val = process_phone_box(gray, box, debug_img, f"Phone{i+1}")
            detected_phones.append(val)
        # Ensure only 2 phones are kept (remove Instructions box)
        detected_phones = detected_phones[:2]
    else:
        detected_phones = ["ERR", "ERR"]
        print("Error: Could not find 2 phone boxes")

    # 3. Process Questions with Scoring Visualization
    answer_key = load_answer_key()
    detected_answers = {}
    if question_box:
        detected_answers = process_question_grid(gray, question_box, debug_img, "Questions", answer_key)
    else:
        print("Error: Could not find Question box")
        
    # 4. Save Visual Debug Image
    cv2.imwrite(f"{DEBUG_DIR}/visual_debug_scored.jpg", debug_img)
    print(f"Saved visual debug image to {DEBUG_DIR}/visual_debug_scored.jpg")
    
    # 5. Generate Report
    report_lines = []
    report_lines.append("="*40)
    report_lines.append("EVALUATION REPORT")
    report_lines.append("="*40)
    report_lines.append(f"Detected Phone 1: {detected_phones[0]}")
    report_lines.append(f"Detected Phone 2: {detected_phones[1]}")
    report_lines.append("-" * 40)
    
    correct_count = 0
    wrong_count = 0
    
    json_results = {}
    json_questions = {}
    
    for q_num in range(1, 61):
        q_str = str(q_num)
        correct_opt = answer_key.get(q_str, "")
        detected_opt = detected_answers.get(q_str, "N/A")
        
        status = "WRONG"
        if detected_opt == correct_opt:
            status = "CORRECT"
            correct_count += 1
        else:
            wrong_count += 1
            
        report_lines.append(f"Q{q_num}: Detected [{detected_opt}] | Correct [{correct_opt}] -> {status}")
        
        json_questions[q_str] = detected_opt
        json_results[q_str] = status

    report_lines.append("-" * 40)
    report_lines.append(f"Total Correct: {correct_count}/60")
    report_lines.append(f"Accuracy: {(correct_count/60)*100:.2f}%")
    report_lines.append("="*40)
    
    report_content = "\n".join(report_lines)
    print(report_content)
    
    with open("evaluation_report_v10.txt", "w") as f:
        f.write(report_content)
    print("Report saved to evaluation_report_v10.txt")
    
    # Save as JSON
    report_data = {
        "phone_numbers": detected_phones,
        "questions": json_questions,
        "results": json_results,
        "summary": {
            "total_correct": correct_count,
            "total_questions": 60,
            "accuracy": f"{(correct_count/60)*100:.2f}%"
        }
    }
    
    with open("evaluation_report.json", "w") as f:
        json.dump(report_data, f, indent=4)
    print("Report saved to evaluation_report.json")

def process_phone_box(gray, box, debug_img, name):
    x, y, w, h = box
    
    # Crop top 20% to remove Write Boxes (at 38pts, Cut at 44pts)
    crop_off = int(h * 0.20)
    roi_y = y + crop_off
    roi_h = h - crop_off
    roi = gray[roi_y:roi_y+roi_h, x:x+w]
    
    thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if 25 < bw < 100 and 20 < bh < 100:
             mask = np.zeros(thresh.shape, dtype="uint8")
             cv2.drawContours(mask, [c], -1, 255, -1)
             
             # CORRECTED: Count White Pixels (Filled Area) in Thresh
             px_count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
             
             if px_count > 800:
                bubbles.append((bx, by, bw, bh))
                cv2.rectangle(debug_img, (x+bx, roi_y+by), (x+bx+bw, roi_y+by+bh), (255, 0, 0), 1)

    roi_h, roi_w = roi.shape[:2]
    
    scale = h / 220.0
    grid_start_x = roi_w * 0.10
    grid_end_x = roi_w * 0.90
    col_step = (grid_end_x - grid_start_x) / 10
    
    cols = ["." for _ in range(10)]
    
    for b in bubbles:
        bx, by, bw, bh = b
        roi_cx = bx + bw/2
        roi_cy = by + bh/2
        
        c_idx = int((roi_cx - grid_start_x) / col_step)
        
        # Explicit Geometry: Bubble 0 Center @ 52pts, Step 17.4pts
        y_pts = (roi_cy + crop_off) / scale
        
        # FILTER: Ignore Write Boxes (above 46 pts)
        if y_pts < 46:
            continue
            
        d_idx = int(round((y_pts - 52) / 17.4))
        d_idx = max(0, min(9, d_idx))
        
        if 0 <= c_idx < 10:
             cols[c_idx] = str(d_idx)
                
    return "".join(cols)

def process_question_grid(gray, box, debug_img, name, answer_key):
    x, y, w, h = box
    crop_off = int(h * 0.12)
    roi_y = y + crop_off
    roi_h = h - crop_off
    roi = gray[roi_y:roi_y+roi_h, x:x+w]
    thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if 20 < bw < 100 and 20 < bh < 100:
            bubbles.append((bx, by, bw, bh))
            
    detected_list = {}
    for i in range(1, 61):
        detected_list[str(i)] = []
        
    if not bubbles:
        return {str(i): "N/A" for i in range(1, 61)}
        
    # Robust Row Step using Span
    c_ys = [b[1] + b[3]/2 for b in bubbles]
    min_y = min(c_ys)
    max_y = max(c_ys)
    grid_h = max_y - min_y
    row_step = grid_h / 14 if grid_h > 0 else 1
    
    roi_h, roi_w = roi.shape[:2]
    large_col_w = roi_w / 4
    options = ['A', 'B', 'C', 'D']
    
    for b in bubbles:
        bx, by, bw, bh = b
        cx = bx + bw/2
        cy = by + bh/2
        
        c_idx = int(cx / large_col_w)
        if not (0 <= c_idx < 4): continue
        
        rel_y = cy - min_y
        r_idx = int(round(rel_y / row_step))
        r_idx = max(0, min(14, r_idx))
            
        col_center_x = (c_idx + 0.5) * large_col_w
        px_scale = roi_w / 535.0
        ideal_A_offset = -16.5 * px_scale
        ideal_B_offset = 5.5 * px_scale
        ideal_C_offset = 27.5 * px_scale
        ideal_D_offset = 49.5 * px_scale
        
        offset = cx - col_center_x
        dists = [abs(offset - ideal_A_offset), abs(offset - ideal_B_offset),
                 abs(offset - ideal_C_offset), abs(offset - ideal_D_offset)]
        opt_idx = np.argmin(dists)
        
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.circle(mask, (int(bx + bw/2), int(by + bh/2)), int(bw/2 - 2), 255, -1)
        px_count = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        
        if px_count > 600:
            opt = options[opt_idx]
            q_num = 1 + (c_idx * 15) + r_idx
            detected_list[str(q_num)].append(opt)
            
            # Visual Scoring (Per Bubble)
            correct_opt = answer_key.get(str(q_num), "")
            # Green if this bubble is IN the correct answer (simple check)
            # But correct_opt is usually single "A".
            color = (0, 255, 0) if opt in correct_opt else (0, 0, 255)
            cv2.rectangle(debug_img, (x+bx, roi_y+by), (x+bx+bw, roi_y+by+bh), color, 2)

    # Finalize Answers (Sort and Join)
    detected_answers = {}
    for i in range(1, 61):
        lst = detected_list[str(i)]
        if not lst:
            detected_answers[str(i)] = "N/A"
        else:
            # Unique Sorted
            detected_answers[str(i)] = "".join(sorted(list(set(lst))))
            
            # Visual Scoring
            correct_opt = answer_key.get(str(q_num), "")
            color = (0, 255, 0) if opt == correct_opt else (0, 0, 255)
            cv2.rectangle(debug_img, (x+bx, roi_y+by), (x+bx+bw, roi_y+by+bh), color, 2)

    # Visual Scoring: Correct Answers
    for q_num in range(1, 61):
        q_str = str(q_num)
        detected = detected_answers.get(q_str, "N/A")
        correct = answer_key.get(q_str, "")
        
        if detected != correct and correct in options:
            col_idx = (q_num - 1) // 15
            row_idx = (q_num - 1) % 15
            opt_idx = options.index(correct)
            
            col_center_x = (col_idx + 0.5) * large_col_w
            px_scale = roi_w / 535.0
            
            ideal_offsets = [
                -16.5 * px_scale, 5.5 * px_scale, 27.5 * px_scale, 49.5 * px_scale
            ]
            
            target_cx = col_center_x + ideal_offsets[opt_idx]
            target_cy = min_y + (row_idx * row_step)
            
            cv2.circle(debug_img, (int(x + target_cx), int(roi_y + target_cy)), 10, (255, 0, 0), 2)

    return detected_answers

if __name__ == "__main__":
    evaluate_omr()
