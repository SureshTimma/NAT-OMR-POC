import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours as imutils_contours
import argparse

import json

def evaluate_omr(image_path, answer_key_path="answer_key.json", num_questions=60):
    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    # Load Answer Key
    try:
        with open(answer_key_path, 'r') as f:
            loaded_key = json.load(f)
            # Map 'A', 'B', 'C', 'D' to 0, 1, 2, 3
            mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            ANSWER_KEY = {int(k): mapping.get(v, 0) for k, v in loaded_key.items()}
    except Exception as e:
        print(f"Error loading answer key: {e}")
        return

    # 2. Preprocessing
    # Resize for consistent processing width
    width = 800
    ratio = width / float(image.shape[1])
    resized = cv2.resize(image, (width, int(image.shape[0] * ratio)))
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 3. Find Document Corners (Alignment Markers)
    # We look for 4 circular contours that form a rectangle
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    doc_cnts = []
    
    # Filter contours
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Our markers are circles, so approx might have more than 4 points, 
        # OR we can look for specific area/shape factor.
        # But wait, we drew solid circles.
        # Let's try a different approach: Thresholding to find black blobs
        pass

    # Alternative: Thresholding to find the black markers
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    markers = []
    for c in cnts:
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        
    # Filter for squares
        # Use approxPolyDP to verify it has ~4 corners
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Square should have 4 corners (or close to it), reasonable area, and AR ~1
        # Relaxed approx constraint (4-8) because scanned corners can be noisy
        if 100 < area < 3000 and 0.8 <= ar <= 1.2 and 4 <= len(approx) <= 8:
            markers.append(c)

    # Sort markers by area (descending) to get the most prominent ones
    markers = sorted(markers, key=cv2.contourArea, reverse=True)[:4]
    
    # DEBUG: Save image with detected markers
    debug_markers = resized.copy()
    cv2.drawContours(debug_markers, markers, -1, (0, 255, 0), 2)
    cv2.imwrite("debug_markers.jpg", debug_markers)

    if len(markers) < 4:
        print("Error: Could not find 4 alignment markers. Found:", len(markers))
        # Debug: Show threshold
        cv2.imwrite("debug_thresh.jpg", thresh)
        return

    # Sort markers to find Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # We can use the centers
    centers = []
    for c in markers:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
        else:
             pass
    
    if len(centers) != 4:
        print(f"Error: Expected 4 marker centers, got {len(centers)}")
        return

    # Sort centers to form 4 points
    pts = np.array(centers, dtype="float32")
    
    # Sort: Top points (lowest Y), Bottom points (highest Y)
    # Then sorting Left/Right
    from imutils.perspective import order_points
    rect = order_points(pts)
    
    # 4. Perspective Transform
    warped = four_point_transform(gray, rect)
    warped_color = four_point_transform(resized, rect)
    cv2.imwrite("debug_warped.jpg", warped_color) # Save warped logic
    
    # 5. Bubble Detection in Warped Image
    # Resizing warped image to a known standard improves reliability
    # Let's say we warp to a standard A4-ish ratio
    # But four_point_transform keeps original scale relative to corners
    
    # OTSU Threshold to find bubbles
    thresh_warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh_warped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    question_cnts = []
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)
        
        # Typical bubble size filter
        # Tune these if detection fails
        if w >= 15 and h >= 15 and ar >= 0.8 and ar <= 1.2:
            # Filter out edge artifacts (markers might show up at corners)
            h_img, w_img = thresh_warped.shape
            if x < 5 or y < 5 or (x + w) > (w_img - 5) or (y + h) > (h_img - 5):
                continue
            question_cnts.append(c)

    # Sort bubbles: Top-to-bottom
    # Since we have 2 columns now, we must sort by columns first.
    # Simple heuristic: Split contours into Left and Right groups based on X coordinate.
    
    question_cnts = imutils_contours.sort_contours(question_cnts, method="top-to-bottom")[0] # Initial sort to handle random order
    
    # Calculate average X to find split point
    xs = [cv2.boundingRect(c)[0] for c in question_cnts]
    if not xs:
        print("No bubbles found.")
        return
        
    avg_x = sum(xs) / len(xs)
    
    left_cnts = []
    right_cnts = []
    
    for c in question_cnts:
        if cv2.boundingRect(c)[0] < avg_x:
            left_cnts.append(c)
        else:
            right_cnts.append(c)
            
    # Sort each column top-to-bottom
    if left_cnts:
        left_cnts = imutils_contours.sort_contours(left_cnts, method="top-to-bottom")[0]
    if right_cnts:
        right_cnts = imutils_contours.sort_contours(right_cnts, method="top-to-bottom")[0]
        
    # Combine (Left Column first, then Right Column)
    question_cnts = list(left_cnts) + list(right_cnts)
    
    # Verify count
    expected_bubbles = num_questions * 4
    print(f"Detected {len(question_cnts)} bubbles. Expected ~{expected_bubbles}.")
    
    if len(question_cnts) != expected_bubbles:
        print("Warning: Detected bubble count mismatch. Check lighting/fill.")
        # We might continue if it's close, or fail. Strict for now:
        # Actually, let's try to proceed by batching. 
        # If we missed some, logic will break. 
        # For PoC, let's assume good input.
    
    # Grading
    # Loop over every row of 4 bubbles
    correct_count = 0
    results = {} # {q_num: (detected_idx, is_correct)}
    
    for (q, i) in enumerate(range(0, len(question_cnts), 4)):
        # Get the row of 4 bubbles
        cnts_row = imutils_contours.sort_contours(question_cnts[i:i+4], method="left-to-right")[0]
        
        if len(cnts_row) < 4:
            print(f"Warning: Row {q} has < 4 bubbles. Skipping.")
            continue
            
        bubbled = None
        max_pixels = 0
        
        # Check which bubble is filled
        for (j, c) in enumerate(cnts_row):
            mask = np.zeros(thresh_warped.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # Use the mask to count non-zero pixels in the thresholded image
            mask = cv2.bitwise_and(thresh_warped, thresh_warped, mask=mask)
            total = cv2.countNonZero(mask)
            
            # Keep track of most filled bubble
            # We can also set a threshold for "filled" to avoid noise
            if bubbled is None or total > max_pixels:
                max_pixels = total
                bubbled = j
                
        # Grading
        color = (0, 0, 255) # Red for wrong
        k = ANSWER_KEY.get(q) # q is 0-indexed here
        
        if k == bubbled:
            color = (0, 255, 0) # Green for correct
            correct_count += 1
            
        # Draw on debug image
        # Outline correct answer (Blue)
        if k is not None and k < 4:
             cv2.drawContours(warped_color, [cnts_row[k]], -1, (255, 0, 0), 2)
             
        if bubbled is not None:
             cv2.drawContours(warped_color, [cnts_row[bubbled]], -1, color, 3) # Outline filled answer (Green/Red)

    score = (correct_count / num_questions) * 100
    print(f"SCORE: {score:.2f}% ({correct_count}/{num_questions})")
    
    cv2.imwrite("result_debug.jpg", warped_color)
    print("Saved result_debug.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the OMR image")
    parser.add_argument("--key", default="answer_key.json", help="Path to the answer key JSON file")
    args = parser.parse_args()
    
    evaluate_omr(args.image, args.key)

