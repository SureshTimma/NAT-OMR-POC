import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import fitz  # PyMuPDF
import os
import random
import json
import time
import shutil
import subprocess

# --- CONFIGURATION (Same as generate_filled.py) ---
SECTIONS_CONFIG = [
    {
        "name": "Section: 1 (Pyschometric)",
        "start_q": 1,
        "end_q": 25,
        "options": ["A", "B", "C", "D"],
        "columns": 3,
        "questions_per_col": [9, 8, 8]
    },
    {
        "name": "Section: 2 (Aptitude)",
        "start_q": 26,
        "end_q": 43,
        "options": ["A", "B", "C", "D"],
        "columns": 3,
        "questions_per_col": [6, 6, 6]
    },
    {
        "name": "Section: 3 (Math)",
        "start_q": 44,
        "end_q": 60,
        "options": ["A", "B", "C", "D"],
        "columns": 3,
        "questions_per_col": [6, 6, 5]
    }
]

BORDER_COLOR = colors.black
TEXT_COLOR = colors.black

def draw_complex_omr(user_answers, filename):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    margin_left = 40
    margin_right = 40
    margin_top = 60
    
    available_width = width - margin_left - margin_right
    
    # Title
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2, height - 20, "Bubble Your Answers (Complex Test)")
    
    # Markers
    marker_size = 15
    marker_margin = 25
    marker_top_y = height - 45
    
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)
    c.rect(marker_margin, marker_top_y - marker_size, marker_size, marker_size, stroke=0, fill=1)
    c.rect(width - marker_margin - marker_size, marker_top_y - marker_size, marker_size, marker_size, stroke=0, fill=1)
    c.rect(marker_margin, marker_margin, marker_size, marker_size, stroke=0, fill=1)
    c.rect(width - marker_margin - marker_size, marker_margin, marker_size, marker_size, stroke=0, fill=1)
    
    section_gap = 15
    max_rows_per_section = [max(s["questions_per_col"]) for s in SECTIONS_CONFIG]
    
    row_height = 22
    header_height = 30
    bubble_area_top_margin = 10
    
    section_heights = []
    for max_rows in max_rows_per_section:
        h = header_height + bubble_area_top_margin + (max_rows * row_height) + 15
        section_heights.append(h)
    
    current_y = height - margin_top
    
    for sec_idx, section in enumerate(SECTIONS_CONFIG):
        sec_height = section_heights[sec_idx]
        box_x = margin_left
        box_y = current_y - sec_height
        box_width = available_width
        box_height = sec_height
        
        c.setStrokeColor(BORDER_COLOR)
        c.setLineWidth(1.5)
        c.rect(box_x, box_y, box_width, box_height, stroke=1, fill=0)
        
        c.setFillColor(TEXT_COLOR)
        c.setFont("Helvetica-Bold", 11)
        title_y = current_y - 20
        c.drawCentredString(width / 2, title_y, section["name"])
        
        options = section["options"]
        num_cols = section["columns"]
        questions_per_col = section["questions_per_col"]
        col_width = box_width / num_cols
        
        bubble_radius = 8
        bubble_spacing = 23
        q_num_width = 25
        
        c.setFont("Helvetica-Bold", 9)
        header_y = title_y - 25
        
        # Header Options
        for col in range(num_cols):
            col_x = box_x + (col * col_width)
            options_start_x = col_x + q_num_width + 20
            for opt_idx, opt in enumerate(options):
                opt_x = options_start_x + (opt_idx * bubble_spacing)
                c.drawCentredString(opt_x, header_y, opt)
        
        c.setFont("Helvetica", 10)
        q_counter = section["start_q"]
        bubble_y_start = header_y - 20
        
        for col in range(num_cols):
            col_x = box_x + (col * col_width)
            
            # Timing mark
            timing_mark_size = 6
            timing_mark_offset = 3
            
            num_questions_in_col = questions_per_col[col]
            for row in range(num_questions_in_col):
                q_y = bubble_y_start - (row * row_height)
                
                # Timing mark
                c.setFillColor(colors.black)
                c.rect(col_x + timing_mark_offset, q_y - timing_mark_size/2, 
                       timing_mark_size, timing_mark_size, stroke=0, fill=1)
                
                # Q Number
                c.setFillColor(TEXT_COLOR)
                q_text = f"{q_counter}"
                c.drawString(col_x + timing_mark_offset + timing_mark_size + 5, q_y - 3, q_text)
                
                # Bubbles
                options_start_x = col_x + q_num_width + 20
                
                # Retrieve answers for this question
                filled_opts = user_answers.get(str(q_counter), [])
                
                for opt_idx, opt in enumerate(options):
                    bubble_x = options_start_x + (opt_idx * bubble_spacing)
                    c.setStrokeColor(BORDER_COLOR)
                    c.setLineWidth(0.8)
                    
                    is_filled = 1 if opt in filled_opts else 0
                    c.circle(bubble_x, q_y, bubble_radius, stroke=1, fill=is_filled)
                    
                    if not is_filled:
                        c.setFillColor(TEXT_COLOR)
                        c.setFont("Helvetica", 7)
                        c.drawCentredString(bubble_x, q_y - 2.5, opt)
                        c.setFont("Helvetica", 10)
                
                q_counter += 1
        
        current_y = box_y - section_gap
    
    c.save()
    convert_pdf_to_jpg(filename)
    try:
        os.remove(filename)
    except OSError:
        pass

def convert_pdf_to_jpg(pdf_path, dpi=300):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        jpg_filename = pdf_path.replace(".pdf", ".jpg")
        pix.save(jpg_filename)
        doc.close()
    except Exception as e:
        print(f"Error converting: {e}")

def run_complex_tests():
    # Load Answer Key
    with open("answer_key.json", "r") as f:
        answer_key = json.load(f)
        
    num_tests = 10
    timestamp = int(time.time())
    base_dir = os.path.join(os.getcwd(), "test")
    complex_test_dir = os.path.join(base_dir, f"complex_test_{timestamp}")
    
    if not os.path.exists(complex_test_dir):
        os.makedirs(complex_test_dir)
        
    print(f"Generating {num_tests} complex tests in {complex_test_dir}")
    
    options_pool = ["A", "B", "C", "D"]
    
    for i in range(1, num_tests + 1):
        run_dir = os.path.join(complex_test_dir, f"run_{i}")
        os.makedirs(run_dir)
        
        # Generate User Answers
        user_answers = {}
        stats = {"Correct": 0, "Wrong": 0, "Unfilled": 0, "Multi": 0}
        
        for q in range(1, 61):
            q_str = str(q)
            correct_opt = answer_key.get(q_str, "A")
            
            r = random.random()
            
            if r < 0.6: # 60% Correct
                user_answers[q_str] = [correct_opt]
                stats["Correct"] += 1
            elif r < 0.8: # 20% Wrong
                available = [o for o in options_pool if o != correct_opt]
                user_answers[q_str] = [random.choice(available)]
                stats["Wrong"] += 1
            elif r < 0.9: # 10% Unfilled
                user_answers[q_str] = []
                stats["Unfilled"] += 1
            else: # 10% Multi-filled
                # Pick 2 or 3 random options
                k = random.choice([2, 3])
                user_answers[q_str] = random.sample(options_pool, k)
                stats["Multi"] += 1
                
        # Draw OMR
        pdf_path = os.path.join(run_dir, "filled.pdf")
        draw_complex_omr(user_answers, pdf_path)
        
        # Copy Key
        shutil.copy("answer_key.json", os.path.join(run_dir, "expected_key.json"))
        
        # Evaluate
        jpg_path = os.path.join(run_dir, "filled.jpg")
        key_path = os.path.join(run_dir, "expected_key.json")
        
        # Use absolute path for proper execution from any depth
        evaluate_script = os.path.abspath("../../evaluate.py")
        cmd = ["python", evaluate_script, "--image", "filled.jpg", "--key", "expected_key.json"]
        result = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
        
        # Save Report
        with open(os.path.join(run_dir, "report.txt"), "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\nErrors:\n")
                f.write(result.stderr)
        
        print(f"Test {i}: {stats}")
        
    print("\nCompleted.")

if __name__ == "__main__":
    run_complex_tests()
