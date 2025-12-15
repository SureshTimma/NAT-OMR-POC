
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import fitz  # PyMuPDF
import math

SECTIONS_CONFIG = [
    {
        "name": "Section: 1 (Pyschometric)",
        "start_q": 1,
        "end_q": 25,
        "options": ["A", "B", "C", "D"],
        "columns": 3,
        "questions_per_col": [9, 8, 8]  # Q1-9, Q10-17, Q18-25
    },
    {
        "name": "Section: 2 (Aptitude)",
        "start_q": 26,
        "end_q": 43,
        "options": ["A", "B", "C", "D"],
        "columns": 3,
        "questions_per_col": [6, 6, 6]  # Q26-31, Q32-37, Q38-43
    },
    {
        "name": "Section: 3 (Math)",
        "start_q": 44,
        "end_q": 60,
        "options": ["A", "B", "C", "D"],
        "columns": 3,
        "questions_per_col": [6, 6, 5]  # Q44-49, Q50-55, Q56-60
    }
]

BORDER_COLOR = colors.black
TEXT_COLOR = colors.black

def draw_filled_omr(target_option, filename):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    margin_left = 40
    margin_right = 40
    margin_top = 60
    margin_bottom = 40
    
    available_width = width - margin_left - margin_right
    
    # Title
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2, height - 20, f"Bubble Your Answers (Filled {target_option})")
    
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
    max_rows_per_section = []
    for sec in SECTIONS_CONFIG:
        max_rows = max(sec["questions_per_col"])
        max_rows_per_section.append(max_rows)
    
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
        
        for col in range(num_cols):
            col_x = box_x + (col * col_width)
            # options_start_x = col_x + q_num_width + 15
            options_start_x = col_x + q_num_width + 20  # Same offset as bubbles
            for opt_idx, opt in enumerate(options):
                opt_x = options_start_x + (opt_idx * bubble_spacing)
                c.drawCentredString(opt_x, header_y, opt)
        
        c.setFont("Helvetica", 10)
        q_counter = section["start_q"]
        bubble_y_start = header_y - 20
        
        for col in range(num_cols):
            col_x = box_x + (col * col_width)
            
            # Calculate starting X for options in this column (aligned with bubbles)
            # options_start_x = col_x + q_num_width + 20  # Same offset as bubbles
            
            # Draw option letters as header
           
            # Timing mark parameters
            timing_mark_size = 6  # Size of timing mark square
            timing_mark_offset = 3  # Offset from left edge of column
            
            num_questions_in_col = questions_per_col[col]
            for row in range(num_questions_in_col):
                q_y = bubble_y_start - (row * row_height)
                
                # ===== TIMING MARK (black square at start of each row) =====
                c.setFillColor(colors.black)
                c.rect(col_x + timing_mark_offset, 
                       q_y - timing_mark_size/2, 
                       timing_mark_size, 
                       timing_mark_size, 
                       stroke=0, fill=1)
                
                # Question number (shifted right to make room for timing mark)
                c.setFillColor(TEXT_COLOR)
                q_text = f"{q_counter}"
                # c.drawString(col_x + 10, q_y - 3, q_text)
                c.drawString(col_x + timing_mark_offset + timing_mark_size + 5, q_y - 3, q_text)
                
                # Bubbles
                options_start_x = col_x + q_num_width + 20  # Shifted right for timing mark
                # options_start_x = col_x + q_num_width + 15
                
                for opt_idx, opt in enumerate(options):
                    bubble_x = options_start_x + (opt_idx * bubble_spacing)
                    c.setStrokeColor(BORDER_COLOR)
                    c.setLineWidth(0.8)
                    
                    # FILL LOGIC
                    is_filled = 1 if opt == target_option else 0
                    c.circle(bubble_x, q_y, bubble_radius, stroke=1, fill=is_filled)
                    
                    if not is_filled:
                        c.setFillColor(TEXT_COLOR)
                        c.setFont("Helvetica", 7)
                        c.drawCentredString(bubble_x, q_y - 2.5, opt)
                        c.setFont("Helvetica", 10)
                
                q_counter += 1
        
        current_y = box_y - section_gap
    
    c.save()
    print(f"Generated PDF: {filename}")
    convert_pdf_to_jpg(filename)
    try:
        os.remove(filename)
        print(f"Deleted PDF: {filename}")
    except OSError:
        pass

def convert_pdf_to_jpg(pdf_path, dpi=300):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        jpg_filename = pdf_path.replace(".pdf", ".jpg")
        pix.save(jpg_filename)
        print(f"Generated Image: {jpg_filename}")
        doc.close()
    except Exception as e:
        print(f"Error converting: {e}")

import os

if __name__ == "__main__":
    options_to_fill = ["A", "B", "C", "D"]
    
    for opt in options_to_fill:
        pdf_name = f"omr_sheet_filled_{opt.lower()}.pdf"
        draw_filled_omr(opt, pdf_name)
        
        # Cleanup PDF if desired (commented out to keep debug)
        # if os.path.exists(pdf_name):
        #     os.remove(pdf_name)
