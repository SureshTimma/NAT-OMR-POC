import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import fitz  # PyMuPDF
import math

# Import from generate_new_updates
from generate_new_updates import (
    SECTIONS_CONFIG, 
    BORDER_COLOR, 
    TEXT_COLOR,
    draw_alignment_markers,
    draw_header,
    draw_phone_column
)

def draw_section_box_filled(c, x, y, width, height, config, answers):
    """Draw a question section box with filled answers"""
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1.5)
    c.rect(x, y, width, height, stroke=1, fill=0)
    
    # Section Name
    c.setFillColor(TEXT_COLOR)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + width/2, y + height - 14, config["name"])
    
    options = config["options"]
    num_cols = config["columns"]
    col_width = width / num_cols
    
    bubble_radius = 6.5
    bubble_spacing = 18
    q_num_width = 18
    
    # Headers A B C D
    header_y = y + height - 25
    c.setFont("Helvetica-Bold", 8)
    
    for col in range(num_cols):
        col_x = x + (col * col_width)
        start_x = col_x + q_num_width + 8
        for i, opt in enumerate(options):
            c.drawCentredString(start_x + i*bubble_spacing, header_y, opt)
            
    # Questions
    c.setFont("Helvetica", 9)
    start_q = config["start_q"]
    q_y_start = header_y - 15
    row_h = 16.5
    
    q_counter = start_q
    cols_q_counts = config["questions_per_col"]
    
    for col in range(num_cols):
        col_x = x + (col * col_width)
        num_q = cols_q_counts[col]
        
        for r in range(num_q):
            qy = q_y_start - r*row_h
            
            # Number
            c.drawString(col_x + 5, qy - 3, str(q_counter))
            
            # Bubbles
            start_x = col_x + q_num_width + 8
            student_answer = answers.get(str(q_counter), None)
            
            for i, opt in enumerate(options):
                bx = start_x + i*bubble_spacing
                c.setStrokeColor(BORDER_COLOR)
                c.setLineWidth(0.6)
                
                # Fill if this is the student's answer
                if student_answer == opt:
                    c.circle(bx, qy, bubble_radius, stroke=1, fill=1)
                else:
                    c.circle(bx, qy, bubble_radius, stroke=1, fill=0)
                
                c.setFillColor(TEXT_COLOR)
                c.setFont("Helvetica", 6)
                c.drawCentredString(bx, qy - 2, opt)
                c.setFont("Helvetica", 9)
                
            q_counter += 1

def generate_filled_new_updates_omr():
    """Generate a filled OMR sheet with all A answers"""
    filename = "omr_sheet_new_updates_filled.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4
    
    margin = 30
    content_w = w - 2*margin
    
    # 1. Alignment Markers
    draw_alignment_markers(c, w, h)
    
    # 2. Header Area
    header_h = 120
    header_y = h - margin - header_h - 10
    draw_header(c, margin, header_y, content_w, header_h)
    
    # 3. Body Layout
    body_y_top = header_y - 10
    
    # --- PHONE NUMBERS (Row 1) ---
    phone_gap = 20
    phone_width = (content_w - phone_gap) / 2
    phone_height = 150
    
    draw_phone_column(c, margin, body_y_top - phone_height, phone_width, phone_height, "Phone Number (WhatsApp No.)")
    draw_phone_column(c, margin + phone_width + phone_gap, body_y_top - phone_height, phone_width, phone_height, "Parent / Guardian Phone No.")
    
    current_y = body_y_top - phone_height - 15
    
    # --- SECTIONS (Vertical Stack) ---
    # Create answers dict with all A's
    answers = {}
    for i in range(1, 61):  # Questions 1-60
        answers[str(i)] = "A"
    
    section_gap = 10
    
    for i, sec in enumerate(SECTIONS_CONFIG):
        max_rows = max(sec["questions_per_col"])
        sec_h = 32 + (max_rows * 16.5) + 5
        
        draw_section_box_filled(c, margin, current_y - sec_h, content_w, sec_h, sec, answers)
        current_y -= (sec_h + section_gap)
    
    # --- SIGNATURE (Bottom) ---
    sig_h = 40
    sig_y = margin + 10
    
    # Signature Box
    c.setStrokeColor(colors.black) 
    c.rect(margin, sig_y, content_w, sig_h)
    
    c.setFont("Helvetica", 8)
    c.drawString(margin + 10, sig_y + 25, "By signing, I affirm that information provided is true.")
    c.drawString(margin + 10, sig_y + 12, "I understand that attempting this test does not guarantee admission.")
    
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(margin + content_w - 20, sig_y + 15, "Signature & Date: __________________________")
    
    c.save()
    print(f"Generated Filled PDF: {filename}")
    
    # Convert to JPG
    try:
        doc = fitz.open(filename)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        jpg_filename = filename.replace(".pdf", ".jpg")
        pix.save(jpg_filename)
        print(f"Generated Filled Image: {jpg_filename}")
    except Exception as e:
        print(f"Error converting: {e}")

if __name__ == "__main__":
    generate_filled_new_updates_omr()
