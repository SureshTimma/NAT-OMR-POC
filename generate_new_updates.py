import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import fitz  # PyMuPDF
import math

# Configuration
SECTIONS_CONFIG = [
    {
        "name": "Section: 1 (Pyschometric)",
        "start_q": 1,
        "end_q": 25,
        "options": ["A", "B", "C", "D"],
        "columns": 4,
        "questions_per_col": [7, 6, 6, 6]
    },
    {
        "name": "Section: 2 (Aptitude)",
        "start_q": 26,
        "end_q": 43,
        "options": ["A", "B", "C", "D"],
        "columns": 4,
        "questions_per_col": [5, 5, 4, 4]
    },
    {
        "name": "Section: 3 (Math)",
        "start_q": 44,
        "end_q": 60,
        "options": ["A", "B", "C", "D"],
        "columns": 4,
        "questions_per_col": [5, 4, 4, 4]
    }
]

# Colors
BORDER_COLOR = HexColor("#8B0000")
TEXT_COLOR = colors.black
LIGHT_GRAY = colors.lightgrey

def draw_alignment_markers(c, width, height, margin=25, size=15):
    """Draw square markers at 4 corners"""
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)
    # TL, TR, BL, BR
    c.rect(margin, height - margin - size, size, size, fill=1, stroke=0)
    c.rect(width - margin - size, height - margin - size, size, size, fill=1, stroke=0)
    c.rect(margin, margin, size, size, fill=1, stroke=0)
    c.rect(width - margin - size, margin, size, size, fill=1, stroke=0)

def draw_header(c, x, y, width, height):
    """Draw header with Logo, Name fields, and Instructions"""
    title_height = 45 
    content_height = height - title_height
    
    # ---------------------------------------------------------
    # 1. TOP SECTION: Logo/Text | Divider | Title
    # ---------------------------------------------------------
    
    divider_x = x + width * 0.45 
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1)
    c.line(divider_x, y + height - 10, divider_x, y + height - title_height + 10)
    
    # LEFT: Logo Placeholder & Text
    logo_size = 35
    logo_x = x
    logo_y = y + height - title_height + 5
    
    c.setFillColor(colors.HexColor("#7B1113"))
    c.rect(logo_x, logo_y, logo_size, logo_size, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(logo_x + 5, logo_y + 18, "NI")
    c.drawString(logo_x + 5, logo_y + 6, "AT")
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(logo_x + logo_size + 10, logo_y + 20, "NxtWave Institute of")
    c.drawString(logo_x + logo_size + 10, logo_y + 8, "Advanced Technologies")
    
    # RIGHT: NAT Title
    c.setFillColor(BORDER_COLOR)
    c.setFont("Helvetica-Bold", 16)
    title_center_x = divider_x + (width - (divider_x - x)) / 2
    c.drawCentredString(title_center_x, y + height - 30, "NAT (NxtWave Admission Test)")

    # ---------------------------------------------------------
    # 2. CONTENT SECTION (Below Title)
    # ---------------------------------------------------------
    content_y_start = y 
    content_y_top = y + content_height
    
    left_w = width * 0.60
    right_w = width * 0.40
    
    # --- LEFT PANEL: Names + Barcode ---
    c.setFont("Helvetica", 10)
    c.setFillColor(TEXT_COLOR)
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(0.5)
    
    field_start_y = content_y_top - 12
    
    box_w = 14
    box_h = 16
    gap = 2
    label_w = 50 
    
    # Row 1: First Name
    c.rect(x, field_start_y - 5, label_w, box_h)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawCentredString(x + label_w/2, field_start_y - 2, "First Name")
    
    c.setStrokeColor(BORDER_COLOR)
    fname_x = x + label_w + 5
    for i in range(15):
        c.rect(fname_x + i*(box_w+gap), field_start_y - 5, box_w, box_h)
        
    # Row 2: Surname
    field_start_y -= 23 
    c.setStrokeColor(BORDER_COLOR)
    c.rect(x, field_start_y - 5, label_w, box_h)
    c.setFillColor(colors.black)
    c.drawCentredString(x + label_w/2, field_start_y - 2, "Last Name")
    
    sname_x = x + label_w + 5
    for i in range(15):
        c.rect(sname_x + i*(box_w+gap), field_start_y - 5, box_w, box_h)
        
    # Row 3: Barcode
    bar_y = field_start_y - 45 # Adjusted position
    c.setFillColor(colors.black)
    
    # Centre the barcode in the available left panel width
    # Left panel width is roughly 60% of content_w. 
    # Let's say we want it centered in that 'x' to 'divider_x' area (minus margin)
    # Actually 'left_w' is the width of this panel.
    
    barcode_width = 150
    barcode_start_x = x + (left_w - barcode_width) / 2
    
    current_bx = barcode_start_x
    import random
    random.seed(42)
    for _ in range(40):
        bw = random.choice([1, 2, 3])
        c.rect(current_bx, bar_y + 5, bw, 25, fill=1, stroke=0)
        current_bx += bw + random.choice([1, 2])
        
    c.setFont("Helvetica", 8)
    c.drawCentredString(x + left_w/2, bar_y - 5, "* B A R C O D E *")
    
    # --- RIGHT PANEL: Instructions ---
    inst_x = x + width - right_w
    
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1)
    c.rect(inst_x, content_y_start, right_w, content_height)
    
    c.line(inst_x, content_y_top - 18, inst_x + right_w, content_y_top - 18)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(inst_x + right_w/2, content_y_top - 12, "INSTRUCTIONS FOR FILLING THE SHEET")
    
    lines = [
        "1. Do not fold or crush this sheet.",
        "2. Use Blue/Black ball point pen only.",
        "3. Use of pencil is strictly prohibited.",
        "4. Darken circles completely.",
        "5. No cutting/erasing allowed."
    ]
    c.setFont("Helvetica", 7)
    ly = content_y_top - 28
    for line in lines:
        c.drawString(inst_x + 5, ly, line)
        ly -= 9
        
    # Visual Examples - Now beside instructions
    examples_y_start = content_y_top - 28
    examples_x_start = inst_x + 5
    
    c.setFont("Helvetica-Bold", 6)
    c.drawString(examples_x_start, examples_y_start - 55, "WRONG:")
    c.drawString(examples_x_start + right_w/2, examples_y_start - 55, "CORRECT:")
    
    # Wrong Examples
    r = 4
    gap_circ = 12
    y_circ = examples_y_start - 65
    
    cx = examples_x_start + 25
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.line(cx-r+2, y_circ-r+2, cx+r-2, y_circ+r-2)
    c.line(cx-r+2, y_circ+r-2, cx+r-2, y_circ-r+2)
    
    cx += gap_circ
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.circle(cx, y_circ, 1.5, stroke=0, fill=1)
    
    cx += gap_circ
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.line(cx-r, y_circ, cx+r, y_circ) 
    
    cx += gap_circ
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.line(cx-2, y_circ-2, cx, y_circ+2)
    c.line(cx, y_circ+2, cx+3, y_circ-4)

    # Correct Pattern
    cx = examples_x_start + right_w/2 + 30
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.circle(cx + gap_circ, y_circ, r, stroke=1, fill=0)
    c.circle(cx + gap_circ*2, y_circ, r, stroke=1, fill=0)
    c.circle(cx + gap_circ*3, y_circ, r, stroke=1, fill=1) 
        
    return y

def draw_phone_column(c, x, y, width, height, title, rows=10):
    """Draw a phone number/numeric field column"""
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1)
    c.rect(x, y, width, height, stroke=1, fill=0)
    
    # Title
    c.setFillColor(TEXT_COLOR)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + width/2, y + height - 12, title)
    
    # Write Boxes - Increased spacing
    box_w = 16
    box_h = 16
    gap = 5 # Increased gap between boxes
    total_boxes_width = (rows * box_w) + ((rows - 1) * gap)
    margin_x = (width - total_boxes_width) / 2
    
    write_y = y + height - 35
    
    for i in range(rows):
        bx = x + margin_x + i*(box_w + gap)
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.rect(bx, write_y, box_w, box_h)
    
    # Bubbles 0-9 - Sized to fit within boxes
    bubble_radius = 3.8 # Reduced to ensure bubbles stay inside boxes
    bubble_v_gap = 3.0
    
    bubble_start_y = write_y - 14
    
    c.setFont("Helvetica", 6)
    
    for i in range(rows): 
        col_center_x = x + margin_x + i*(box_w + gap) + box_w/2
        
        for digit in range(10): 
            by = bubble_start_y - digit * (bubble_radius*2 + bubble_v_gap)
            
            c.setStrokeColor(BORDER_COLOR)
            c.setLineWidth(0.8)
            c.circle(col_center_x, by, bubble_radius, stroke=1, fill=0)
            
            c.setFillColor(TEXT_COLOR)
            c.drawCentredString(col_center_x, by - 2.2, str(digit))

def draw_section_box(c, x, y, width, height, config):
    """Draw a question section box"""
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
    
    # Optimized Bubble Size
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
    row_h = 16.5 # Increased row height for better readabilty
    
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
            for i, opt in enumerate(options):
                bx = start_x + i*bubble_spacing
                c.setStrokeColor(BORDER_COLOR)
                c.setLineWidth(0.6)
                c.circle(bx, qy, bubble_radius, stroke=1, fill=0)
                
                c.setFillColor(TEXT_COLOR)
                c.setFont("Helvetica", 6)
                c.drawCentredString(bx, qy - 2, opt)
                c.setFont("Helvetica", 9)
                
            q_counter += 1

def generate_new_updates_omr():
    filename = "omr_sheet_new_updates.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4 # 595.27, 841.89
    
    margin = 30
    content_w = w - 2*margin
    
    # 1. Alignment Markers
    draw_alignment_markers(c, w, h)
    
    # 2. Header Area
    header_h = 120 # Reduced Height
    header_y = h - margin - header_h - 10
    draw_header(c, margin, header_y, content_w, header_h)
    
    # 3. Body Layout
    body_y_top = header_y - 10 # Gap
    
    # --- PHONE NUMBERS (Row 1) ---
    # Relaxed height since we have more space now
    phone_gap = 20
    phone_width = (content_w - phone_gap) / 2
    phone_height = 150 # Reduced slightly
    
    draw_phone_column(c, margin, body_y_top - phone_height, phone_width, phone_height, "Phone Number (WhatsApp No.)")
    draw_phone_column(c, margin + phone_width + phone_gap, body_y_top - phone_height, phone_width, phone_height, "Parent / Guardian Phone No.")
    
    current_y = body_y_top - phone_height - 15 # Gap
    
    # --- SECTIONS (Vertical Stack) ---
    # Layout:
    # [ Section 1 (Full Width) ]
    # [ Section 2 (Full Width) ]
    # [ Section 3 (Full Width) ]
    
    # Section Heights
    # Allow 30px for title/header + rows * 15.8 + padding
    
    section_gap = 10 
    
    for i, sec in enumerate(SECTIONS_CONFIG):
        max_rows = max(sec["questions_per_col"])
        sec_h = 32 + (max_rows * 16.5) + 5
        
        draw_section_box(c, margin, current_y - sec_h, content_w, sec_h, sec)
        current_y -= (sec_h + section_gap)
    
    # --- SIGNATURE (Bottom) ---
    sig_h = 40
    sig_y = margin + 10 # Bottom margin
    
    # Ensure we didn't overlap
    if current_y < (sig_y + sig_h):
        print("WARNING: Content Overlap likely! current_y:", current_y)
    
    # Signature Box
    c.setStrokeColor(colors.black) 
    c.rect(margin, sig_y, content_w, sig_h)
    
    c.setFont("Helvetica", 8)
    c.drawString(margin + 10, sig_y + 25, "By signing, I affirm that information provided is true.")
    c.drawString(margin + 10, sig_y + 12, "I understand that attempting this test does not guarantee admission.")
    
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(margin + content_w - 20, sig_y + 15, "Signature & Date: __________________________")
    
    c.save()
    print(f"Generated PDF: {filename}")
    
    # Convert to JPG
    try:
        doc = fitz.open(filename)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        jpg_filename = filename.replace(".pdf", ".jpg")
        pix.save(jpg_filename)
        print(f"Generated Image: {jpg_filename}")
    except Exception as e:
        print(f"Error converting: {e}")

if __name__ == "__main__":
    generate_new_updates_omr()
