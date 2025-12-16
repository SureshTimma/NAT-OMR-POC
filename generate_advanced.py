import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import fitz  # PyMuPDF
import math

# Reuse Section Configuration
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
        "options": ["A", "B", "C", "D", "E"], # NOTE: evaluate.py uses 4 options for sec 3 but valid options A-E in answer_letters. sticking to 4 as per user request "100% accuracy" config.
        # Actually generate.py had ["A","B","C","D"] for section 3. I will stick to that.
        "columns": 3,
        "questions_per_col": [6, 6, 5]
    }
]

# Update Section 3 options to match generate.py exactly
SECTIONS_CONFIG[2]["options"] = ["A", "B", "C", "D"]


# Colors
BORDER_COLOR = HexColor("#8B0000")  # Dark red/maroon
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
    # Define Layout Heights
    title_height = 50 # Increased for Logo area
    content_height = height - title_height
    
    # ---------------------------------------------------------
    # 1. TOP SECTION: Logo/Text | Divider | Title
    # ---------------------------------------------------------
    
    # Divider Line
    divider_x = x + width * 0.45 
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1)
    c.line(divider_x, y + height - 10, divider_x, y + height - title_height + 10)
    
    # LEFT: Logo Placeholder & Text
    # Simulate Logo: Red Shield with "NI AT"
    logo_size = 35
    logo_x = x
    logo_y = y + height - title_height + 8
    
    c.setFillColor(colors.HexColor("#7B1113")) # Dark Red like image
    c.path = c.beginPath()
    c.path.moveTo(logo_x, logo_y + logo_size)
    c.path.lineTo(logo_x + logo_size, logo_y + logo_size)
    c.path.lineTo(logo_x + logo_size/2, logo_y)
    c.path.close()
    # c.drawPath(c.path, fill=1, stroke=0) # Simple shield shape - actually let's just use a rect with text for now to be safe
    
    # Draw Logo Box
    c.rect(logo_x, logo_y, logo_size, logo_size, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(logo_x + 5, logo_y + 18, "NI")
    c.drawString(logo_x + 5, logo_y + 6, "AT")
    
    # Text next to logo
    c.setFillColor(colors.black) # Or dark grey
    c.setFont("Helvetica-Bold", 10)
    c.drawString(logo_x + logo_size + 10, logo_y + 20, "NxtWave Institute of")
    c.drawString(logo_x + logo_size + 10, logo_y + 8, "Advanced Technologies")
    
    # RIGHT: NAT Title
    c.setFillColor(BORDER_COLOR) # Dark Red
    c.setFont("Helvetica-Bold", 18)
    # Centered in the remaining space
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
    c.setStrokeColor(BORDER_COLOR) # Red borders for name boxes
    c.setLineWidth(0.5)
    
    field_start_y = content_y_top - 15
    
    box_w = 14
    box_h = 18
    gap = 2
    label_w = 50 # Width for "First Name" box
    
    # Row 1: First Name
    # Label Box
    c.rect(x, field_start_y - 5, label_w, box_h)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawCentredString(x + label_w/2, field_start_y - 2, "First Name")
    
    # Character Boxes
    c.setStrokeColor(BORDER_COLOR) # Red outlines
    fname_x = x + label_w + 5
    for i in range(15):
        c.rect(fname_x + i*(box_w+gap), field_start_y - 5, box_w, box_h)
        
    # Row 2: Surname
    field_start_y -= 25 # Move down
    # Label Box
    c.setStrokeColor(BORDER_COLOR)
    c.rect(x, field_start_y - 5, label_w, box_h)
    c.setFillColor(colors.black)
    c.drawCentredString(x + label_w/2, field_start_y - 2, "Last Name")
    
    # Character Boxes
    sname_x = x + label_w + 5
    for i in range(15):
        c.rect(sname_x + i*(box_w+gap), field_start_y - 5, box_w, box_h)
        
    # Row 3: Barcode
    # Large rectangle enrolling barcode? Image shows just barcode. 
    # Let's keep the placeholder but align it nicely.
    bar_y = field_start_y - 50
    bar_width = 200
    bar_height = 35
    # c.rect(x + 50, bar_y, bar_width, bar_height) # Optional box around barcode
    c.setFillColor(colors.black)
    # c.setFont("Free 3 of 9 Extended", 30) # Font not available
    # Since we might not have the font, we draw lines or keep text
    # Drawing simple vertical lines to look like barcode
    current_bx = x + 60
    import random
    random.seed(42)
    for _ in range(40):
        bw = random.choice([1, 2, 3])
        c.rect(current_bx, bar_y + 5, bw, 25, fill=1, stroke=0)
        current_bx += bw + random.choice([1, 2])
        
    c.setFont("Helvetica", 8)
    c.drawCentredString(x + 60 + 75, bar_y - 5, "* B A R C O D E *")
    
    # --- RIGHT PANEL: Instructions ---
    inst_x = x + width - right_w
    
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1)
    c.rect(inst_x, content_y_start, right_w, content_height)
    
    # Header
    c.line(inst_x, content_y_top - 20, inst_x + right_w, content_y_top - 20)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(inst_x + right_w/2, content_y_top - 14, "INSTRUCTIONS FOR FILLING THE SHEET")
    
    # List
    lines = [
        "1. This sheet should not be folded or crushed.",
        "2. Use only blue/black ball point pen to fill the circles.",
        "3. Use of pencil is strictly prohibited.",
        "4. Circles should be darkened completely and properly.",
        "5. Cutting and erasing on this sheet is not allowed."
    ]
    c.setFont("Helvetica", 7)
    ly = content_y_top - 32
    for line in lines:
        c.drawString(inst_x + 5, ly, line)
        ly -= 10
        
    # Visual Examples
    c.setFont("Helvetica-Bold", 7)
    
    # Centers for the two groups
    center_wrong_x = inst_x + right_w/4
    center_correct_x = inst_x + 3*right_w/4 - 10 # Slightly left of true center to account for wider "Correct" bubbles? No, just center it. 
    # Actually, visual balance might need slight tweak. Let's start with true centers.
    center_correct_x = inst_x + 3*right_w/4
    
    # Draw Centered Strings
    c.drawCentredString(center_wrong_x, ly - 10, "WRONG METHODS")
    c.drawCentredString(center_correct_x, ly - 10, "CORRECT METHODS")
    
    # Draw circles for examples
    y_circ = ly - 18 # Reduced gap
    r = 5
    gap_circ = 15
    
    # Calculate start x for centered groups
    # Group width = 3 gaps * 15 + 2*5 (edges) = 45 + 10 = 55? 
    # Center of group to center of first bubble:
    # First bubble center is at X. Last is X + 45. Midpoint is X + 22.5.
    # So X = Center - 22.5
    
    start_wrong_cx = center_wrong_x - 22.5
    start_correct_cx = center_correct_x - 22.5
    
    # Wrong 1 (Cross)
    cx = start_wrong_cx
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.line(cx-r+2, y_circ-r+2, cx+r-2, y_circ+r-2)
    c.line(cx-r+2, y_circ+r-2, cx+r-2, y_circ-r+2)
    
    # Wrong 2 (Dot)
    cx += gap_circ
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.circle(cx, y_circ, 2, stroke=0, fill=1)
    
    # Wrong 3 (Scribble)
    cx += gap_circ
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.line(cx-r, y_circ, cx+r, y_circ) # simple strike
    
    # Wrong 4 (Check)
    cx += gap_circ
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.line(cx-2, y_circ-2, cx, y_circ+2)
    c.line(cx, y_circ+2, cx+3, y_circ-4)

    # Correct (Filled)
    cx = start_correct_cx
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.circle(cx + gap_circ, y_circ, r, stroke=1, fill=0)
    c.circle(cx + gap_circ*2, y_circ, r, stroke=1, fill=0)
    c.circle(cx + gap_circ*3, y_circ, r, stroke=1, fill=1) # The filled one
        
    return y

def draw_phone_column(c, x, y, width, height, title, rows=10):
    """Draw a phone number/numeric field column"""
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1) # Thinner border for fields
    c.rect(x, y, width, height, stroke=1, fill=0)
    
    # Title
    c.setFillColor(TEXT_COLOR)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + width/2, y + height - 15, title)
    
    # Write Boxes
    box_w = 14
    box_h = 15 # Increased from 14
    gap = 3
    margin_x = (width - (rows * (box_w + gap))) / 2
    
    write_y = y + height - 38 # Increased offset from 35
    
    for i in range(rows):
        bx = x + margin_x + i*(box_w + gap)
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.rect(bx, write_y, box_w, box_h)
    
    # Bubbles 0-9
    bubble_start_y = write_y - 13 # Increased gap
    bubble_radius = 4.4 # Increased from 4.2
    bubble_v_gap = 3.5 # Increased from 3
    
    c.setFont("Helvetica", 7)
    
    for i in range(rows): # Columns 0-9 (digits of phone number)
        col_center_x = x + margin_x + i*(box_w + gap) + box_w/2
        
        for digit in range(10): # Rows 0-9
            by = bubble_start_y - digit * (bubble_radius*2 + bubble_v_gap)
            
            # Draw circle
            c.setStrokeColor(BORDER_COLOR)
            c.setLineWidth(0.8)
            c.circle(col_center_x, by, bubble_radius, stroke=1, fill=0)
            
            # Text
            c.setFillColor(TEXT_COLOR)
            c.drawCentredString(col_center_x, by - 2.5, str(digit))


def draw_section_box(c, x, y, width, height, config):
    """Draw a question section box (reused logic)"""
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1.5)
    c.rect(x, y, width, height, stroke=1, fill=0)
    
    # Section Name
    c.setFillColor(TEXT_COLOR)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + width/2, y + height - 15, config["name"])
    
    options = config["options"]
    num_cols = config["columns"]
    col_width = width / num_cols
    
    bubble_radius = 6.3 # Increased from 6.0
    bubble_spacing = 16
    q_num_width = 20
    
    # Headers A B C D
    header_y = y + height - 28 # Move header down slightly
    c.setFont("Helvetica-Bold", 8)
    
    for col in range(num_cols):
        col_x = x + (col * col_width)
        start_x = col_x + q_num_width + 10
        for i, opt in enumerate(options):
            c.drawCentredString(start_x + i*bubble_spacing, header_y, opt)
            
    # Questions
    c.setFont("Helvetica", 9)
    start_q = config["start_q"]
    q_y_start = header_y - 17
    row_h = 16.5 # Increased from 15.5
    
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
            start_x = col_x + q_num_width + 10
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

def generate_advanced_omr():
    filename = "omr_sheet_advanced.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4 # 595.27, 841.89
    
    # Grid/Layout System
    margin = 30
    content_w = w - 2*margin
    
    # 1. Alignment Markers
    draw_alignment_markers(c, w, h)
    
    # 2. Header Area (Top 25% approx)
    header_h = 160 # Increased from 145 for more instruction space
    header_y = h - margin - header_h - 20 # 20px padding from top marker
    draw_header(c, margin, header_y, content_w, header_h)
    
    # 3. Main Body Split
    # Calculate Right Column Height FIRST to align Left Column
    
    # Right Column Layout Calculation
    sig_h = 55 # Increased from 50
    title_space = 22 # Increased from 20
    section_gap = 10 # Increased from 8
    
    # Calculate section heights
    section_heights = []
    for sec in SECTIONS_CONFIG:
        max_rows = max(sec["questions_per_col"])
        est_h = 38 + (max_rows * 16.5) + 5 # Adjusted for thicker rows
        section_heights.append(est_h)
        
    total_sections_h = sum(section_heights) + (len(section_heights) - 1) * section_gap
    total_right_h = sig_h + title_space + total_sections_h

    # Anchor body to bottom margin + padding
    # This pushes everything down, creating gap below header
    bottom_margin = margin + 20 
    body_y_top = bottom_margin + total_right_h
    
    # Left Column Layout Calculation
    # Align bottom of Alternat Number with bottom of Section 3
    total_left_h = total_right_h
    
    # We have 3 phone boxes and 2 gaps
    phone_gap = 10
    phone_box_h = (total_left_h - 2 * phone_gap) / 3
    
    # Column Widths
    # Reduce Left Width to decrease gaps as requested (was 0.40, tried 0.33, settling on 0.35)
    col_gap = 15
    left_w = content_w * 0.35
    right_w = content_w - left_w - col_gap
    
    # LEFT COLUMN - 3 Phone Fields
    # Start Y is same as Right Column Start Y (body_y_top)
    
    draw_phone_column(c, margin, body_y_top - phone_box_h, left_w, phone_box_h, "Phone Number (WhatsApp No.)")
    draw_phone_column(c, margin, body_y_top - 2*phone_box_h - phone_gap, left_w, phone_box_h, "Parent / Guardian Phone No.")
    draw_phone_column(c, margin, body_y_top - 3*phone_box_h - 2*phone_gap, left_w, phone_box_h, "Alternate Number")
    
    # RIGHT COLUMN - Signature + 3 Sections
    right_x = margin + left_w + col_gap
    
    # Signature Box
    c.setStrokeColor(colors.black) 
    c.rect(right_x, body_y_top - sig_h, right_w, sig_h)
    c.setFont("Helvetica", 7)
    c.drawString(right_x + 5, body_y_top - 15, "By signing, I affirm that information provided is true.")
    c.drawString(right_x + 5, body_y_top - 25, "I understand that attempting this test does not guarantee admission.")
    c.drawRightString(right_x + right_w - 10, body_y_top - 45, "Signature & Date: __________________________")
    
    # Answer Sections
    # "Bubble Your Answers" Title
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(right_x + right_w/2, body_y_top - sig_h - 15, "Bubble Your Answers")
    
    answers_start_y = body_y_top - sig_h - title_space
    current_y = answers_start_y
    
    for i, sec in enumerate(SECTIONS_CONFIG):
        sec_h = section_heights[i]
        draw_section_box(c, right_x, current_y - sec_h, right_w, sec_h, sec)
        current_y -= (sec_h + section_gap)
        
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
    generate_advanced_omr()
