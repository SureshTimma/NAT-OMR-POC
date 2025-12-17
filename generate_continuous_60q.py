import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import fitz  # PyMuPDF
import math

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
    bar_y = field_start_y - 45 
    c.setFillColor(colors.black)
    
    barcode_width = 150
    barcode_start_x = x + (left_w - barcode_width) / 2
    
    current_bx = barcode_start_x
    import random
    random.seed(42)
    for _ in range(40):
        bw = random.choice([1, 2, 3])
        c.rect(current_bx, bar_y + 5, bw, 25, fill=1, stroke=0)
        current_bx += bw + random.choice([1, 2])
    
    # --- RIGHT PANEL: Instructions ---
    inst_x = x + width - right_w
    
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1)
    c.rect(inst_x, content_y_start, right_w, content_height)
    
    # Title
    c.line(inst_x, content_y_top - 18, inst_x + right_w, content_y_top - 18)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(inst_x + right_w/2, content_y_top - 12, "INSTRUCTIONS FOR FILLING THE SHEET")
    
    # Text Instructions (Left side of box)
    lines = [
        "1. Do not fold/crush.",
        "2. Blue/Black pen only.",
        "3. No pencils allowed.",
        "4. Darken completely.",
        "5. No erasing."
    ]
    c.setFont("Helvetica", 7)
    ly = content_y_top - 28
    text_margin_x = inst_x + 5
    
    for line in lines:
        c.drawString(text_margin_x, ly, line)
        ly -= 10
        
    # Visual Examples (Right side of box)
    examples_x_start = inst_x + right_w * 0.55 
    examples_y_start = content_y_top - 25
    
    c.setFont("Helvetica-Bold", 6)
    c.drawString(examples_x_start, examples_y_start, "WRONG:")
    # Requested: Decrease space between wrong and correct
    c.drawString(examples_x_start, examples_y_start - 28, "CORRECT:") 
    
    # Wrong Examples
    r = 3.5
    gap_circ = 10
    y_circ = examples_y_start - 12
    
    cx = examples_x_start + 8
    # 1. Cross
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.line(cx-r+1, y_circ-r+1, cx+r-1, y_circ+r-1)
    c.line(cx-r+1, y_circ+r-1, cx+r-1, y_circ-r+1)
    
    cx += gap_circ
    # 2. Dot
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.circle(cx, y_circ, 1.5, stroke=0, fill=1)
    
    cx += gap_circ
    # 3. Horizontal Line
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.line(cx-r, y_circ, cx+r, y_circ) 
    
    cx += gap_circ
    # 4. Tick
    c.circle(cx, y_circ, r, stroke=1, fill=0)
    c.line(cx-2, y_circ-1, cx, y_circ+2)
    c.line(cx, y_circ+2, cx+3, y_circ-3)

    # Correct Pattern
    # Adjusted y position to match reduced text gap
    y_circ_correct = examples_y_start - 40 
    cx = examples_x_start + 8
    
    c.circle(cx, y_circ_correct, r, stroke=1, fill=0)
    c.circle(cx + gap_circ, y_circ_correct, r, stroke=1, fill=0)
    c.circle(cx + gap_circ*2, y_circ_correct, r, stroke=1, fill=0)
    c.circle(cx + gap_circ*3, y_circ_correct, r, stroke=1, fill=1) 
        
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
    
    # Header Separator Line
    c.line(x, y + height - 16, x + width, y + height - 16)
    
    # Write Boxes 
    box_w = 17 
    box_h = 17 
    gap = 4    
    
    total_boxes_width = (rows * box_w) + ((rows - 1) * gap)
    margin_x = (width - total_boxes_width) / 2
    
    # Write box position
    write_y = y + height - 38 
    
    for i in range(rows):
        bx = x + margin_x + i*(box_w + gap)
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.rect(bx, write_y, box_w, box_h)
    
    # Bubbles 0-9 
    # Radius 6.0 is safe and very large.
    bubble_radius = 6.0 
    
    # Vertical Spacing - use all available space
    max_bubble_start_y = write_y - 8
    available_space_below = max_bubble_start_y - y
    
    bubble_v_gap = (available_space_below - (10 * bubble_radius * 2)) / 10
    
    # Start bubble position
    bubble_start_y = max_bubble_start_y - bubble_radius

    c.setFont("Helvetica", 7) 
    
    for i in range(rows): 
        col_center_x = x + margin_x + i*(box_w + gap) + box_w/2
        
        # COLUMN TIMING MARK (BELOW)
        c.setFillColor(colors.black)
        c.rect(col_center_x - 4, y - 5, 8, 3, fill=1, stroke=0)

        for digit in range(10): 
            by = bubble_start_y - digit * (bubble_radius*2 + bubble_v_gap)
            
            # Removed beside marker
            
            c.setStrokeColor(BORDER_COLOR)
            c.setLineWidth(0.8)
            c.circle(col_center_x, by, bubble_radius, stroke=1, fill=0)
            
            c.setFillColor(TEXT_COLOR)
            c.drawCentredString(col_center_x, by - 2.5, str(digit))

def draw_continuous_questions(c, x, y, width, height):
    """Draw continuous 60 questions in 4 columns"""
    c.setStrokeColor(BORDER_COLOR)
    c.setLineWidth(1.5)
    c.rect(x, y, width, height, stroke=1, fill=0)
    
    # Layout Config
    num_cols = 4
    total_q = 60
    q_per_col = 15
    col_width = width / num_cols
    
    bubble_radius = 7.0 # Further Increased
    bubble_spacing = 22 # Increased
    
    # We want to center the whole row content: [Number]  [A] [B] [C] [D]
    # Estimates
    number_gap = 15 # Gap between number-right and A-center
    number_width = 25 # Reserve space for "60"
    
    options = ["A", "B", "C", "D"]
    
    # Headers positions
    header_y = y + height - 40 # Moved Down
    
    # "Bubble Your Answers" Title
    c.setFont("Helvetica-Bold", 10) # Larger Font
    c.setFillColor(TEXT_COLOR)
    # Vertically Centered: +18 -> +23
    c.drawCentredString(x + width/2, header_y + 23, "Bubble Your Answers")
    
    # Separator Line for Questions
    c.setLineWidth(1) # Match border
    # Moved Up: +8 -> +12
    c.line(x, header_y + 12, x + width, header_y + 12)

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(TEXT_COLOR)
    
    for col in range(num_cols):
        col_x = x + (col * col_width)
        col_center_x = col_x + col_width/2
        
        # Calculate centers
        # We model the row as having a "visual left" and "visual right".
        # A_center is our reference 0.
        # D_center is at 3 * spacing.
        # Radius extends D by bubble_radius.
        # Number ends at -number_gap.
        # Number starts at -number_gap - number_width.
        
        # Visual Bounds Relative to A_center:
        # Left: -number_gap - number_width = -15 - 25 = -40
        # Right: 3*spacing + bubble_radius = 3*22 + 7 = 66 + 7 = 73
        
        # Center of this block relative to A_center:
        # (-40 + 73) / 2 = 33 / 2 = 16.5
        
        # So the "Visual Center" is at A_center + 16.5
        # We want Visual Center to be at col_center_x.
        # col_center_x = A_center + 16.5
        # A_center = col_center_x - 16.5
        
        # Adjusted centering logic for A_center (start_x_bubbles)
        block_visual_offset = 16.5
        start_x_bubbles = col_center_x - block_visual_offset
        
        if col < num_cols - 1:
            c.setStrokeColor(LIGHT_GRAY)
            c.setLineWidth(0.5)
            # Vertical line stops at Separator Line (header_y + 12)
            c.line(col_x + col_width, y + 5, col_x + col_width, header_y + 12)
            
        # Draw Options Header (A B C D)
        for i, opt in enumerate(options):
            c.drawCentredString(start_x_bubbles + i*bubble_spacing, header_y, opt)
            
    # Draw Questions
    start_q = 1
    # LOWERED BUBBLES: decreased gap
    q_y_start = header_y - 10 
    
    row_h = 18.0 
    
    c.setFont("Helvetica", 9)
    
    for col in range(num_cols):
        col_x = x + (col * col_width)
        col_center_x = col_x + col_width/2
        
        # Re-calc for loop local consistency
        block_visual_offset = 16.5
        start_x_bubbles = col_center_x - block_visual_offset
        
        for r in range(q_per_col):
            q_num = start_q + (col * q_per_col) + r
            if q_num > total_q: 
                break
                
            y_pos = q_y_start - (r * row_h)
            
            # TIMING MARKS: Add small black rect at start of row in first column
            if col == 0:
                c.setFillColor(colors.black)
                # Left edge of grid is 'x'. We place it slightly inside (+2)
                c.rect(x + 2, y_pos - 1.5, 8, 3, fill=1, stroke=0)
            
            # Question Number
            c.setFillColor(TEXT_COLOR)
            # number_gap = 15
            c.drawRightString(start_x_bubbles - 15, y_pos - 3, str(q_num))
            
            # Bubbles
            for i, opt in enumerate(options):
                bx = start_x_bubbles + i*bubble_spacing
                
                c.setStrokeColor(BORDER_COLOR)
                c.setLineWidth(0.6)
                c.circle(bx, y_pos, bubble_radius, stroke=1, fill=0)
                
                c.setFillColor(TEXT_COLOR)
                c.setFont("Helvetica", 6)
                c.drawCentredString(bx, y_pos - 2, opt)
                c.setFont("Helvetica", 9)
                
def generate_continuous_60q_omr():
    filename = "omr_sheet_continuous_60q.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4 # 595.27, 841.89
    
    margin = 30
    content_w = w - 2*margin
    
    # 1. Alignment Markers
    draw_alignment_markers(c, w, h)
    
    # 2. Header Area
    header_h = 120
    header_y = h - margin - header_h - 10
    draw_header(c, margin, header_y, content_w, header_h)
    
    # 3. Phone Numbers
    body_y_top = header_y - 5 
    
    phone_gap = 20
    phone_width = (content_w - phone_gap) / 2
    
    phone_height = 220 
    
    draw_phone_column(c, margin, body_y_top - phone_height, phone_width, phone_height, "Phone Number (WhatsApp No.)")
    draw_phone_column(c, margin + phone_width + phone_gap, body_y_top - phone_height, phone_width, phone_height, "Parent / Guardian Phone No.")
    
    # 4. Continuous Questions Area
    questions_y_top = body_y_top - phone_height - 15 
    
    sig_h = 40
    sig_y = margin + 5 
    
    # Compact Grid
    total_grid_h = 320 
    
    draw_continuous_questions(c, margin, questions_y_top - total_grid_h, content_w, total_grid_h)
    
    # 5. Signature
    adjusted_sig_y = questions_y_top - total_grid_h - sig_h - 15
    
    c.setStrokeColor(colors.black) 
    c.setLineWidth(1)
    c.rect(margin, adjusted_sig_y, content_w, sig_h)
    
    c.setFillColor(TEXT_COLOR)
    c.setFont("Helvetica", 8)
    c.drawString(margin + 10, adjusted_sig_y + 25, "By signing, I affirm that information provided is true.")
    c.drawString(margin + 10, adjusted_sig_y + 12, "I understand that attempting this test does not guarantee admission.")
    
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(margin + content_w - 20, adjusted_sig_y + 15, "Signature & Date: __________________________")
    
    # "Continuous Layout Update" keyword
    c.setFont("Helvetica", 6)
    c.setFillColor(colors.lightgrey)
    c.drawRightString(w - margin, 5, "OMR_UPDATED_CONTINUOUS_60Q_V10_V12_CENTERED")
    
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
    generate_continuous_60q_omr()
