"""
Generate OMR Sheet with Rectangular Section Boxes
Layout similar to the reference NAT exam sheet with:
- Section 1 (Math): Q1-13, 4 options, 3 columns
- Section 2 (Critical Thinking): Q14-27, 4 options, 3 columns  
- Section 3 (Pyschometric): Q28-71, 5 options, 3 columns
Each section enclosed in a rectangular border
"""

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import fitz  # PyMuPDF
import math

# Section Configuration - Updated layout
# Section 1: Pyschometric - 25 questions (4 options A-D)
# Section 2: Aptitude - 18 questions (4 options A-D)
# Section 3: Math - 17 questions (4 options A-D)
# Total: 60 questions

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

# Colors
BORDER_COLOR = HexColor("#8B0000")  # Dark red/maroon for borders
TEXT_COLOR = colors.black


def draw_omr_sheet_sectioned(filename="omr_sheet_sectioned.pdf"):
    """Generate OMR sheet with sections in rectangular boxes"""
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    # Margins
    margin_left = 40
    margin_right = 40
    margin_top = 60
    margin_bottom = 40
    
    # Calculate available space
    available_width = width - margin_left - margin_right
    available_height = height - margin_top - margin_bottom
    
    # ========== ALIGNMENT MARKERS (Square boxes at 4 corners) ==========
    marker_size = 15
    marker_margin = 25  # Distance from page edge
    
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)
    
    # Top-Left marker
    c.rect(marker_margin, height - marker_margin - marker_size, 
           marker_size, marker_size, stroke=0, fill=1)
    
    # Top-Right marker
    c.rect(width - marker_margin - marker_size, height - marker_margin - marker_size, 
           marker_size, marker_size, stroke=0, fill=1)
    
    # Bottom-Left marker
    c.rect(marker_margin, marker_margin, 
           marker_size, marker_size, stroke=0, fill=1)
    
    # Bottom-Right marker
    c.rect(width - marker_margin - marker_size, marker_margin, 
           marker_size, marker_size, stroke=0, fill=1)
    
    # ========== END ALIGNMENT MARKERS ==========
    
    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, height - 35, "Bubble Your Answers")
    
    # Section layout parameters
    section_gap = 15  # Gap between sections
    
    # Calculate section heights based on number of questions
    max_rows_per_section = []
    for sec in SECTIONS_CONFIG:
        max_rows = max(sec["questions_per_col"])
        max_rows_per_section.append(max_rows)
    
    # Row height and header space
    row_height = 22
    header_height = 30
    bubble_area_top_margin = 10
    
    # Calculate actual section heights
    section_heights = []
    for max_rows in max_rows_per_section:
        h = header_height + bubble_area_top_margin + (max_rows * row_height) + 15
        section_heights.append(h)
    
    total_sections_height = sum(section_heights) + (len(SECTIONS_CONFIG) - 1) * section_gap
    
    # Start Y position (from top)
    current_y = height - margin_top
    
    # Draw each section
    for sec_idx, section in enumerate(SECTIONS_CONFIG):
        sec_height = section_heights[sec_idx]
        
        # Section box coordinates
        box_x = margin_left
        box_y = current_y - sec_height
        box_width = available_width
        box_height = sec_height
        
        # Draw section border (dark red rectangle)
        c.setStrokeColor(BORDER_COLOR)
        c.setLineWidth(1.5)
        c.rect(box_x, box_y, box_width, box_height, stroke=1, fill=0)
        
        # Section title (centered at top of box)
        c.setFillColor(TEXT_COLOR)
        c.setFont("Helvetica-Bold", 11)
        title_y = current_y - 20
        c.drawCentredString(width / 2, title_y, section["name"])
        
        # Draw options header (A B C D or A B C D E)
        options = section["options"]
        num_cols = section["columns"]
        questions_per_col = section["questions_per_col"]
        
        # Calculate column widths
        col_width = box_width / num_cols
        
        # Bubble parameters
        bubble_radius = 8
        bubble_spacing = 23  # Horizontal spacing between bubbles
        q_num_width = 25  # Space for question number
        
        # Draw column headers (options labels)
        c.setFont("Helvetica-Bold", 9)
        header_y = title_y - 25
        
        for col in range(num_cols):
            col_x = box_x + (col * col_width)
            
            # Calculate starting X for options in this column
            options_start_x = col_x + q_num_width + 15
            
            # Draw option letters as header
            for opt_idx, opt in enumerate(options):
                opt_x = options_start_x + (opt_idx * bubble_spacing)
                c.drawCentredString(opt_x, header_y, opt)
        
        # Draw questions and bubbles
        c.setFont("Helvetica", 10)
        q_counter = section["start_q"]
        bubble_y_start = header_y - 20
        
        for col in range(num_cols):
            col_x = box_x + (col * col_width)
            num_questions_in_col = questions_per_col[col]
            
            for row in range(num_questions_in_col):
                q_y = bubble_y_start - (row * row_height)
                
                # Question number
                q_text = f"{q_counter}"
                c.drawString(col_x + 10, q_y - 3, q_text)
                
                # Bubbles
                options_start_x = col_x + q_num_width + 15
                
                for opt_idx, opt in enumerate(options):
                    bubble_x = options_start_x + (opt_idx * bubble_spacing)
                    
                    # Draw bubble circle
                    c.setStrokeColor(BORDER_COLOR)
                    c.setLineWidth(0.8)
                    c.circle(bubble_x, q_y, bubble_radius, stroke=1, fill=0)
                    
                    # Option letter inside bubble (smaller font)
                    c.setFillColor(TEXT_COLOR)
                    c.setFont("Helvetica", 7)
                    c.drawCentredString(bubble_x, q_y - 2.5, opt)
                    c.setFont("Helvetica", 10)
                
                q_counter += 1
        
        # Move to next section
        current_y = box_y - section_gap
    
    c.save()
    print(f"Generated PDF: {filename}")
    
    # Convert to JPG
    convert_pdf_to_jpg(filename)
    

def convert_pdf_to_jpg(pdf_path, dpi=300):
    """Convert PDF to high-quality JPG"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        
        jpg_filename = pdf_path.replace(".pdf", ".jpg")
        pix.save(jpg_filename)
        print(f"Generated Image: {jpg_filename}")
        
        
        doc.close()
    except Exception as e:
        print(f"Error converting PDF to Image: {e}")


def draw_omr_sheet_custom(filename, sections_config):
    """
    Generate custom OMR sheet with provided section configuration
    
    sections_config: list of dicts with keys:
        - name: Section name
        - start_q: Starting question number
        - end_q: Ending question number
        - options: List of option letters (e.g., ["A", "B", "C", "D"])
        - columns: Number of columns
        - questions_per_col: List of questions per column (optional, auto-calculated if not provided)
    """
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    margin_left = 40
    margin_right = 40
    margin_top = 60
    
    available_width = width - margin_left - margin_right
    
    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, height - 35, "Bubble Your Answers")
    
    section_gap = 15
    row_height = 22
    header_height = 30
    bubble_area_top_margin = 10
    
    # Auto-calculate questions_per_col if not provided
    for sec in sections_config:
        if "questions_per_col" not in sec:
            total_q = sec["end_q"] - sec["start_q"] + 1
            cols = sec["columns"]
            base = total_q // cols
            remainder = total_q % cols
            sec["questions_per_col"] = [base + (1 if i < remainder else 0) for i in range(cols)]
    
    # Calculate section heights
    section_heights = []
    for sec in sections_config:
        max_rows = max(sec["questions_per_col"])
        h = header_height + bubble_area_top_margin + (max_rows * row_height) + 15
        section_heights.append(h)
    
    current_y = height - margin_top
    
    for sec_idx, section in enumerate(sections_config):
        sec_height = section_heights[sec_idx]
        
        box_x = margin_left
        box_y = current_y - sec_height
        box_width = available_width
        box_height = sec_height
        
        # Border
        c.setStrokeColor(BORDER_COLOR)
        c.setLineWidth(1.5)
        c.rect(box_x, box_y, box_width, box_height, stroke=1, fill=0)
        
        # Title
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
        
        # Headers
        c.setFont("Helvetica-Bold", 9)
        header_y = title_y - 25
        
        for col in range(num_cols):
            col_x = box_x + (col * col_width)
            options_start_x = col_x + q_num_width + 15
            
            for opt_idx, opt in enumerate(options):
                opt_x = options_start_x + (opt_idx * bubble_spacing)
                c.drawCentredString(opt_x, header_y, opt)
        
        # Questions and bubbles
        c.setFont("Helvetica", 10)
        q_counter = section["start_q"]
        bubble_y_start = header_y - 20
        
        for col in range(num_cols):
            col_x = box_x + (col * col_width)
            num_questions_in_col = questions_per_col[col]
            
            for row in range(num_questions_in_col):
                q_y = bubble_y_start - (row * row_height)
                
                c.drawString(col_x + 10, q_y - 3, f"{q_counter}")
                
                options_start_x = col_x + q_num_width + 15
                
                for opt_idx, opt in enumerate(options):
                    bubble_x = options_start_x + (opt_idx * bubble_spacing)
                    
                    c.setStrokeColor(BORDER_COLOR)
                    c.setLineWidth(0.8)
                    c.circle(bubble_x, q_y, bubble_radius, stroke=1, fill=0)
                    
                    c.setFillColor(TEXT_COLOR)
                    c.setFont("Helvetica", 7)
                    c.drawCentredString(bubble_x, q_y - 2.5, opt)
                    c.setFont("Helvetica", 10)
                
                q_counter += 1
        
        current_y = box_y - section_gap
    
    c.save()
    print(f"Generated PDF: {filename}")
    convert_pdf_to_jpg(filename)


if __name__ == "__main__":
    # Generate the standard NAT-style OMR sheet
    draw_omr_sheet_sectioned("omr_sheet_sectioned.pdf")
    
    # Example: Generate custom sheet
    # custom_sections = [
    #     {"name": "English", "start_q": 1, "end_q": 20, "options": ["A","B","C","D"], "columns": 2},
    #     {"name": "Math", "start_q": 21, "end_q": 40, "options": ["A","B","C","D","E"], "columns": 2},
    # ]
    # draw_omr_sheet_custom("custom_omr.pdf", custom_sections)

