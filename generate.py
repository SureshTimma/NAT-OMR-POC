import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import fitz  # PyMuPDF
import os

def draw_omr_sheet(filename="omr_sheet.pdf", sections=[25, 18, 17]):
    # PDF Setup
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    # Define calibration markers (Square, 20x20)
    marker_size = 20
    margin = 50
    # Top markers moved DOWN to exclude header
    top_marker_y = height - 90 
    bottom_marker_y = margin
    
    # Title & Instructions (OUTSIDE the markers - Top of Page)
    # Move text higher up
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 40, "OMR Answer Sheet")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 60, "Instructions: Fill bubbles completely.")

    # Draw Markers
    c.setFillColor(colors.black)
    
    # Markers Loop
    markers = [
        (margin, top_marker_y), # TL
        (width - margin - marker_size, top_marker_y), # TR
        (margin, bottom_marker_y), # BL
        (width - margin - marker_size, bottom_marker_y) # BR
    ]
    
    for (mx, my) in markers:
        # PDF Only
        c.rect(mx, my, marker_size, marker_size, fill=1)
    
    # Layout Config
    # Optimize vertical space - Questions start inside the marker region
    start_y = top_marker_y - 20 
    col_x_start = 80
    col_width = 250
    row_height = 18 
    
    current_y = start_y
    current_col = 0
    
    q_counter = 1
    
    c.setFont("Helvetica", 11)
    
    for sec_idx, num_q in enumerate(sections):
        # Draw Section Header
        if current_y < margin + 50:
            current_col += 1
            current_y = start_y
            
        if current_col > 1:
             print("Warning: Content might exceed 2 columns and A4 width.")
        
        # PDF Header
        c.setFont("Helvetica-Bold", 12)
        c.drawString(col_x_start + (current_col * col_width), current_y, f"Section {sec_idx + 1}")
        
        current_y -= 25
        
        c.setFont("Helvetica", 11)
        
        for i in range(num_q):
            # Check for column break
            if current_y < margin + 30:
                current_col += 1
                current_y = start_y
                
            x_pos = col_x_start + (current_col * col_width)
            
            # Draw Question Num
            c.drawString(x_pos, current_y, f"{q_counter}.")
            
            # Draw options
            options = ['A', 'B', 'C', 'D']
            for idx, opt in enumerate(options):
                opt_x = x_pos + 35 + (idx * 35)
                # Bubble - Reduced radius to 6
                c.circle(opt_x + 6, current_y + 4, 6, stroke=1, fill=0)
                
                # Text
                c.setFont("Helvetica", 7)
                c.drawCentredString(opt_x + 6, current_y + 1.5, opt)
                
                c.setFont("Helvetica", 11)

            current_y -= row_height
            q_counter += 1
        
        current_y -= 10 # Gap between sections

    c.save()
    print(f"Generated PDF: {filename}")
    
    # Convert to JPG using PyMuPDF (fitz)
    convert_pdf_to_jpg(filename)

def convert_pdf_to_jpg(pdf_path, dpi=300):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)  # 0 is the first page
        pix = page.get_pixmap(dpi=dpi)
        
        jpg_filename = pdf_path.replace(".pdf", ".jpg")
        pix.save(jpg_filename)
        print(f"Generated Image: {jpg_filename}")
    except Exception as e:
        print(f"Error converting PDF to Image: {e}")

if __name__ == "__main__":
    draw_omr_sheet(sections=[25, 18, 17])


