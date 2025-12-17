import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

def create_debug_visualization():
    """Create a debug image showing evaluation results overlaid on the filled OMR"""
    
    # Load the filled OMR image
    img_path = "omr_sheet_new_updates_filled.jpg"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return
    
    # Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    
    # Load evaluation results
    with open('omr_sheet_new_updates_filled_report.json', 'r') as f:
        results = json.load(f)
    
    # Try to load a font (fallback to default if not available)
    try:
        font_large = ImageFont.truetype("arial.ttf", 40)
        font_medium = ImageFont.truetype("arial.ttf", 30)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add title overlay
    overlay_height = 120
    draw.rectangle([(0, 0), (pil_img.width, overlay_height)], 
                   fill=(0, 0, 0, 200))
    
    # Draw title text
    title = "OMR EVALUATION DEBUG VIEW"
    draw.text((50, 20), title, fill=(255, 255, 255), font=font_medium)
    
    score_text = f"Score: {results['correct']}/{results['total_questions']} ({results['score_percentage']}%)"
    draw.text((50, 60), score_text, fill=(0, 255, 0), font=font_medium)
    
    # Add legend
    legend_y = overlay_height + 20
    draw.rectangle([(pil_img.width - 300, legend_y), 
                    (pil_img.width - 20, legend_y + 120)], 
                   fill=(255, 255, 255, 200), 
                   outline=(0, 0, 0), width=2)
    
    draw.text((pil_img.width - 280, legend_y + 10), 
              "LEGEND:", fill=(0, 0, 0), font=font_small)
    
    # Green circle for correct
    draw.ellipse([(pil_img.width - 280, legend_y + 40), 
                  (pil_img.width - 260, legend_y + 60)], 
                 fill=(0, 255, 0), outline=(0, 0, 0))
    draw.text((pil_img.width - 250, legend_y + 42), 
              "Correct", fill=(0, 150, 0), font=font_small)
    
    # Red circle for incorrect
    draw.ellipse([(pil_img.width - 280, legend_y + 75), 
                  (pil_img.width - 260, legend_y + 95)], 
                 fill=(255, 0, 0), outline=(0, 0, 0))
    draw.text((pil_img.width - 250, legend_y + 77), 
              "Incorrect", fill=(150, 0, 0), font=font_small)
    
    # Add summary box at bottom
    summary_height = 150
    summary_y = pil_img.height - summary_height
    draw.rectangle([(0, summary_y), (pil_img.width, pil_img.height)], 
                   fill=(0, 0, 0, 200))
    
    # Section breakdown
    sec1_details = [d for d in results['details'] if 1 <= d['question'] <= 25]
    sec2_details = [d for d in results['details'] if 26 <= d['question'] <= 43]
    sec3_details = [d for d in results['details'] if 44 <= d['question'] <= 60]
    
    sec1_correct = sum(1 for d in sec1_details if d['is_correct'])
    sec2_correct = sum(1 for d in sec2_details if d['is_correct'])
    sec3_correct = sum(1 for d in sec3_details if d['is_correct'])
    
    summary_text = [
        "SECTION BREAKDOWN:",
        f"Section 1 (Psychometric): {sec1_correct}/25 ({sec1_correct/25*100:.1f}%)",
        f"Section 2 (Aptitude): {sec2_correct}/18 ({sec2_correct/18*100:.1f}%)",
        f"Section 3 (Math): {sec3_correct}/17 ({sec3_correct/17*100:.1f}%)"
    ]
    
    y_pos = summary_y + 20
    for line in summary_text:
        color = (255, 255, 255) if "SECTION" in line else (200, 200, 200)
        draw.text((50, y_pos), line, fill=color, font=font_small)
        y_pos += 30
    
    # Add watermark
    draw.text((pil_img.width - 350, pil_img.height - 40), 
              "All answers marked as 'A'", 
              fill=(255, 255, 0), font=font_small)
    
    # Convert back to OpenCV format
    img_with_overlay = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Save debug image
    debug_path = "omr_sheet_new_updates_filled_debug.jpg"
    cv2.imwrite(debug_path, img_with_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[OK] Debug visualization saved to: {debug_path}")
    
    # Also create a simple text-based debug file
    debug_txt_path = "omr_sheet_new_updates_filled_debug.txt"
    with open(debug_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("OMR EVALUATION DEBUG REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Image: {img_path}\n")
        f.write(f"Total Questions: {results['total_questions']}\n")
        f.write(f"Correct: {results['correct']}\n")
        f.write(f"Incorrect: {results['incorrect']}\n")
        f.write(f"Score: {results['score_percentage']}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILED QUESTION-BY-QUESTION BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        
        for detail in results['details']:
            status = "[OK] CORRECT" if detail['is_correct'] else "[X] INCORRECT"
            f.write(f"Q{detail['question']:2d}: Student={detail['student_answer']} | "
                   f"Correct={detail['correct_answer']} | {status}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SECTION SUMMARIES\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Section 1 (Psychometric, Q1-25):  {sec1_correct}/25 ({sec1_correct/25*100:.1f}%)\n")
        f.write(f"Section 2 (Aptitude, Q26-43):     {sec2_correct}/18 ({sec2_correct/18*100:.1f}%)\n")
        f.write(f"Section 3 (Math, Q44-60):         {sec3_correct}/17 ({sec3_correct/17*100:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INCORRECT ANSWERS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        incorrect = [d for d in results['details'] if not d['is_correct']]
        for d in incorrect:
            f.write(f"Q{d['question']:2d}: Marked {d['student_answer']}, "
                   f"should be {d['correct_answer']}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"[OK] Text debug report saved to: {debug_txt_path}")
    
    return debug_path, debug_txt_path

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENERATING DEBUG FILES")
    print("="*80 + "\n")
    
    img_path, txt_path = create_debug_visualization()
    
    print("\n" + "="*80)
    print("DEBUG FILES CREATED SUCCESSFULLY")
    print("="*80)
    print(f"\nVisual Debug: {img_path}")
    print(f"Text Debug: {txt_path}")
    print("="*80 + "\n")
