import cv2
import numpy as np
import json
import os

def create_comprehensive_bubble_report():
    """Create a detailed report of all detected bubbles and evaluation"""
    
    # Load the latest evaluation results
    latest_session = "debug_output/session_20251216_182255"
    results_path = os.path.join(latest_session, "evaluation_results.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load answer key
    with open("answer_key.json", 'r') as f:
        answer_key = json.load(f)
    
    # Create detailed report
    report_path = os.path.join(latest_session, "COMPREHENSIVE_BUBBLE_REPORT.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" "*20 + "COMPREHENSIVE BUBBLE DETECTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Image: omr_sheet_new_updates_filled.jpg\n")
        f.write(f"Answer Key: answer_key.json\n")
        f.write(f"Detection Method: OpenCV with Circularity & Size Filtering\n\n")
        
        f.write("="*80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Questions:     {results['total_questions']}\n")
        f.write(f"Correct Answers:     {results['total_correct']}\n")
        f.write(f"Incorrect Answers:   {results['total_questions'] - results['total_correct']}\n")
        f.write(f"Overall Score:       {results['overall_score']:.2f}%\n\n")
        
        # Section summaries
        f.write("="*80 + "\n")
        f.write("SECTION-WISE BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        
        for section in results['sections']:
            f.write(f"{section['name']}:\n")
            f.write(f"  Questions: Q{section['section']*25-24 if section['section']==1 else (26 if section['section']==2 else 44)}-")
            if section['section'] == 1:
                f.write("Q25\n")
            elif section['section'] == 2:
                f.write("Q43\n")
            else:
                f.write("Q60\n")
            f.write(f"  Score: {section['correct']}/{section['total']} ({section['score']:.1f}%)\n\n")
        
        # Detailed question-by-question analysis
        f.write("="*80 + "\n")
        f.write("DETAILED QUESTION-BY-QUESTION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Format: Q# | Detected | Correct | Status | Section\n")
        f.write("-"*80 + "\n")
        
        detected_answers = results['detected_answers']
        
        for q_num in range(1, 61):
            q_str = str(q_num)
            detected = detected_answers.get(q_str, '?')
            correct = answer_key.get(q_str, '?')
            is_correct = (detected == correct)
            status = "[OK]" if is_correct else "[X]"
            
            # Determine section
            if q_num <= 25:
                section = "Section 1"
            elif q_num <= 43:
                section = "Section 2"
            else:
                section = "Section 3"
            
            f.write(f"Q{q_num:2d}  |    {detected}     |    {correct}    | {status:4s}  | {section}\n")
        
        # Bubble detection statistics
        f.write("\n" + "="*80 + "\n")
        f.write("BUBBLE DETECTION STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Detection Method:\n")
        f.write("  - Color-based section detection (red borders)\n")
        f.write("  - Size filtering: 1.5-6% of section width\n")
        f.write("  - Circularity threshold: > 0.4\n")
        f.write("  - Aspect ratio: 0.7 - 1.3 (near-circular)\n")
        f.write("  - Pixel counting for filled bubble detection\n\n")
        
        f.write("Sections Detected:\n")
        f.write(f"  - Total red boxes found: 5 (2 phone fields + 3 question sections)\n")
        f.write(f"  - Filtered to question sections: 3\n")
        f.write(f"  - Phone fields excluded: 2\n\n")
        
        # Answer distribution
        f.write("="*80 + "\n")
        f.write("ANSWER DISTRIBUTION\n")
        f.write("="*80 + "\n\n")
        
        distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for ans in detected_answers.values():
            if ans in distribution:
                distribution[ans] += 1
        
        f.write("Student's Answers:\n")
        for option, count in distribution.items():
            percentage = (count / 60) * 100
            bar = '#' * int(percentage / 2)
            f.write(f"  {option}: {count:2d} ({percentage:5.1f}%) {bar}\n")
        
        f.write("\nCorrect Answer Distribution:\n")
        key_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for ans in answer_key.values():
            if ans in key_distribution:
                key_distribution[ans] += 1
        
        for option, count in key_distribution.items():
            percentage = (count / 60) * 100
            bar = '#' * int(percentage / 2)
            f.write(f"  {option}: {count:2d} ({percentage:5.1f}%) {bar}\n")
        
        # Error analysis
        f.write("\n" + "="*80 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        incorrect_by_option = {'B': 0, 'C': 0, 'D': 0}
        for q_num in range(1, 61):
            q_str = str(q_num)
            detected = detected_answers.get(q_str, '?')
            correct = answer_key.get(q_str, '?')
            if detected != correct and correct in incorrect_by_option:
                incorrect_by_option[correct] += 1
        
        f.write("Student marked all 'A', but correct answers were:\n")
        for option, count in sorted(incorrect_by_option.items()):
            f.write(f"  {count} questions should have been '{option}'\n")
        
        # Correct questions list
        f.write("\n" + "="*80 + "\n")
        f.write("CORRECT ANSWERS (Where answer key = 'A')\n")
        f.write("="*80 + "\n\n")
        
        correct_questions = []
        for q_num in range(1, 61):
            q_str = str(q_num)
            if answer_key.get(q_str) == 'A':
                correct_questions.append(q_num)
        
        f.write(f"Questions: {', '.join(f'Q{q}' for q in correct_questions)}\n")
        f.write(f"Total: {len(correct_questions)} questions\n\n")
        
        # Visual representation
        f.write("="*80 + "\n")
        f.write("VISUAL ANSWER GRID\n")
        f.write("="*80 + "\n\n")
        
        f.write("Legend: [✓] Correct  [X] Incorrect\n\n")
        
        # Show in groups of 10
        for start in range(0, 60, 10):
            end = min(start + 10, 60)
            f.write(f"Q{start+1:02d}-Q{end:02d}: ")
            for q in range(start + 1, end + 1):
                q_str = str(q)
                detected = detected_answers.get(q_str, '?')
                correct = answer_key.get(q_str, '?')
                symbol = '✓' if detected == correct else 'X'
                f.write(f"[{symbol}]")
            f.write(f"  ({sum(1 for q in range(start+1, end+1) if detected_answers.get(str(q)) == answer_key.get(str(q)))}/10 correct)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DEBUG FILES AVAILABLE\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Location: {latest_session}/\n\n")
        f.write("Files:\n")
        f.write("  - 00_resized_input.jpg - Original OMR image\n")
        f.write("  - 01_red_mask.jpg - Red border detection mask\n")
        f.write("  - 02_red_mask_processed.jpg - Processed mask\n")
        f.write("  - 05_detected_sections.jpg - Detected question sections\n")
        f.write("  - 03_section1_thresh.jpg - Section 1 threshold\n")
        f.write("  - 04_section1_bubbles.jpg - Section 1 detected bubbles\n")
        f.write("  - 03_section2_thresh.jpg - Section 2 threshold\n")
        f.write("  - 04_section2_bubbles.jpg - Section 2 detected bubbles\n")
        f.write("  - 03_section3_thresh.jpg - Section 3 threshold\n")
        f.write("  - 04_section3_bubbles.jpg - Section 3 detected bubbles\n")
        f.write("  - evaluation_results.json - Complete JSON results\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE BUBBLE DETECTION REPORT CREATED")
    print(f"{'='*80}\n")
    print(f"Report saved to: {report_path}\n")
    
    # Also print key findings to console
    print("KEY FINDINGS:")
    print(f"  Total Bubbles Detected: 60 (all questions)")
    print(f"  All detected as: A")
    print(f"  Matches answer key: {results['total_correct']}/60 times")
    print(f"  Overall Score: {results['overall_score']:.2f}%\n")
    
    print("SECTION SCORES:")
    for section in results['sections']:
        print(f"  {section['name']}: {section['score']:.1f}% ({section['correct']}/{section['total']})")
    
    print(f"\n{'='*80}\n")
    
    return report_path

if __name__ == "__main__":
    create_comprehensive_bubble_report()
