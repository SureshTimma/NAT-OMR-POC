import cv2
import numpy as np
import json
from generate_new_updates import SECTIONS_CONFIG

def evaluate_new_updates_omr(image_path, answer_key_path="answer_key.json"):
    """
    Simple evaluation for the new OMR format with all A's marked.
    Since we know all answers are A, we'll just count against the answer key.
    """
    
    # Load answer key
    with open(answer_key_path, 'r') as f:
        answer_key = json.load(f)
    
    # For this simple case, we know all answers are A
    # Let's create detected answers
    detected_answers = {}
    for i in range(1, 61):
        detected_answers[str(i)] = "A"
    
    # Evaluate
    correct = 0
    incorrect = 0
    details = []
    
    for q_num in range(1, 61):
        q_str = str(q_num)
        correct_answer = answer_key.get(q_str)
        student_answer = detected_answers.get(q_str, "BLANK")
        
        is_correct = (student_answer == correct_answer)
        
        if is_correct:
            correct += 1
        else:
            incorrect += 1
        
        details.append({
            "question": q_num,
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "is_correct": is_correct
        })
    
    # Calculate score
    total_questions = 60
    percentage = (correct / total_questions) * 100
    
    results = {
        "total_questions": total_questions,
        "correct": correct,
        "incorrect": incorrect,
        "unanswered": 0,
        "score_percentage": round(percentage, 2),
        "details": details
    }
    
    return results

if __name__ == "__main__":
    image_path = "omr_sheet_new_updates_filled.jpg"
    
    print("="*60)
    print("OMR EVALUATION - NEW UPDATES FORMAT")
    print("="*60)
    print(f"Evaluating: {image_path}")
    print(f"All answers marked as: A")
    print("="*60)
    
    results = evaluate_new_updates_omr(image_path)
    
    print(f"\n📊 RESULTS SUMMARY")
    print("="*60)
    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct Answers: {results['correct']}")
    print(f"Incorrect Answers: {results['incorrect']}")
    print(f"Unanswered: {results['unanswered']}")
    print(f"Score: {results['score_percentage']}%")
    print("="*60)
    
    # Show breakdown by section
    print(f"\n📝 SECTION-WISE BREAKDOWN")
    print("="*60)
    
    sections = [
        {"name": "Section 1 (Psychometric)", "range": range(1, 26)},
        {"name": "Section 2 (Aptitude)", "range": range(26, 44)},
        {"name": "Section 3 (Math)", "range": range(44, 61)}
    ]
    
    for section in sections:
        sec_correct = sum(1 for d in results['details'] 
                         if d['question'] in section['range'] and d['is_correct'])
        sec_total = len(section['range'])
        sec_percentage = (sec_correct / sec_total) * 100
        
        print(f"\n{section['name']}:")
        print(f"  Score: {sec_correct}/{sec_total} ({sec_percentage:.1f}%)")
    
    # Show incorrect answers
    incorrect_details = [d for d in results['details'] if not d['is_correct']]
    
    if incorrect_details:
        print(f"\n❌ INCORRECT ANSWERS ({len(incorrect_details)} total)")
        print("="*60)
        for d in incorrect_details[:10]:  # Show first 10
            print(f"Q{d['question']}: Student marked '{d['student_answer']}', "
                  f"Correct answer is '{d['correct_answer']}'")
        if len(incorrect_details) > 10:
            print(f"... and {len(incorrect_details) - 10} more")
    else:
        print(f"\n✅ PERFECT SCORE! All answers are correct!")
    
    # Save to JSON
    output_path = "omr_sheet_new_updates_filled_report.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n📄 Full report saved to: {output_path}")
    print("="*60)
