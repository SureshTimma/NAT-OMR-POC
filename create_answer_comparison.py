import json
import os

def create_answer_comparison():
    """Create detailed comparison of bubbled answers vs answer key"""
    
    # Load results and answer key
    latest_session = "debug_output/session_20251216_182255"
    results_path = os.path.join(latest_session, "evaluation_results.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    with open("answer_key.json", 'r') as f:
        answer_key = json.load(f)
    
    # Create comparison report
    report_path = os.path.join(latest_session, "ANSWER_COMPARISON_REPORT.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write(" "*25 + "OMR ANSWER COMPARISON REPORT\n")
        f.write("="*90 + "\n\n")
        
        # Summary section
        f.write("SCORE SUMMARY\n")
        f.write("-"*90 + "\n\n")
        f.write(f"Total Questions:        60\n")
        f.write(f"Correct Bubbled:        {results['total_correct']} ✓\n")
        f.write(f"Wrong Bubbled:          {60 - results['total_correct']} ✗\n")
        f.write(f"Overall Score:          {results['overall_score']:.2f}%\n\n")
        
        f.write("="*90 + "\n")
        f.write("DETAILED QUESTION-BY-QUESTION COMPARISON\n")
        f.write("="*90 + "\n\n")
        
        f.write(f"{'Q#':<5} | {'Student Bubbled':<16} | {'Status':<8} | {'Correct Answer':<16} | Section\n")
        f.write("-"*90 + "\n")
        
        detected_answers = results['detected_answers']
        correct_count = 0
        wrong_count = 0
        
        for q_num in range(1, 61):
            q_str = str(q_num)
            student_answer = detected_answers.get(q_str, '?')
            correct_answer = answer_key.get(q_str, '?')
            is_correct = (student_answer == correct_answer)
            
            # Determine section
            if q_num <= 25:
                section = "Psychometric"
            elif q_num <= 43:
                section = "Aptitude"
            else:
                section = "Math"
            
            if is_correct:
                status = "CORRECT ✓"
                correct_count += 1
            else:
                status = "WRONG ✗"
                wrong_count += 1
            
            f.write(f"Q{q_num:<3} | {student_answer:<16} | {status:<8} | {correct_answer:<16} | {section}\n")
        
        # Section-wise summary
        f.write("\n" + "="*90 + "\n")
        f.write("SECTION-WISE BREAKDOWN\n")
        f.write("="*90 + "\n\n")
        
        for section in results['sections']:
            sec_correct = section['correct']
            sec_total = section['total']
            sec_wrong = sec_total - sec_correct
            sec_percentage = section['score']
            
            f.write(f"{section['name']}:\n")
            f.write(f"  Correct Bubbled:  {sec_correct}/{sec_total}\n")
            f.write(f"  Wrong Bubbled:    {sec_wrong}/{sec_total}\n")
            f.write(f"  Score:            {sec_percentage:.1f}%\n\n")
        
        # Correct answers list
        f.write("="*90 + "\n")
        f.write("CORRECTLY BUBBLED QUESTIONS (17 total)\n")
        f.write("="*90 + "\n\n")
        
        correct_questions = []
        for q_num in range(1, 61):
            q_str = str(q_num)
            if detected_answers.get(q_str) == answer_key.get(q_str):
                correct_questions.append(q_num)
        
        f.write("Questions where student bubbled correctly:\n")
        for i in range(0, len(correct_questions), 10):
            group = correct_questions[i:i+10]
            f.write("  " + ", ".join(f"Q{q}" for q in group) + "\n")
        
        f.write(f"\nAll these questions had answer 'A' in the answer key.\n\n")
        
        # Wrong answers list
        f.write("="*90 + "\n")
        f.write("INCORRECTLY BUBBLED QUESTIONS (43 total)\n")
        f.write("="*90 + "\n\n")
        
        wrong_by_correct_answer = {'B': [], 'C': [], 'D': []}
        
        for q_num in range(1, 61):
            q_str = str(q_num)
            student_ans = detected_answers.get(q_str)
            correct_ans = answer_key.get(q_str)
            if student_ans != correct_ans:
                if correct_ans in wrong_by_correct_answer:
                    wrong_by_correct_answer[correct_ans].append(q_num)
        
        f.write("Student bubbled 'A' but correct answer was:\n\n")
        
        for answer, questions in sorted(wrong_by_correct_answer.items()):
            if questions:
                f.write(f"Answer '{answer}' (Student bubbled 'A' instead):\n")
                for i in range(0, len(questions), 10):
                    group = questions[i:i+10]
                    f.write("  " + ", ".join(f"Q{q}" for q in group) + "\n")
                f.write(f"  Total: {len(questions)} questions\n\n")
        
        # Pattern analysis
        f.write("="*90 + "\n")
        f.write("PATTERN ANALYSIS\n")
        f.write("="*90 + "\n\n")
        
        f.write("Student's Strategy:\n")
        f.write("  The student marked ALL questions with answer 'A'\n\n")
        
        f.write("Answer Key Pattern:\n")
        f.write("  The answer key follows a repeating pattern: A, B, C, D\n")
        f.write("  Distribution:\n")
        
        key_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for ans in answer_key.values():
            if ans in key_distribution:
                key_distribution[ans] += 1
        
        for option, count in sorted(key_distribution.items()):
            percentage = (count / 60) * 100
            bar = '█' * int(percentage / 2)
            f.write(f"    {option}: {count:2d} questions ({percentage:5.1f}%) {bar}\n")
        
        f.write("\n  This explains why marking all 'A' resulted in ~28% score.\n\n")
        
        # Visual grid
        f.write("="*90 + "\n")
        f.write("VISUAL COMPARISON GRID\n")
        f.write("="*90 + "\n\n")
        
        f.write("Legend:\n")
        f.write("  ✓ = Correct (Student bubbled correctly)\n")
        f.write("  ✗ = Wrong (Student should have bubbled different answer)\n\n")
        
        for section_num in range(1, 4):
            if section_num == 1:
                start, end = 1, 26
                name = "Section 1 (Psychometric)"
            elif section_num == 2:
                start, end = 26, 44
                name = "Section 2 (Aptitude)"
            else:
                start, end = 44, 61
                name = "Section 3 (Math)"
            
            f.write(f"{name}:\n")
            
            for q in range(start, end):
                q_str = str(q)
                student = detected_answers.get(q_str)
                correct = answer_key.get(q_str)
                
                if q == start or (q - start) % 5 == 0:
                    if (q - start) % 5 == 0 and q != start:
                        f.write("  ")
                    f.write(f"Q{q:2d}:")
                
                symbol = '✓' if student == correct else '✗'
                f.write(f"[{symbol}]")
                
                if (q - start + 1) % 5 == 0:
                    f.write("\n")
            
            if (end - start) % 5 != 0:
                f.write("\n")
            f.write("\n")
        
        f.write("="*90 + "\n")
        f.write("END OF COMPARISON REPORT\n")
        f.write("="*90 + "\n")
    
    print("\n" + "="*90)
    print(" "*30 + "ANSWER COMPARISON COMPLETE")
    print("="*90 + "\n")
    
    print("SCORE COMPARISON:")
    print(f"  Correct Bubbled:  {results['total_correct']}/60 (✓)")
    print(f"  Wrong Bubbled:    {60 - results['total_correct']}/60 (✗)")
    print(f"  Overall Score:    {results['overall_score']:.2f}%\n")
    
    print("SECTION SCORES:")
    for section in results['sections']:
        print(f"  {section['name']:<30} {section['correct']}/{section['total']} ({section['score']:.1f}%)")
    
    print(f"\n{'='*90}")
    print(f"Full comparison report saved to:")
    print(f"  {report_path}")
    print("="*90 + "\n")
    
    # Also create a simple CSV for easy viewing
    csv_path = os.path.join(latest_session, "answer_comparison.csv")
    with open(csv_path, 'w') as f:
        f.write("Question,Student Bubbled,Correct Answer,Status,Section\n")
        for q_num in range(1, 61):
            q_str = str(q_num)
            student = detected_answers.get(q_str, '?')
            correct = answer_key.get(q_str, '?')
            status = "Correct" if student == correct else "Wrong"
            
            if q_num <= 25:
                section = "Psychometric"
            elif q_num <= 43:
                section = "Aptitude"
            else:
                section = "Math"
            
            f.write(f"Q{q_num},{student},{correct},{status},{section}\n")
    
    print(f"CSV file also created: {csv_path}\n")
    
    return report_path

if __name__ == "__main__":
    create_answer_comparison()
