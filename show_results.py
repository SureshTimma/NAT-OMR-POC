import json

# Load the report
with open('omr_sheet_new_updates_filled_report.json', 'r') as f:
    data = json.load(f)

print('\n' + '='*70)
print('         OMR EVALUATION RESULTS - ALL A ANSWERS')
print('='*70)
print(f'\n📊 OVERALL PERFORMANCE:')
print(f'   Total Questions: {data["total_questions"]}')
print(f'   ✅ Correct Answers: {data["correct"]}')
print(f'   ❌ Incorrect Answers: {data["incorrect"]}')
print(f'   📈 Score: {data["score_percentage"]}%')

print('\n' + '='*70)
print('📝 SECTION-WISE BREAKDOWN:')
print('='*70)

sec1 = [d for d in data['details'] if 1 <= d['question'] <= 25]
sec2 = [d for d in data['details'] if 26 <= d['question'] <= 43]
sec3 = [d for d in data['details'] if 44 <= d['question'] <= 60]

sec1_correct = sum(1 for d in sec1 if d["is_correct"])
sec2_correct = sum(1 for d in sec2 if d["is_correct"])
sec3_correct = sum(1 for d in sec3 if d["is_correct"])

print(f'\n   Section 1 (Psychometric - Q1-25):')
print(f'      Score: {sec1_correct}/25 ({sec1_correct/25*100:.1f}%)')

print(f'\n   Section 2 (Aptitude - Q26-43):')
print(f'      Score: {sec2_correct}/18 ({sec2_correct/18*100:.1f}%)')

print(f'\n   Section 3 (Math - Q44-60):')
print(f'      Score: {sec3_correct}/17 ({sec3_correct/17*100:.1f}%)')

print('\n' + '='*70)
print('❌ SAMPLE OF INCORRECT ANSWERS:')
print('='*70)

incorrect = [d for d in data['details'] if not d['is_correct']]
for d in incorrect[:10]:
    print(f'   Q{d["question"]}: Marked A, Correct answer is {d["correct_answer"]}')

if len(incorrect) > 10:
    print(f'   ... and {len(incorrect) - 10} more incorrect answers')

print('\n' + '='*70)
print('📄 Full report saved to: omr_sheet_new_updates_filled_report.json')
print('='*70)
print()
