import os
import uuid
import shutil
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from evaluate import evaluate_omr_sections

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Required for flash output
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is present
        if 'omr_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['omr_image']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            # Create unique run folder
            run_id = str(uuid.uuid4())
            run_dir = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
            os.makedirs(run_dir)
            
            # Save uploaded image
            filename = "filled.jpg" # Standardize name
            file_path = os.path.join(run_dir, filename)
            file.save(file_path)
            
            # Key Path (Using standard key for now, could be uploaded too)
            # Assuming 'answer_key.json' exists in root
            base_key_path = "answer_key.json"
            if not os.path.exists(base_key_path):
                 flash('Server Error: Default answer_key.json not found.')
                 return redirect(request.url)
            
            # Copy key to run dir for record keeping (optional but good)
            run_key_path = os.path.join(run_dir, "expected_key.json")
            shutil.copy(base_key_path, run_key_path)
            
            try:
                # Run Evaluation - Capture return values
                _, section_results = evaluate_omr_sections(file_path, run_key_path, output_dir=run_dir)
                
                # Save section results for persistence (optional, but good for reloading)
                with open(os.path.join(run_dir, "section_results.json"), 'w') as f:
                    json.dump(section_results, f, indent=2)
                
                return redirect(url_for('result', run_id=run_id))
            
            except Exception as e:
                flash(f"Error processing OMR: {str(e)}")
                return redirect(request.url)

    return render_template('index.html')

@app.route('/result/<run_id>')
def result(run_id):
    run_dir = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
    if not os.path.exists(run_dir):
        return "Run ID not found", 404
        
    # JSONs
    try:
        with open(os.path.join(run_dir, "detected_answers.json"), 'r') as f:
            detected = json.load(f)
        with open(os.path.join(run_dir, "expected_key.json"), 'r') as f:
            expected = json.load(f)
        
        # Load section results if available
        section_results = []
        sec_path = os.path.join(run_dir, "section_results.json")
        if os.path.exists(sec_path):
             with open(sec_path, 'r') as f:
                 section_results = json.load(f)
                 
    except FileNotFoundError:
        return "Results not ready or failed", 500

    # Calculate basic stats for the view
    total = len(expected)
    correct = 0
    comparison = []
    
    for q, ans in expected.items():
        user_ans = detected.get(q, "")
        is_match = (user_ans == ans)
        if is_match:
            correct += 1
        comparison.append({
            "q": q,
            "expected": ans,
            "detected": user_ans,
            "match": is_match
        })
    
    # Sort comparison numerically by Question Number
    comparison.sort(key=lambda x: int(x['q']))
    
    score_pct = (correct / total * 100) if total > 0 else 0
    
    return render_template('result.html', 
                           run_id=run_id, 
                           score=f"{score_pct:.2f}%", 
                           correct=correct, 
                           total=total,
                           section_results=section_results,
                           comparison=comparison)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
