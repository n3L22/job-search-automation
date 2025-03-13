"""
Main web application module for the Automated Job Search Assistant.

This module provides the Flask web application that serves as the user interface
and connects all the components of the job search automation system.
"""

import os
import logging
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_wtf import CSRFProtect
from flask_wtf.file import FileField
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import job search modules
from jobsearch.job_matcher import JobMatcher
from jobsearch.resume_analyzer import ResumeAnalyzer
from jobsearch.application_filler import ApplicationFiller
# We'll implement the recommendation engine later
# from jobsearch.recommendation_engine import RecommendationEngine

# Import scrapers
# These will be implemented later
# from jobsearch.scrapers.indeed import IndeedScraper
# from jobsearch.scrapers.linkedin import LinkedInScraper
# from jobsearch.scrapers.generic import GenericScraper

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key-for-development')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
csrf = CSRFProtect(app)

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'resumes'), exist_ok=True)

# Initialize core components
use_ai_api = os.getenv('USE_AI_API', 'false').lower() == 'true'
api_key = os.getenv('OPENAI_API_KEY')

job_matcher = JobMatcher(use_api=use_ai_api, api_key=api_key)
resume_analyzer = ResumeAnalyzer(use_api=use_ai_api, api_key=api_key)
application_filler = ApplicationFiller(use_api=use_ai_api, api_key=api_key)

# Global variables
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the user dashboard."""
    # This will be filled with real data when we implement user accounts
    sample_data = {
        'profile_complete': False,
        'resume_uploaded': False,
        'applications': [],
        'recommended_jobs': [],
        'saved_jobs': []
    }
    return render_template('dashboard.html', data=sample_data)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    """Handle user profile information."""
    if request.method == 'POST':
        profile_data = {
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'phone': request.form.get('phone'),
            'current_role': request.form.get('current_role'),
            'desired_title': request.form.get('desired_title'),
            'experience_years': request.form.get('experience_years'),
            'skills': request.form.get('skills', '').split(','),
            'location': request.form.get('location'),
            'field': request.form.get('field'),
            'current_company': request.form.get('current_company'),
            'previous_company': request.form.get('previous_company'),
            'education': request.form.get('education'),
            'linkedin': request.form.get('linkedin'),
            'github': request.form.get('github'),
            'portfolio': request.form.get('portfolio'),
            'salary_range': request.form.get('salary_range')
        }
        
        # In a real application, this would be saved to a database
        # For now, we'll just store it in the session
        session['profile'] = profile_data
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('dashboard'))
        
    # For GET requests, display the form
    profile_data = session.get('profile', {})
    return render_template('profile.html', profile=profile_data)

@app.route('/resume/upload', methods=['GET', 'POST'])
def upload_resume():
    """Handle resume uploads."""
    if request.method == 'POST':
        if 'resume' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['resume']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Secure the filename to prevent security issues
            filename = secure_filename(file.filename)
            
            # Add timestamp to ensure uniqueness
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'resumes', filename)
            file.save(filepath)
            
            # Store resume path in session
            session['resume_path'] = filepath
            
            # Analyze resume (in a real app, this would be a background task)
            try:
                with open(filepath, 'r') as f:
                    resume_text = f.read()
                
                analysis = resume_analyzer.analyze_resume(resume_text)
                
                # Store analysis in session
                session['resume_analysis'] = analysis
                
                # If profile exists, update skills from resume
                if 'profile' in session and analysis.get('skills'):
                    session['profile']['skills'] = analysis.get('skills')
                
                flash('Resume uploaded and analyzed successfully!', 'success')
            except Exception as e:
                logger.error(f"Error analyzing resume: {e}")
                flash('Resume uploaded, but there was an error during analysis.', 'warning')
                
            return redirect(url_for('dashboard'))
            
        else:
            flash(f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
            return redirect(request.url)
            
    return render_template('resume_upload.html')

@app.route('/jobs/search', methods=['GET', 'POST'])
def search_jobs():
    """Search for jobs based on user criteria."""
    if request.method == 'POST':
        search_criteria = {
            'keywords': request.form.get('keywords'),
            'location': request.form.get('location'),
            'job_title': request.form.get('job_title'),
            'company': request.form.get('company'),
            'job_type': request.form.get('job_type')
        }
        
        # In a real app, this would trigger a job search using scrapers
        # For now, return a placeholder response
        return render_template('job_search_results.html', 
                              criteria=search_criteria, 
                              jobs=[])
        
    return render_template('job_search.html')

@app.route('/jobs/match', methods=['POST'])
def match_job():
    """Match a job with user profile."""
    # This endpoint would be called from the extension or frontend
    job_data = request.json
    
    if not job_data:
        return jsonify({'error': 'No job data provided'}), 400
        
    user_profile = session.get('profile', {})
    
    # Perform job matching
    match_result = job_matcher.match_jobs(user_profile, [job_data])[0]
    
    return jsonify(match_result)

@app.route('/jobs/apply', methods=['POST'])
def apply_job():
    """Apply to a job."""
    # This endpoint would be called from the extension or frontend
    job_data = request.json
    form_fields = job_data.get('form_fields', {})
    
    if not job_data or not form_fields:
        return jsonify({'error': 'Invalid job application data'}), 400
        
    user_profile = session.get('profile', {})
    
    # Generate content for application
    cover_letter = application_filler.generate_cover_letter(user_profile, job_data)
    
    # Fill application form
    form_values = application_filler.fill_application_form(
        form_fields, user_profile, job_data, cover_letter
    )
    
    # In a real app, this would either:
    # 1. Submit the application through a scraper
    # 2. Return the values to the extension for form filling
    
    return jsonify({
        'status': 'success',
        'form_values': form_values,
        'cover_letter': cover_letter
    })

@app.route('/api/extension/status', methods=['GET'])
def extension_status():
    """API endpoint for the browser extension to check server status."""
    return jsonify({
        'status': 'online',
        'version': '0.1.0',
        'profile_complete': 'profile' in session,
        'resume_uploaded': 'resume_path' in session
    })

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Handle user settings."""
    if request.method == 'POST':
        settings_data = {
            'use_ai': request.form.get('use_ai') == 'on',
            'api_key': request.form.get('api_key'),
            'auto_apply': request.form.get('auto_apply') == 'on',
            'job_boards': request.form.getlist('job_boards'),
            'min_match_score': request.form.get('min_match_score', 70)
        }
        
        # In a real app, this would be saved to a database
        # For now, we'll just store it in the session
        session['settings'] = settings_data
        
        flash('Settings updated successfully!', 'success')
        return redirect(url_for('dashboard'))
        
    # For GET requests, display the form
    settings_data = session.get('settings', {
        'use_ai': use_ai_api,
        'auto_apply': False,
        'job_boards': ['indeed', 'linkedin'],
        'min_match_score': 70
    })
    
    return render_template('settings.html', settings=settings_data)

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500

def main():
    """Main function to run the Flask application."""
    # Create necessary directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true', 
            host=os.getenv('FLASK_HOST', '0.0.0.0'),
            port=int(os.getenv('FLASK_PORT', 5000)))

if __name__ == '__main__':
    main()