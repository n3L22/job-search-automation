Automated Job Search Assistant
An AI-powered job search extension that automates the process of finding and applying to relevant jobs.
Features

🤖 Automated job application for matching positions
🔍 Intelligent job matching based on your resume and preferences
📝 AI-enhanced resume analysis and optimization
✉️ Automatic cover letter generation
📊 Application tracking and recommendations

Installation
Prerequisites

Python 3.8+
Google Chrome or Firefox
Git

Setup

Clone the repository:

git clone https://github.com/yourusername/job-search-automation.git
cd job-search-automation

Create a virtual environment:

python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Configure your settings:

cp config/default_config.yaml config/config.yaml
# Edit config/config.yaml with your personal preferences

Install the browser extension:

Chrome: Open chrome://extensions/, enable Developer Mode, click "Load unpacked", and select the "extension" folder
Firefox: Open about:debugging, click "This Firefox", click "Load Temporary Add-on", and select manifest.json in the "extension" folder


Start the local server:

python webapp/app.py

Open your browser to http://localhost:5000 to access the web interface

Using the Extension

Fill out your profile in the web interface
Upload your resume
Specify target job boards and websites
Set your job preferences
Click "Start Automation" in the extension

AI Features (Optional)
This extension can work with or without AI features. To enable AI-powered features:

Obtain an API key from OpenAI or another supported provider
Add your API key in the settings page
Enable AI features in the extension options

Without an API key, the extension will still function using pattern matching and basic algorithms.
Privacy
Your resume and personal data stay on your local machine. If you enable AI features, only anonymized job descriptions and resume sections are sent to the API for analysis.
Development
Running Tests
pytest
Building the Extension
cd extension
npm install
npm run build
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.