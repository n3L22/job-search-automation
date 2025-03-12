"""
Resume Analyzer Module

This module provides functionality to analyze resumes and compare them with job descriptions.
It includes both local processing and optional AI-powered analysis.
"""

import re
import os
import logging
from typing import Dict, List, Set, Tuple, Optional, Union
import json
from collections import Counter

import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    """
    A class that provides methods for analyzing resumes and comparing them with job descriptions.
    Can operate in local mode or with AI assistance if an API key is provided.
    """
    
    def __init__(self, use_api: bool = False, api_key: Optional[str] = None):
        """
        Initialize the ResumeAnalyzer.
        
        Args:
            use_api: Whether to use external AI API for enhanced analysis
            api_key: API key for external AI service (required if use_api is True)
        """
        self.use_api = use_api
        self.api_key = api_key
        
        # Load a lightweight local model for basic analysis
        try:
            self.local_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            logger.info("Successfully loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            self.local_model = None
            
        # Common skills list - this could be loaded from a file
        self.common_skills = {
            "python", "javascript", "java", "c++", "c#", "ruby", "go", "swift",
            "react", "angular", "vue", "node.js", "express", "django", "flask",
            "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "sqlite",
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
            "git", "agile", "scrum", "kanban", "jira", "ci/cd", "devops",
            "machine learning", "data science", "ai", "nlp", "computer vision",
            "excel", "powerpoint", "word", "google sheets", "tableau", "power bi",
            "communication", "teamwork", "leadership", "problem-solving", "critical thinking"
        }
        
        # Get stopwords for keyword filtering
        self.stopwords = set(stopwords.words('english'))
        
    def analyze_resume(self, resume_text: str, job_description: Optional[str] = None) -> Dict:
        """
        Analyze a resume and optionally compare it with a job description.
        
        Args:
            resume_text: The text content of the resume
            job_description: Optional job description to compare with
            
        Returns:
            Dict containing analysis results
        """
        # Basic analysis that works without AI
        skills_found = self._extract_skills(resume_text)
        education = self._extract_education(resume_text)
        experience_years = self._estimate_experience_years(resume_text)
        
        result = {
            'skills': skills_found,
            'education': education,
            'experience_years': experience_years,
        }
        
        # If job description is provided, perform matching
        if job_description:
            if not self.use_api or not self.api_key:
                # Local matching without API
                match_score, missing_keywords = self._local_matching(resume_text, job_description)
                suggestions = self._generate_basic_suggestions(missing_keywords, job_description)
                
                result.update({
                    'match_score': match_score,
                    'missing_keywords': missing_keywords,
                    'suggestions': suggestions
                })
            else:
                # Use AI API for enhanced analysis
                try:
                    api_results = self._api_analysis(resume_text, job_description)
                    result.update(api_results)
                except Exception as e:
                    logger.error(f"Error in API analysis: {e}")
                    # Fall back to local matching if API fails
                    match_score, missing_keywords = self._local_matching(resume_text, job_description)
                    suggestions = self._generate_basic_suggestions(missing_keywords, job_description)
                    
                    result.update({
                        'match_score': match_score,
                        'missing_keywords': missing_keywords,
                        'suggestions': suggestions,
                        'api_error': str(e)
                    })
        
        return result
    
    def optimize_resume(self, resume_text: str, job_description: str) -> Dict:
        """
        Generate suggestions to optimize a resume for a specific job.
        
        Args:
            resume_text: The text content of the resume
            job_description: The job description to optimize for
            
        Returns:
            Dict containing optimization suggestions
        """
        if not self.use_api or not self.api_key:
            # Use basic pattern matching for optimization
            _, missing_keywords = self._local_matching(resume_text, job_description)
            suggestions = self._generate_basic_suggestions(missing_keywords, job_description)
            
            return {
                'suggestions': suggestions,
                'keywords_to_add': missing_keywords
            }
        else:
            # Use AI for enhanced optimization
            try:
                prompt = f"""
                I need to optimize my resume for this job description. Please help by:
                1. Identifying key skills and qualifications mentioned in the job description
                2. Suggesting specific bullet points I could add to my resume
                3. Identifying skills I should highlight from my current resume
                
                My current resume:
                {resume_text[:2000]}...
                
                Job Description:
                {job_description[:2000]}...
                
                Please format your response as JSON with the following structure:
                {{
                    "skills_to_highlight": ["skill1", "skill2"...],
                    "suggested_bullets": ["bullet1", "bullet2"...],
                    "keywords_to_add": ["keyword1", "keyword2"...]
                }}
                """
                
                response = self._call_ai_api(prompt)
                
                try:
                    # Parse the JSON response
                    optimization_data = json.loads(response)
                    return optimization_data
                except json.JSONDecodeError:
                    # If AI response isn't valid JSON, extract manually
                    skills = re.findall(r'"skills_to_highlight":\s*\[(.*?)\]', response, re.DOTALL)
                    bullets = re.findall(r'"suggested_bullets":\s*\[(.*?)\]', response, re.DOTALL)
                    keywords = re.findall(r'"keywords_to_add":\s*\[(.*?)\]', response, re.DOTALL)
                    
                    return {
                        'skills_to_highlight': [s.strip().strip('"\'') for s in skills[0].split(',') if s.strip()] if skills else [],
                        'suggested_bullets': [b.strip().strip('"\'') for b in bullets[0].split(',') if b.strip()] if bullets else [],
                        'keywords_to_add': [k.strip().strip('"\'') for k in keywords[0].split(',') if k.strip()] if keywords else []
                    }
                    
            except Exception as e:
                logger.error(f"Error in AI optimization: {e}")
                # Fall back to basic optimization
                _, missing_keywords = self._local_matching(resume_text, job_description)
                suggestions = self._generate_basic_suggestions(missing_keywords, job_description)
                
                return {
                    'suggestions': suggestions,
                    'keywords_to_add': missing_keywords,
                    'api_error': str(e)
                }
    
    def _extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using pattern matching against common skills list.
        
        Args:
            text: Text to extract skills from
            
        Returns:
            List of found skills
        """
        text = text.lower()
        found_skills = []
        
        # Find exact matches for multi-word skills
        multi_word_skills = [s for s in self.common_skills if ' ' in s]
        for skill in multi_word_skills:
            if skill in text:
                found_skills.append(skill)
                
        # Find word boundary matches for single-word skills
        single_word_skills = [s for s in self.common_skills if ' ' not in s]
        for skill in single_word_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                found_skills.append(skill)
                
        return found_skills
    
    def _extract_education(self, text: str) -> List[Dict]:
        """
        Extract education information from resume text.
        
        Args:
            text: Resume text
            
        Returns:
            List of education entries with degree and institution
        """
        education = []
        
        # Look for common degree patterns
        degree_patterns = [
            r'(B\.?S\.?|Bachelor of Science|Bachelor\'?s?)(\s+degree)?(\s+in\s+([^\n,.]+))?',
            r'(B\.?A\.?|Bachelor of Arts|Bachelor\'?s?)(\s+degree)?(\s+in\s+([^\n,.]+))?',
            r'(M\.?S\.?|Master of Science|Master\'?s?)(\s+degree)?(\s+in\s+([^\n,.]+))?',
            r'(M\.?B\.?A\.?|Master of Business Administration)',
            r'(Ph\.?D\.?|Doctor of Philosophy)',
            r'(M\.?D\.?|Doctor of Medicine)',
            r'(J\.?D\.?|Juris Doctor)'
        ]
        
        # Look for institutions near degree mentions
        institution_pattern = r'(?:from|at|University|College|Institute|School)(?:\s+of)?\s+([A-Z][^\n,.]+)'
        
        for pattern in degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                degree = match.group(0)
                
                # Look for institution within 50 characters after degree
                context = text[match.end():match.end()+100]
                institution_match = re.search(institution_pattern, context)
                institution = institution_match.group(1) if institution_match else None
                
                education.append({
                    'degree': degree.strip(),
                    'institution': institution.strip() if institution else None
                })
                
        return education
    
    def _estimate_experience_years(self, text: str) -> float:
        """
        Estimate years of experience from resume text.
        
        Args:
            text: Resume text
            
        Returns:
            Estimated years of experience
        """
        # Look for patterns like "X years of experience"
        year_patterns = [
            r'(\d+)(?:\+)?\s+years?\s+(?:of\s+)?experience',
            r'experience\s+(?:of\s+)?(\d+)(?:\+)?\s+years?',
            r'(\d+)(?:\+)?-year\s+experience'
        ]
        
        max_years = 0
        for pattern in year_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                years = int(match.group(1))
                max_years = max(max_years, years)
                
        return max_years
    
    def _local_matching(self, resume: str, job_desc: str) -> Tuple[float, List[str]]:
        """
        Perform local text matching between resume and job description.
        
        Args:
            resume: Resume text
            job_desc: Job description text
            
        Returns:
            Tuple of (match_score, missing_keywords)
        """
        if not self.local_model:
            # Fall back to simple keyword matching if model is not available
            return self._simple_keyword_matching(resume, job_desc)
            
        try:
            # Encode texts
            resume_emb = self.local_model.encode(resume)
            job_emb = self.local_model.encode(job_desc)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([resume_emb], [job_emb])[0][0]
            match_score = similarity * 100
            
            # Extract keywords from job that don't appear in resume
            job_tokens = set(nltk.word_tokenize(job_desc.lower()))
            resume_tokens = set(nltk.word_tokenize(resume.lower()))
            
            # Filter common words and short words
            job_keywords = {word for word in job_tokens 
                          if word not in self.stopwords 
                          and len(word) > 3 
                          and word.isalpha()}
            
            resume_keywords = {word for word in resume_tokens 
                             if word not in self.stopwords 
                             and len(word) > 3 
                             and word.isalpha()}
            
            # Find keywords that are in job description but not in resume
            missing_keywords = list(job_keywords - resume_keywords)
            
            # Count frequency of keywords in job description to prioritize important ones
            job_desc_counts = Counter(word.lower() for word in nltk.word_tokenize(job_desc)
                                    if word.lower() in missing_keywords)
            
            # Sort missing keywords by frequency
            missing_keywords = sorted(missing_keywords, 
                                    key=lambda k: job_desc_counts.get(k, 0), 
                                    reverse=True)
            
            return match_score, missing_keywords[:10]  # Return top 10 missing keywords
            
        except Exception as e:
            logger.error(f"Error in local matching: {e}")
            return self._simple_keyword_matching(resume, job_desc)
    
    def _simple_keyword_matching(self, resume: str, job_desc: str) -> Tuple[float, List[str]]:
        """
        Perform simple keyword matching as a fallback method.
        
        Args:
            resume: Resume text
            job_desc: Job description text
            
        Returns:
            Tuple of (match_score, missing_keywords)
        """
        # Extract keywords
        job_tokens = nltk.word_tokenize(job_desc.lower())
        resume_tokens = nltk.word_tokenize(resume.lower())
        
        # Filter stopwords and short words
        job_keywords = [word for word in job_tokens 
                       if word not in self.stopwords 
                       and len(word) > 3 
                       and word.isalpha()]
        
        resume_keywords = [word for word in resume_tokens 
                          if word not in self.stopwords 
                          and len(word) > 3 
                          and word.isalpha()]
        
        # Count unique keywords
        unique_job_keywords = set(job_keywords)
        unique_resume_keywords = set(resume_keywords)
        
        # Calculate overlap
        common_keywords = unique_job_keywords.intersection(unique_resume_keywords)
        
        # Match score based on percentage of job keywords found in resume
        if len(unique_job_keywords) > 0:
            match_score = len(common_keywords) / len(unique_job_keywords) * 100
        else:
            match_score = 0
            
        # Missing keywords
        missing_keywords = list(unique_job_keywords - unique_resume_keywords)
        
        # Count frequency in job description
        job_keyword_counts = Counter(job_keywords)
        
        # Sort by frequency
        missing_keywords.sort(key=lambda k: job_keyword_counts[k], reverse=True)
        
        return match_score, missing_keywords[:10]
    
    def _generate_basic_suggestions(self, missing_keywords: List[str], job_desc: str) -> List[str]:
        """
        Generate basic suggestions for resume improvement based on missing keywords.
        
        Args:
            missing_keywords: List of keywords missing from resume
            job_desc: Job description text
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        if missing_keywords:
            # Group similar keywords
            grouped_keywords = self._group_similar_keywords(missing_keywords)
            
            for group in grouped_keywords[:3]:  # Limit to top 3 groups
                keywords_str = ", ".join(group)
                suggestions.append(f"Consider adding these keywords to your resume: {keywords_str}")
                
        # Add general improvement suggestions
        suggestions.append("Quantify your achievements with numbers and percentages")
        suggestions.append("Use action verbs at the beginning of bullet points")
        suggestions.append("Tailor your resume to highlight experience relevant to this position")
        
        # Look for specific requirements in job description
        if re.search(r'years of experience', job_desc, re.IGNORECASE):
            suggestions.append("Clearly state your years of experience in relevant areas")
            
        if re.search(r'team|collaborate|cross-functional', job_desc, re.IGNORECASE):
            suggestions.append("Emphasize teamwork and collaboration examples")
            
        if re.search(r'lead|manage|supervise', job_desc, re.IGNORECASE):
            suggestions.append("Highlight leadership and management experiences")
            
        return suggestions
    
    def _group_similar_keywords(self, keywords: List[str]) -> List[List[str]]:
        """
        Group similar keywords together.
        
        Args:
            keywords: List of keywords to group
            
        Returns:
            List of lists, where each inner list contains similar keywords
        """
        if not keywords:
            return []
            
        # Simple grouping based on common prefixes
        groups = []
        used = set()
        
        for keyword in keywords:
            if keyword in used:
                continue
                
            # Find similar keywords
            similar = [keyword]
            used.add(keyword)
            
            for other in keywords:
                if other not in used:
                    # Check if words share a prefix
                    prefix_len = min(len(keyword), len(other), 5)
                    if keyword[:prefix_len] == other[:prefix_len]:
                        similar.append(other)
                        used.add(other)
                        
            groups.append(similar)
            
        return groups
    
    def _api_analysis(self, resume: str, job_desc: str) -> Dict:
        """
        Use external AI API for enhanced resume analysis.
        
        Args:
            resume: Resume text
            job_desc: Job description text
            
        Returns:
            Dict containing API analysis results
        """
        if not self.use_api or not self.api_key:
            raise ValueError("API analysis requested but API usage is disabled or key is missing")
            
        prompt = f"""
        I need to analyze how well my resume matches this job description. Please provide:
        1. A match score from 0-100
        2. Key missing keywords and skills from my resume
        3. Specific suggestions to improve my resume for this job
        
        My resume:
        {resume[:3000]}...
        
        Job Description:
        {job_desc[:3000]}...
        
        Please format your response as JSON with the following structure:
        {{
            "match_score": number,
            "missing_keywords": ["keyword1", "keyword2"...],
            "suggestions": ["suggestion1", "suggestion2"...]
        }}
        """
        
        response = self._call_ai_api(prompt)
        
        try:
            # Try to parse the response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # If not valid JSON, extract data using regex
            match_score_match = re.search(r'"match_score":\s*(\d+)', response)
            match_score = float(match_score_match.group(1)) if match_score_match else 50.0
            
            keywords = re.findall(r'"missing_keywords":\s*\[(.*?)\]', response, re.DOTALL)
            suggestions = re.findall(r'"suggestions":\s*\[(.*?)\]', response, re.DOTALL)
            
            return {
                'match_score': match_score,
                'missing_keywords': [k.strip().strip('"\'') for k in keywords[0].split(',') if k.strip()] if keywords else [],
                'suggestions': [s.strip().strip('"\'') for s in suggestions[0].split(',') if s.strip()] if suggestions else []
            }
    
    def _call_ai_api(self, prompt: str) -> str:
        """
        Call external AI API with prompt.
        
        Args:
            prompt: Prompt text to send to API
            
        Returns:
            API response text
        """
        if not self.api_key:
            raise ValueError("API key is required for API calls")
            
        try:
            # OpenAI API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",  # Use cheaper model to reduce costs
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,  # Low temperature for more predictable results
                "max_tokens": 500  # Limit token usage
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise Exception(f"API returned error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error calling AI API: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # This is just for testing
    analyzer = ResumeAnalyzer(use_api=False)
    
    test_resume = """
    John Doe
    Software Engineer
    
    Experience:
    - Senior Developer, ABC Tech (2018-Present)
      * Led development of a cloud-based analytics platform using Python and AWS
      * Implemented CI/CD pipelines reducing deployment time by 40%
      * Mentored junior developers and conducted code reviews
    
    - Software Developer, XYZ Solutions (2015-2018)
      * Developed RESTful APIs using Django and Flask
      * Optimized database queries resulting in 30% performance improvement
      * Contributed to open-source projects
    
    Education:
    - M.S. Computer Science, University of Technology (2015)
    - B.S. Computer Science, State University (2013)
    
    Skills:
    Python, JavaScript, React, AWS, Docker, Kubernetes, CI/CD, RESTful APIs, 
    Django, Flask, SQL, NoSQL, Git, Agile methodologies
    """
    
    test_job = """
    Senior Software Engineer
    
    We are looking for a Senior Software Engineer to join our dynamic team.
    
    Required Skills:
    - 5+ years of experience in software development
    - Strong proficiency in Python and JavaScript
    - Experience with web frameworks (Django, Flask)
    - Knowledge of cloud services (AWS, Azure, or GCP)
    - Experience with containerization technologies (Docker, Kubernetes)
    - Understanding of CI/CD principles
    
    Responsibilities:
    - Design and develop scalable and maintainable software
    - Collaborate with cross-functional teams
    - Mentor junior developers
    - Participate in code reviews
    - Troubleshoot and debug applications
    
    Nice to have:
    - Experience with frontend frameworks (React, Angular)
    - Machine learning experience
    - Contributions to open-source projects
    """
    
    # Analyze resume without job matching
    basic_analysis = analyzer.analyze_resume(test_resume)
    print("Skills found:", basic_analysis["skills"])
    print("Education:", basic_analysis["education"])
    print("Experience years:", basic_analysis["experience_years"])
    print("\n")
    
    # Analyze resume with job matching
    job_match = analyzer.analyze_resume(test_resume, test_job)
    print(f"Job match score: {job_match['match_score']:.1f}%")
    print("Missing keywords:", job_match["missing_keywords"])
    print("\nSuggestions for improvement:")
    for suggestion in job_match["suggestions"]:
        print(f"- {suggestion}")