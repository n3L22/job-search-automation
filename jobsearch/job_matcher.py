"""
Job Matcher Module

This module provides functionality to match job listings with user profiles.
It includes both local processing and optional AI-powered matching.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
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

class JobMatcher:
    """
    A class that provides methods for matching job listings with user profiles.
    Can operate in local mode or with AI assistance if an API key is provided.
    """
    
    def __init__(self, use_api: bool = False, api_key: Optional[str] = None):
        """
        Initialize the JobMatcher.
        
        Args:
            use_api: Whether to use external AI API for enhanced matching
            api_key: API key for external AI service (required if use_api is True)
        """
        self.use_api = use_api
        self.api_key = api_key
        
        # Load a lightweight local model for semantic matching
        try:
            self.local_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            logger.info("Successfully loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            self.local_model = None
            
        # Common skills list for extraction
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
        
        # Cache for extracted skills from job descriptions
        self.job_skills_cache = {}
        
    def match_jobs(self, user_profile: Dict, job_listings: List[Dict]) -> List[Dict]:
        """
        Match a user profile with a list of job listings.
        
        Args:
            user_profile: User profile containing skills, experience, etc.
            job_listings: List of job listings to match against
            
        Returns:
            List of job listings with match scores and details
        """
        if not job_listings:
            return []
            
        # Extract user details
        user_skills = set(user_profile.get('skills', []))
        user_experience = user_profile.get('experience', '')
        user_title = user_profile.get('desired_title', '')
        user_years_experience = user_profile.get('experience_years', 0)
        
        # Create combined user text for semantic matching
        user_text = f"{user_title} {user_experience}"
        
        # Encode user text for semantic matching if model is available
        user_embedding = None
        if self.local_model:
            try:
                user_embedding = self.local_model.encode(user_text)
            except Exception as e:
                logger.error(f"Error encoding user profile: {e}")
                
        matches = []
        for job in job_listings:
            # Extract job details
            job_title = job.get('title', '')
            job_desc = job.get('description', '')
            job_company = job.get('company', '')
            job_location = job.get('location', '')
            
            # Extract job requirements
            job_skills = self._extract_job_skills(job_desc)
            required_experience = self._extract_required_experience(job_desc)
            
            # Calculate skill match
            skill_match = 0
            if job_skills:
                skill_match = len(user_skills.intersection(job_skills)) / len(job_skills) * 100
                
            # Calculate semantic match
            semantic_match = 0
            if self.local_model and user_embedding is not None:
                try:
                    job_text = f"{job_title} {job_desc}"
                    job_embedding = self.local_model.encode(job_text)
                    semantic_match = cosine_similarity([user_embedding], [job_embedding])[0][0] * 100
                except Exception as e:
                    logger.error(f"Error calculating semantic match: {e}")
            
            # Title match (exact or partial)
            title_match = 0
            if user_title and job_title:
                # Tokenize titles
                user_title_tokens = set(nltk.word_tokenize(user_title.lower()))
                job_title_tokens = set(nltk.word_tokenize(job_title.lower()))
                
                # Remove stopwords
                user_title_tokens = {t for t in user_title_tokens if t not in self.stopwords}
                job_title_tokens = {t for t in job_title_tokens if t not in self.stopwords}
                
                # Calculate overlap
                if job_title_tokens:
                    title_match = len(user_title_tokens.intersection(job_title_tokens)) / len(job_title_tokens) * 100
            
            # Experience match
            experience_match = 100
            if required_experience > 0 and user_years_experience < required_experience:
                # Decrease match score based on experience gap
                experience_gap = required_experience - user_years_experience
                experience_match = max(0, 100 - (experience_gap * 20))  # 20% penalty per year of missing experience
            
            # Calculate overall match score (weighted)
            match_score = (
                skill_match * 0.4 +         # Skills are important
                semantic_match * 0.3 +      # Overall semantic match
                title_match * 0.2 +         # Title match
                experience_match * 0.1      # Experience match
            )
            
            # Store missing skills
            missing_skills = list(job_skills - user_skills)
            
            # Create match object
            match = {
                'job': job,
                'match_score': round(match_score, 1),
                'skill_match': round(skill_match, 1),
                'semantic_match': round(semantic_match, 1),
                'title_match': round(title_match, 1),
                'experience_match': round(experience_match, 1),
                'missing_skills': missing_skills,
                'required_experience': required_experience
            }
            
            matches.append(match)
        
        # Sort by match score (descending)
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        # If API use is enabled, enhance top results
        if self.use_api and self.api_key:
            try:
                # Only analyze top results to save costs
                top_matches = matches[:5]
                for i, match in enumerate(top_matches):
                    enhanced_match = self._enhance_match_with_api(user_profile, match['job'])
                    matches[i].update(enhanced_match)
            except Exception as e:
                logger.error(f"Error enhancing matches with API: {e}")
        
        return matches
    
    def _extract_job_skills(self, job_desc: str) -> Set[str]:
        """
        Extract skills mentioned in a job description.
        
        Args:
            job_desc: Job description text
            
        Returns:
            Set of skills found in the description
        """
        # Check cache first
        cache_key = hash(job_desc)
        if cache_key in self.job_skills_cache:
            return self.job_skills_cache[cache_key]
            
        skills = set()
        
        # Extract skills using pattern matching
        job_desc_lower = job_desc.lower()
        
        # Find multi-word skills
        multi_word_skills = {s for s in self.common_skills if ' ' in s}
        for skill in multi_word_skills:
            if skill in job_desc_lower:
                skills.add(skill)
                
        # Find single-word skills with word boundary check
        single_word_skills = {s for s in self.common_skills if ' ' not in s}
        for skill in single_word_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', job_desc_lower):
                skills.add(skill)
        
        # Look for skill sections in job description
        skill_section_pattern = r'(?:skills|requirements|qualifications)(?:\s+required)?(?:\s*:|(?:\s+include))'
        skill_sections = re.split(skill_section_pattern, job_desc_lower, flags=re.IGNORECASE)
        
        if len(skill_sections) > 1:
            # Take the section after the "skills" header
            skill_section = skill_sections[1]
            
            # Look for bullet points or list items
            bullet_items = re.findall(r'(?:•|\*|\-|\d+\.)\s*([^\n•\*\-\d\.]+)', skill_section)
            
            for item in bullet_items:
                # Check if item contains any common skill
                for skill in self.common_skills:
                    if skill in item.lower():
                        skills.add(skill)
        
        # Cache the result
        self.job_skills_cache[cache_key] = skills
        return skills
    
    def _extract_required_experience(self, job_desc: str) -> int:
        """
        Extract years of experience required from job description.
        
        Args:
            job_desc: Job description text
            
        Returns:
            Required years of experience (0 if not specified)
        """
        # Look for patterns like "X years of experience"
        experience_patterns = [
            r'(\d+)(?:\+)?\s+years?\s+(?:of\s+)?experience',
            r'experience\s+(?:of\s+)?(\d+)(?:\+)?\s+years?',
            r'(\d+)(?:\+)?-year\s+experience',
            r'minimum\s+(?:of\s+)?(\d+)(?:\+)?\s+years?',
            r'at\s+least\s+(\d+)(?:\+)?\s+years?'
        ]
        
        for pattern in experience_patterns:
            matches = list(re.finditer(pattern, job_desc, re.IGNORECASE))
            if matches:
                # Use the first match
                return int(matches[0].group(1))
                
        return 0  # Default if no requirement found
    
    def _enhance_match_with_api(self, user_profile: Dict, job: Dict) -> Dict:
        """
        Use external AI API for enhanced job matching.
        
        Args:
            user_profile: User profile
            job: Job listing
            
        Returns:
            Dictionary with enhanced matching details
        """
        if not self.use_api or not self.api_key:
            raise ValueError("API enhancement requested but API usage is disabled or key is missing")
            
        # Extract relevant user details
        user_skills = user_profile.get('skills', [])
        user_experience = user_profile.get('experience', '')
        user_title = user_profile.get('desired_title', '')
        
        # Extract job details
        job_title = job.get('title', '')
        job_desc = job.get('description', '')
        
        prompt = f"""
        I need to analyze how well this job matches my profile. Please provide:
        1. An honest match score from 0-100
        2. Key skills I'm missing for this job
        3. Specific reasons why this job might be a good fit for me
        4. Any potential challenges I might face in this role
        
        My profile:
        - Desired title: {user_title}
        - Skills: {', '.join(user_skills)}
        - Experience summary: {user_experience[:500]}...
        
        Job details:
        - Title: {job_title}
        - Description: {job_desc[:1000]}...
        
        Please format your response as JSON with the following structure:
        {{
            "ai_match_score": number,
            "missing_skills": ["skill1", "skill2"...],
            "fit_reasons": ["reason1", "reason2"...],
            "challenges": ["challenge1", "challenge2"...]
        }}
        """
        
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
                response_text = response.json()["choices"][0]["message"]["content"]
                
                try:
                    # Try to parse as JSON
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    # If not valid JSON, extract using regex
                    score_match = re.search(r'"ai_match_score":\s*(\d+)', response_text)
                    ai_score = float(score_match.group(1)) if score_match else 50.0
                    
                    missing = re.findall(r'"missing_skills":\s*\[(.*?)\]', response_text, re.DOTALL)
                    reasons = re.findall(r'"fit_reasons":\s*\[(.*?)\]', response_text, re.DOTALL)
                    challenges = re.findall(r'"challenges":\s*\[(.*?)\]', response_text, re.DOTALL)
                    
                    return {
                        'ai_match_score': ai_score,
                        'missing_skills': [s.strip().strip('"\'') for s in missing[0].split(',') if s.strip()] if missing else [],
                        'fit_reasons': [r.strip().strip('"\'') for r in reasons[0].split(',') if r.strip()] if reasons else [],
                        'challenges': [c.strip().strip('"\'') for c in challenges[0].split(',') if c.strip()] if challenges else []
                    }
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {
                    'ai_match_score': None,
                    'api_error': f"API returned error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error calling AI API: {e}")
            return {
                'ai_match_score': None,
                'api_error': str(e)
            }
    
    def filter_jobs(self, matches: List[Dict], min_score: float = 70.0) -> List[Dict]:
        """
        Filter job matches based on score.
        
        Args:
            matches: List of job matches with scores
            min_score: Minimum score to include
            
        Returns:
            Filtered list of job matches
        """
        return [match for match in matches if match['match_score'] >= min_score]
    
    def get_top_matches(self, matches: List[Dict], limit: int = 10) -> List[Dict]:
        """
        Get top N matches.
        
        Args:
            matches: List of job matches with scores
            limit: Maximum number of matches to return
            
        Returns:
            Top N job matches
        """
        return matches[:limit]

# Example usage
if __name__ == "__main__":
    # This is just for testing
    matcher = JobMatcher(use_api=False)
    
    user_profile = {
        'skills': ['python', 'django', 'javascript', 'react', 'sql'],
        'experience': 'Developed web applications using Django and React',
        'desired_title': 'Full Stack Developer',
        'experience_years': 3
    }
    
    test_job = {
        'title': 'Senior Full Stack Developer',
        'company': 'Tech Corp',
        'location': 'Remote',
        'description': '''
        We are looking for a Full Stack Developer with at least 5 years of experience.
        
        Required Skills:
        • Python
        • Django
        • JavaScript
        • React
        • SQL
        • Docker
        • AWS
        
        Responsibilities:
        • Develop web applications
        • Work with cross-functional teams
        • Deploy and maintain applications
        '''
    }
    
    matches = matcher.match_jobs(user_profile, [test_job])
    print(matches[0]['match_score'])
    print(matches[0]['missing_skills'])