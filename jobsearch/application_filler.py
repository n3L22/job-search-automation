"""
Application Filler Module

This module provides functionality to generate content for job applications
and automate form filling. It includes both template-based and optional
AI-powered content generation.
"""

import re
import logging
import json
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

class ApplicationFiller:
    """
    A class that provides methods for generating application content and filling out forms.
    Can operate in template mode or with AI assistance if an API key is provided.
    """
    
    def __init__(self, use_api: bool = False, api_key: Optional[str] = None):
        """
        Initialize the ApplicationFiller.
        
        Args:
            use_api: Whether to use external AI API for enhanced content
            api_key: API key for external AI service (required if use_api is True)
        """
        self.use_api = use_api
        self.api_key = api_key
        
        # Load templates
        self.templates = self._load_templates()
        
        # Cache for generated content to avoid redundant API calls
        self.content_cache = {}
        
    def _load_templates(self) -> Dict:
        """
        Load templates for different application components.
        
        Returns:
            Dictionary of templates
        """
        # Common templates for various application sections
        templates = {
            # Cover letter templates
            "cover_letter": [
                """Dear {hiring_manager},

I am writing to express my interest in the {job_title} position at {company}. With {experience_years} years of experience in {field}, I believe I have the skills and experience necessary to excel in this role.

{experience_paragraph}

What attracts me to {company} is {company_appeal}. I am particularly excited about {job_appeal} and believe my background in {relevant_skill_1} and {relevant_skill_2} would allow me to make significant contributions to your team.

I welcome the opportunity to discuss how my background, skills, and experiences would benefit {company}. Thank you for your consideration, and I look forward to hearing from you.

Sincerely,
{name}""",

                """Dear Hiring Team at {company},

I am excited to apply for the {job_title} position I found on {job_source}. As a {current_role} with {experience_years} years of experience and a strong background in {relevant_skill_1} and {relevant_skill_2}, I am confident in my ability to make valuable contributions to your team.

{experience_paragraph}

{company}'s {company_appeal} aligns perfectly with my professional interests and values. I am particularly drawn to {job_appeal} mentioned in the job description.

Thank you for considering my application. I am looking forward to the possibility of discussing how my skills and experiences align with your needs at {company}.

Best regards,
{name}"""
            ],
            
            # Experience paragraph templates
            "experience_paragraph": [
                "In my current role at {current_company}, I have successfully {achievement_1}. Prior to this, at {previous_company}, I {achievement_2}. Throughout my career, I have developed strong skills in {skill_area} and have consistently {consistent_achievement}.",
                
                "Most recently as a {current_role} at {current_company}, I have been responsible for {responsibility_1} and {responsibility_2}. Previously, I {achievement_1} at {previous_company}, resulting in {achievement_result}. My experience has prepared me well for the challenges of the {job_title} position.",
                
                "My professional experience includes {experience_years} years in {field}, with particular focus on {relevant_skill_1} and {relevant_skill_2}. At {current_company}, I {achievement_1}, and at {previous_company}, I {achievement_2}."
            ],
            
            # Common application questions templates
            "why_interested": [
                "I am interested in this position because it aligns perfectly with my background in {relevant_skill_1} and my career goal of {career_goal}. {company}'s {company_appeal} is particularly exciting to me, and I believe I can contribute significantly to {job_aspect} mentioned in the job description.",
                
                "This role interests me because it combines my experience in {relevant_skill_1} with the opportunity to develop skills in {skill_to_develop}. I've been following {company}'s work in {company_focus}, and I'm excited about the possibility of contributing to {job_aspect}."
            ],
            
            "strengths": [
                "My greatest strengths include my expertise in {relevant_skill_1}, my ability to {strength_1}, and my track record of {achievement_1}. I also excel at {strength_2}, which has helped me {achievement_result} in my previous roles.",
                
                "I bring three key strengths to this role: First, my technical expertise in {relevant_skill_1} and {relevant_skill_2}. Second, my ability to {strength_1} even under tight deadlines. Third, my {strength_2} skills, which have enabled me to {achievement_result}."
            ],
            
            "weaknesses": [
                "One area I continually work to improve is {weakness}. I've addressed this by {improvement_strategy}, which has resulted in {improvement_result}. I believe in ongoing professional development and am always looking for opportunities to strengthen my skills.",
                
                "I sometimes find {weakness} challenging. To overcome this, I've {improvement_strategy}, which has significantly improved my performance in this area. I view areas for growth as opportunities to become a more well-rounded professional."
            ],
            
            "salary_expectations": [
                "Based on my research of similar roles in {location} and my {experience_years} years of experience in {field}, I'm seeking a salary in the range of {salary_range}. However, I'm open to discussing the comprehensive compensation package that aligns with the responsibilities of the role.",
                
                "I'm looking for a compensation package in the range of {salary_range}, which I believe reflects the value I can bring to this position given my experience in {relevant_skill_1} and track record of {achievement_result}. I'm flexible and open to discussing how my expectations align with your budget."
            ]
        }
        
        return templates
        
    def generate_cover_letter(self, user_profile: Dict, job_listing: Dict) -> str:
        """
        Generate a cover letter based on user profile and job listing.
        
        Args:
            user_profile: User profile containing experience, skills, etc.
            job_listing: Job listing details
            
        Returns:
            Generated cover letter text
        """
        # Generate cache key
        cache_key = f"cover_letter_{hash(str(user_profile))}-{hash(str(job_listing))}"
        if cache_key in self.content_cache:
            return self.content_cache[cache_key]
        
        if self.use_api and self.api_key:
            try:
                cover_letter = self._generate_api_cover_letter(user_profile, job_listing)
                self.content_cache[cache_key] = cover_letter
                return cover_letter
            except requests.RequestException as req_err:
        # For HTTP request-specific errors where response is available
                if hasattr(req_err, 'response'):
                    logger.error(f"API error: {req_err.response.status_code} - {req_err.response.text}")
                    raise Exception(f"API returned error: {req_err.response.status_code}")
                else:
                    logger.error(f"Request error: {req_err}")
                    raise Exception("API request failed") from req_err
            except Exception as e:
                logger.error(f"Error generating cover letter with API: {e}")
            # Fall back to template approach instead of raising
                logger.info("Falling back to template-based cover letter")
    
    # Use template-based approach (either as default or fallback)
        cover_letter = self._generate_template_cover_letter(user_profile, job_listing)
        self.content_cache[cache_key] = cover_letter
        return cover_letter
            
    def answer_application_question(self, question: str, user_profile: Dict, job_listing: Dict) -> str:
        """
        Generate an answer to a common application question.
        
        Args:
            question: The application question
            user_profile: User profile containing experience, skills, etc.
            job_listing: Job listing details
            
        Returns:
            Generated answer text
        """
        # Generate cache key
        cache_key = f"question_{hash(question)}_{hash(str(user_profile))}-{hash(str(job_listing))}"
        if cache_key in self.content_cache:
            return self.content_cache[cache_key]
        
        # Classify question type
        question_type = self._classify_question(question)
        
        if self.use_api and self.api_key:
            try:
                answer = self._generate_api_answer(question, question_type, user_profile, job_listing)
                self.content_cache[cache_key] = answer
                return answer
            except Exception as e:
                logger.error(f"Error generating answer with API: {e}")
                # Fall back to template approach
                logger.info("Falling back to template-based answer")
        
        # Use template-based approach
        answer = self._fill_template_answer(question_type, user_profile, job_listing)
        self.content_cache[cache_key] = answer
        return answer
    
    def _classify_question(self, question: str) -> str:
        """
        Classify a question into a known category.
        
        Args:
            question: The question text
            
        Returns:
            Category of the question (or 'other' if unknown)
        """
        question_lower = question.lower()
        
        # Common question patterns
        patterns = {
            "why_interested": [
                r'why .* interested',
                r'why .* apply',
                r'why .* this (position|role|job)',
                r'what interests you',
                r'why .* want .* work'
            ],
            "strengths": [
                r'strengths',
                r'what are you good at',
                r'key skills',
                r'greatest .* strength',
                r'what .* do well'
            ],
            "weaknesses": [
                r'weaknesses',
                r'areas .* improve',
                r'shortcomings',
                r'challenges .* face',
                r'what .* struggle with'
            ],
            "salary_expectations": [
                r'salary',
                r'compensation',
                r'pay .* expect',
                r'desired .* salary',
                r'expected .* compensation'
            ]
        }
        
        # Check each pattern
        for category, category_patterns in patterns.items():
            for pattern in category_patterns:
                if re.search(pattern, question_lower):
                    return category
        
        return "other"  # Default if no match
    
    def _fill_template_answer(self, question_type: str, user_profile: Dict, job_listing: Dict) -> str:
        """
        Fill in a template answer for a given question type.
        
        Args:
            question_type: Type of question
            user_profile: User profile containing experience, skills, etc.
            job_listing: Job listing details
            
        Returns:
            Template-based answer
        """
        # Check if we have a template for this question type
        if question_type not in self.templates:
            # Generic response for unknown questions
            return "I would be happy to discuss this further in an interview. My background in " + \
                   user_profile.get('field', 'this field') + " has prepared me well for this opportunity."
        
        # Select a random template for this question type
        template = random.choice(self.templates[question_type])
        
        # Extract user details
        skills = user_profile.get('skills', [])
        experience_years = user_profile.get('experience_years', '5+')
        location = user_profile.get('location', 'this area')
        
        # Extract job details
        company = job_listing.get('company', 'the company')
        job_title = job_listing.get('title', 'the position')
        
        # Prepare field values
        field_values = {
            "company": company,
            "job_title": job_title,
            "experience_years": str(experience_years),
            "field": user_profile.get('field', 'the industry'),
            "location": location,
            "career_goal": user_profile.get('career_goal', 'growing in this field'),
            "strength_1": user_profile.get('strength_1', 'solve complex problems'),
            "strength_2": user_profile.get('strength_2', 'collaborate effectively with teams'),
            "weakness": user_profile.get('weakness', 'balancing multiple priorities'),
            "improvement_strategy": user_profile.get('improvement_strategy', 'implemented structured planning systems'),
            "improvement_result": user_profile.get('improvement_result', 'better time management'),
            "salary_range": user_profile.get('salary_range', '$X-$Y'),
            "achievement_1": user_profile.get('achievement_1', 'achieved significant results'),
            "achievement_result": user_profile.get('achievement_result', 'improved efficiency'),
            "company_appeal": job_listing.get('company_appeal', 'reputation in the industry'),
            "company_focus": job_listing.get('company_focus', 'innovative projects'),
            "job_aspect": job_listing.get('job_aspect', 'challenging projects'),
            "skill_to_develop": job_listing.get('skill_to_develop', 'new skills')
        }
        
        # Fill in skills if available
        if skills:
            field_values["relevant_skill_1"] = skills[0] if len(skills) > 0 else "relevant skills"
            field_values["relevant_skill_2"] = skills[1] if len(skills) > 1 else "additional expertise"
        else:
            field_values["relevant_skill_1"] = "relevant skills"
            field_values["relevant_skill_2"] = "additional expertise"
        
        # Fill in template
        answer = template.format(**field_values)
        
        return answer
    
    def _generate_api_answer(self, question: str, question_type: str, user_profile: Dict, job_listing: Dict) -> str:
        """
        Generate an answer to a question using AI API.
        
        Args:
            question: The question text
            question_type: Classified question type
            user_profile: User profile containing experience, skills, etc.
            job_listing: Job listing details
            
        Returns:
            AI-generated answer
        """
        if not self.use_api or not self.api_key:
            raise ValueError("API generation requested but API usage is disabled or key is missing")
            
        # Extract user details
        skills = user_profile.get('skills', [])
        experience_years = user_profile.get('experience_years', '5+')
        experience = user_profile.get('experience', 'Professional experience')
        
        # Extract job details
        company = job_listing.get('company', 'the company')
        job_title = job_listing.get('title', 'the position')
        job_description = job_listing.get('description', 'the role')
        
        prompt = f"""
        As a job applicant for a {job_title} position at {company}, write a concise, professional answer to this application question:
        
        "{question}"
        
        About me:
        - Years of experience: {experience_years}
        - Skills: {', '.join(skills)}
        - Experience: {experience[:300]}
        
        Job details:
        - Title: {job_title}
        - Company: {company}
        - Description: {job_description[:300]}
        
        Keep the answer concise (under 200 words), professional, and tailored to the specific job.
        Highlight relevant skills and experiences that match the job requirements.
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
                "temperature": 0.7,  # More creativity for answers
                "max_tokens": 300  # Limit token usage
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                return answer.strip()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise Exception(f"API returned error: {response.status_code}")
        except Exception as e:
            logger.error(f"Error calling AI API: {e}")
            raise
    
    def fill_application_form(self, form_fields: Dict, user_profile: Dict, job_listing: Dict, 
                             cover_letter: Optional[str] = None) -> Dict:
        """
        Generate values for application form fields.
        
        Args:
            form_fields: Dictionary of form field names and types
            user_profile: User profile data
            job_listing: Job listing details
            cover_letter: Optional pre-generated cover letter
            
        Returns:
            Dictionary mapping form fields to values
        """
        form_values = {}
        
        # Map common field names to user profile data
        common_field_mappings = {
            # Personal information
            "name": "name",
            "first_name": "first_name",
            "last_name": "last_name",
            "email": "email",
            "phone": "phone",
            "address": "address",
            "city": "city",
            "state": "state",
            "zip": "zip_code",
            "country": "country",
            
            # Professional information
            "current_company": "current_company",
            "current_title": "current_role",
            "years_experience": "experience_years",
            "education": "education",
            "linkedin": "linkedin",
            "portfolio": "portfolio",
            "github": "github",
            "website": "website",
            
            # Application specific
            "resume": "resume_path",
            "available_start_date": "start_date"
        }
        
        # Fill in basic fields from user profile
        for field_name, field_type in form_fields.items():
            field_name_lower = field_name.lower().replace(" ", "_").replace("-", "_")
            
            # Check for direct mappings
            if field_name_lower in common_field_mappings:
                profile_key = common_field_mappings[field_name_lower]
                if profile_key in user_profile:
                    form_values[field_name] = user_profile[profile_key]
                    continue
            
            # Handle special field types
            if "cover_letter" in field_name_lower or field_type == "textarea" and "cover" in field_name_lower:
                if cover_letter:
                    form_values[field_name] = cover_letter
                else:
                    form_values[field_name] = self.generate_cover_letter(user_profile, job_listing)
                continue
                
            # Generate answers for question fields
            if field_type == "textarea" or "question" in field_name_lower or "explain" in field_name_lower:
                form_values[field_name] = self.answer_application_question(field_name, user_profile, job_listing)
                continue
                
            # Handle specific field patterns
            if "salary" in field_name_lower or "compensation" in field_name_lower:
                form_values[field_name] = user_profile.get("salary_expectation", "")
                continue
                
            if "start_date" in field_name_lower or "available" in field_name_lower:
                # Default to 2 weeks from now
                if "start_date" in user_profile:
                    form_values[field_name] = user_profile["start_date"]
                else:
                    # Format as MM/DD/YYYY
                    form_values[field_name] = (datetime.now().replace(day=datetime.now().day + 14)).strftime("%m/%d/%Y")
                continue
        
        return form_values

if __name__ == "__main__":
    # This is just for testing
    filler = ApplicationFiller(use_api=False)
    
    user_profile = {
        'name': 'John Doe',
        'email': 'john.doe@example.com',
        'phone': '555-123-4567',
        'current_role': 'Software Developer',
        'skills': ['Python', 'Django', 'JavaScript', 'React'],
        'experience_years': 3,
        'field': 'web development',
        'current_company': 'Tech Company Inc.',
        'previous_company': 'Startup Ltd.',
        'achievement_1': 'developed a new feature that increased user engagement by 25%',
        'achievement_2': 'optimized database queries resulting in 40% faster load times'
    }
    
    job_listing = {
        'title': 'Senior Full Stack Developer',
        'company': 'Enterprise Solutions',
        'description': 'Looking for an experienced developer to join our team...'
    }
    
    cover_letter = filler.generate_cover_letter(user_profile, job_listing)
    print(cover_letter)
    
    def _generate_template_cover_letter(self, user_profile: Dict, job_listing: Dict) -> str:
        """
        Generate a cover letter using templates.
        
        Args:
            user_profile: User profile containing experience, skills, etc.
            job_listing: Job listing details
            
        Returns:
            Generated cover letter text
        """
        # Extract user details
        name = user_profile.get('name', 'Your Name')
        current_role = user_profile.get('current_role', 'Professional')
        experience_years = user_profile.get('experience_years', '5+')
        skills = user_profile.get('skills', [])
        
        # Extract job details
        company = job_listing.get('company', 'the company')
        job_title = job_listing.get('title', 'the position')
        job_source = job_listing.get('source', 'your job board')
        
        # Randomly select templates
        cover_letter_template = random.choice(self.templates['cover_letter'])
        experience_template = random.choice(self.templates['experience_paragraph'])
        
        # Prepare field values
        field_values = {
            "name": name,
            "hiring_manager": job_listing.get('hiring_manager', 'Hiring Manager'),
            "job_title": job_title,
            "company": company,
            "job_source": job_source,
            "current_role": current_role,
            "experience_years": str(experience_years),
            "field": user_profile.get('field', 'the industry'),
            "current_company": user_profile.get('current_company', 'my current company'),
            "previous_company": user_profile.get('previous_company', 'my previous company'),
            "achievement_1": user_profile.get('achievement_1', 'achieved significant results'),
            "achievement_2": user_profile.get('achievement_2', 'led several successful projects'),
            "consistent_achievement": user_profile.get('consistent_achievement', 'delivered high-quality work'),
            "achievement_result": user_profile.get('achievement_result', 'improved efficiency and productivity'),
            "responsibility_1": user_profile.get('responsibility_1', 'managing key projects'),
            "responsibility_2": user_profile.get('responsibility_2', 'collaborating with cross-functional teams'),
            "skill_area": user_profile.get('skill_area', 'technical implementation and team leadership'),
            "company_appeal": job_listing.get('company_appeal', 'innovation and commitment to quality'),
            "job_appeal": job_listing.get('job_appeal', 'the opportunity to work on challenging projects'),
            "job_aspect": job_listing.get('job_aspect', 'this role')
        }
        
        # Fill in skills if available
        if skills:
            field_values["relevant_skill_1"] = skills[0] if len(skills) > 0 else "relevant skills"
            field_values["relevant_skill_2"] = skills[1] if len(skills) > 1 else "additional expertise"
        else:
            field_values["relevant_skill_1"] = "relevant skills"
            field_values["relevant_skill_2"] = "additional expertise"
        
        # Generate experience paragraph
        experience_paragraph = experience_template.format(**field_values)
        field_values["experience_paragraph"] = experience_paragraph
        
        # Generate full cover letter
        cover_letter = cover_letter_template.format(**field_values)
        
        return cover_letter
    
    def _generate_api_cover_letter(self, user_profile: Dict, job_listing: Dict) -> str:
        """
        Generate a cover letter using AI API.
        
        Args:
            user_profile: User profile containing experience, skills, etc.
            job_listing: Job listing details
            
        Returns:
            AI-generated cover letter text
        """
        if not self.use_api or not self.api_key:
            raise ValueError("API generation requested but API usage is disabled or key is missing")
            
        # Extract user details
        name = user_profile.get('name', 'Your Name')
        current_role = user_profile.get('current_role', 'Professional')
        experience_years = user_profile.get('experience_years', '5+')
        skills = user_profile.get('skills', [])
        experience = user_profile.get('experience', 'Professional experience')
        
        # Extract job details
        company = job_listing.get('company', 'the company')
        job_title = job_listing.get('title', 'the position')
        job_description = job_listing.get('description', 'the role')
        
        prompt = f"""
        Write a professional cover letter for a {job_title} position at {company}. 
        
        About me:
        - Name: {name}
        - Current role: {current_role}
        - Years of experience: {experience_years}
        - Skills: {', '.join(skills)}
        - Experience: {experience[:500]}
        
        Job details:
        - Title: {job_title}
        - Company: {company}
        - Description: {job_description[:500]}
        
        Keep the cover letter concise (under 400 words), professional, and tailored to the specific job.
        Do not use generic statements that could apply to any job.
        Highlight relevant skills and experiences that match the job requirements.
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
                }
            payload = {
                "model": "gpt-3.5-turbo",  # Use cheaper model to reduce costs
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,  # More creativity for cover letters
                "max_tokens": 500  # Limit token usage
                }
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30)
    
            if response.status_code == 200:
                cover_letter = response.json()["choices"][0]["message"]["content"]
                return cover_letter.strip()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise Exception(f"API returned error: {response.status_code}")
        except requests.RequestException as req_err:
            logger.error(f"Request error: {req_err}")
            raise Exception("API request failed") from req_err
        except Exception as e:
            logger.error(f"Error in API call: {e}")
            raise