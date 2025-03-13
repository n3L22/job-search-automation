"""
Generic Scraper Module

This module provides a base scraper class that can be extended for specific job boards.
It includes common functionality for querying job boards and extracting job listings.
"""

import os
import time
import logging
import re
import random
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

class GenericScraper(ABC):
    """
    Base scraper class that provides common functionality for all job board scrapers.
    This class should be extended for specific job boards.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the scraper.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.driver = None
        self.session = requests.Session()
        
        # Set up common headers for requests
        user_agent = self.config.get('scrapers', {}).get('global', {}).get(
            'user_agent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        })
        
        # Load scraper-specific settings
        self.base_url = ""
        self.results_per_page = 25
        self.request_delay = self.config.get('scrapers', {}).get('global', {}).get('request_delay', 2)
        self.max_retries = self.config.get('scrapers', {}).get('global', {}).get('max_retries', 3)
        self.timeout = self.config.get('scrapers', {}).get('global', {}).get('timeout', 30)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration values
        """
        # Default config path
        if not config_path:
            config_path = 'config/config.yaml'
            
        # Try loading custom config first
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            # Fall back to default config
            try:
                with open('config/default_config.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info("Loaded default configuration")
                    return config
            except FileNotFoundError:
                logger.warning("No configuration file found. Using default values.")
                return {}
                
    def _init_driver(self):
        """Initialize Selenium WebDriver for browser automation."""
        if self.driver:
            return
            
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing WebDriver: {e}")
            raise
            
    def _close_driver(self):
        """Close Selenium WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Selenium WebDriver closed")
            
    def _get_with_retry(self, url: str, use_selenium: bool = False) -> Optional[str]:
        """
        Get a URL with retry mechanism.
        
        Args:
            url: URL to fetch
            use_selenium: Whether to use Selenium for fetching
            
        Returns:
            HTML content or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                if use_selenium:
                    if not self.driver:
                        self._init_driver()
                        
                    self.driver.get(url)
                    
                    # Wait for page to load
                    WebDriverWait(self.driver, self.timeout).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    
                    html = self.driver.page_source
                else:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    html = response.text
                    
                # Add random delay to avoid detection
                delay = self.request_delay + random.uniform(0, 2)
                time.sleep(delay)
                
                return html
                
            except (requests.RequestException, TimeoutException, WebDriverException) as e:
                logger.warning(f"Error fetching URL (attempt {attempt+1}/{self.max_retries}): {e}")
                
                # Add increasing delay between retries
                time.sleep(2 ** attempt)
                
        logger.error(f"Failed to fetch URL after {self.max_retries} attempts: {url}")
        return None
        
    def search_jobs(self, keywords: str, location: str, **kwargs) -> List[Dict]:
        """
        Search for jobs using the given keywords and location.
        
        Args:
            keywords: Job search keywords
            location: Job location
            **kwargs: Additional search parameters
            
        Returns:
            List of job listings
        """
        try:
            # Build search URL
            search_url = self._build_search_url(keywords, location, **kwargs)
            
            # Determine if we need Selenium
            use_selenium = self._requires_selenium()
            
            # Get search results
            all_jobs = []
            total_pages = self._get_total_pages(keywords, location, **kwargs)
            
            # Limit number of pages to scrape
            max_pages = kwargs.get('max_pages', 5)
            pages_to_scrape = min(total_pages, max_pages)
            
            logger.info(f"Scraping {pages_to_scrape} pages of job results")
            
            for page in range(1, pages_to_scrape + 1):
                logger.info(f"Scraping page {page} of {pages_to_scrape}")
                
                page_url = self._get_page_url(search_url, page)
                page_html = self._get_with_retry(page_url, use_selenium)
                
                if not page_html:
                    logger.warning(f"Failed to get page {page}, skipping")
                    continue
                    
                # Parse job listings on this page
                jobs_on_page = self._parse_job_listings(page_html)
                all_jobs.extend(jobs_on_page)
                
                # Check if we've reached the requested limit
                if len(all_jobs) >= kwargs.get('limit', float('inf')):
                    all_jobs = all_jobs[:kwargs.get('limit')]
                    break
                    
            logger.info(f"Found {len(all_jobs)} job listings")
            
            # Get details for each job
            if kwargs.get('fetch_details', True):
                detailed_jobs = []
                
                for job in all_jobs:
                    # Skip if job already has details
                    if job.get('description'):
                        detailed_jobs.append(job)
                        continue
                        
                    # Get job details
                    job_with_details = self._get_job_details(job)
                    if job_with_details:
                        detailed_jobs.append(job_with_details)
                        
                all_jobs = detailed_jobs
                
            return all_jobs
            
        except Exception as e:
            logger.error(f"Error in job search: {e}")
            return []
        finally:
            # Clean up
            self._close_driver()
            
    def _requires_selenium(self) -> bool:
        """
        Determine if this scraper requires Selenium.
        Override in subclasses if needed.
        
        Returns:
            True if Selenium is required, False otherwise
        """
        return False
        
    @abstractmethod
    def _build_search_url(self, keywords: str, location: str, **kwargs) -> str:
        """
        Build the URL for job search.
        Must be implemented by subclasses.
        
        Args:
            keywords: Job search keywords
            location: Job location
            **kwargs: Additional search parameters
            
        Returns:
            Search URL
        """
        pass
        
    @abstractmethod
    def _get_total_pages(self, keywords: str, location: str, **kwargs) -> int:
        """
        Get the total number of pages of search results.
        Must be implemented by subclasses.
        
        Args:
            keywords: Job search keywords
            location: Job location
            **kwargs: Additional search parameters
            
        Returns:
            Total number of pages
        """
        pass
        
    @abstractmethod
    def _get_page_url(self, base_url: str, page: int) -> str:
        """
        Get the URL for a specific page of search results.
        Must be implemented by subclasses.
        
        Args:
            base_url: Base search URL
            page: Page number
            
        Returns:
            Page URL
        """
        pass
        
    @abstractmethod
    def _parse_job_listings(self, html: str) -> List[Dict]:
        """
        Parse job listings from search results HTML.
        Must be implemented by subclasses.
        
        Args:
            html: HTML content
            
        Returns:
            List of job listings
        """
        pass
        
    @abstractmethod
    def _get_job_details(self, job: Dict) -> Dict:
        """
        Get detailed information for a job listing.
        Must be implemented by subclasses.
        
        Args:
            job: Basic job listing
            
        Returns:
            Job listing with details
        """
        pass
        
    def save_jobs(self, jobs: List[Dict], output_file: str):
        """
        Save job listings to file.
        
        Args:
            jobs: List of job listings
            output_file: Path to output file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Add timestamp to each job
            for job in jobs:
                if 'timestamp' not in job:
                    job['timestamp'] = datetime.now().isoformat()
                    
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(jobs, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved {len(jobs)} jobs to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving jobs to file: {e}")
            
    def load_jobs(self, input_file: str) -> List[Dict]:
        """
        Load job listings from file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of job listings
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                jobs = json.load(f)
                
            logger.info(f"Loaded {len(jobs)} jobs from {input_file}")
            return jobs
            
        except Exception as e:
            logger.error(f"Error loading jobs from file: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Cannot instantiate abstract class directly
    pass