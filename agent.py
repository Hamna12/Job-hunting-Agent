
"""
Job Hunting Agent - Complete Working Version
"""

import asyncio
import json
import nest_asyncio
import requests
import pandas as pd
from datetime import datetime
from playwright.async_api import async_playwright
from pydantic import BaseModel
from typing import List, Optional
import re
import sys

# Enable nested event loops
nest_asyncio.apply()

# Structured Data Models
class JobOpportunity(BaseModel):
    title: str
    company: str
    location: str
    summary: str
    job_url: str
    platform: str
    date_posted: str
    
class JobAnalysis(BaseModel):
    relevance_score: int  # 1-10
    key_skills: List[str]
    contact_priority: str  # High, Medium, Low
    outreach_strategy: str

class JobList(BaseModel):
    jobs: List[JobOpportunity]

class SimpleJobAgent:
    def __init__(self, ollama_model="gemma3:4b"):
        self.playwright = None
        self.browser = None
        self.page = None
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = ollama_model
        self.jobs_found = []
        
    async def init_browser(self):
        """Initialize browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,  # Keep visible for debugging
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-web-security"
            ]
        )
        self.page = await self.browser.new_page()
        
        # Set user agent to avoid detection
        await self.page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def scrape_content(self, url):
        """Scrape page content"""
        if not self.page or self.page.is_closed():
            await self.init_browser()
        
        print(f"🔍 Navigating to: {url}")
        await self.page.goto(url, wait_until="load")
        await self.page.wait_for_timeout(3000)
        
        return await self.page.content()
    
    async def take_screenshot(self):
        """Take screenshot for debugging"""
        screenshot_bytes = await self.page.screenshot(type="png", full_page=False)
        return screenshot_bytes
    
    def call_ollama(self, prompt, system_prompt="You are a helpful AI assistant."):
        """Call local Ollama API with proper debugging"""
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": f"System: {system_prompt}\\n\\nUser: {prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
            }
            
            print(f"🤖 Calling Ollama with model: {self.ollama_model}")
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json().get('response', '')
                print(f"✅ Ollama responded with {len(result)} characters")
                return result
            else:
                print(f"❌ Ollama error: {response.status_code}")
                print(f"Response text: {response.text[:500]}")
                
                # Debug: Print the exact payload being sent
                print(f"🔍 Debug - Payload sent:")
                print(f"  Model: {payload['model']}")
                print(f"  URL: {self.ollama_url}")
                return ""
        except Exception as e:
            print(f"❌ Ollama call failed: {e}")
            return ""
    
    async def extract_indeed_jobs(self, search_term="AI developer", location="remote"):
        """Extract jobs from Indeed using HTML parsing"""
        search_url = f"https://www.indeed.com/jobs?q={search_term.replace(' ', '+')}&l={location}&fromage=2"
        
        try:
            html_content = await self.scrape_content(search_url)
            print(f"📄 Got HTML content: {len(html_content)} characters")
            
            # Use Ollama to extract job data from HTML
            extraction_prompt = f"""
            Extract job listings from this Indeed HTML content. 
            Find all job postings and extract ONLY:
            - Job title
            - Company name  
            - Location
            - Job summary/description
            
            Return ONLY a valid JSON array like this example:
            [
                {{
                    "title": "Python Developer",
                    "company": "TechCorp",
                    "location": "Remote",
                    "summary": "Looking for Python developer...",
                    "job_url": "https://indeed.com/job123"
                }}
            ]
            
            HTML Content (first 8000 chars):
            {html_content[:8000]}
            """
            
            system_prompt = """You are an expert at extracting structured data from HTML. 
            Extract ONLY job listings. Return ONLY valid JSON array, no extra text or markdown."""
            
            print("🤖 Extracting jobs with Ollama...")
            response = self.call_ollama(extraction_prompt, system_prompt)
            
            if not response:
                print("❌ No response from Ollama")
                return []
            
            # Debug: Show first 200 chars of response
            print(f"🔍 Ollama response preview: {response[:200]}...")
            
            # Try to parse JSON from response
            jobs_data = self._parse_ollama_json_response(response)
            
            # Add metadata
            for job in jobs_data:
                job.update({
                    'platform': 'Indeed',
                    'date_posted': 'Recent',
                    'search_term': search_term
                })
            
            print(f"✅ Extracted {len(jobs_data)} jobs from Indeed")
            return jobs_data
            
        except Exception as e:
            print(f"❌ Error extracting Indeed jobs: {e}")
            return []
    
    async def extract_remoteok_jobs(self, search_terms=["AI", "python", "developer"]):
        """Extract jobs from RemoteOK"""
        try:
            html_content = await self.scrape_content("https://remoteok.io/")
            print(f"📄 Got RemoteOK HTML: {len(html_content)} characters")
            
            extraction_prompt = f"""
            Extract remote job listings from this RemoteOK HTML.
            Look for jobs related to: {', '.join(search_terms)}
            
            Find job postings and extract:
            - Job title
            - Company name
            - Skills/tags mentioned
            
            Return ONLY valid JSON array:
            [
                {{
                    "title": "Job Title",
                    "company": "Company Name",
                    "location": "Remote",
                    "summary": "Skills mentioned",
                    "job_url": "https://remoteok.io/job123"
                }}
            ]
            
            HTML Content (first 8000 chars):
            {html_content[:8000]}
            """
            
            system_prompt = "Extract only jobs related to AI, ML, or development. Return valid JSON only."
            
            print("🤖 Extracting RemoteOK jobs...")
            response = self.call_ollama(extraction_prompt, system_prompt)
            
            if not response:
                return []
            
            jobs_data = self._parse_ollama_json_response(response)
            
            # Add metadata
            for job in jobs_data:
                job.update({
                    'platform': 'RemoteOK',
                    'location': 'Remote',
                    'date_posted': 'Recent'
                })
            
            print(f"✅ Extracted {len(jobs_data)} jobs from RemoteOK")
            return jobs_data
            
        except Exception as e:
            print(f"❌ Error extracting RemoteOK jobs: {e}")
            return []
    
    def _parse_ollama_json_response(self, response):
        """Parse JSON from Ollama response with better debugging"""
        try:
            # Clean up the response
            response = response.strip()
            
            # Try to find JSON array in response
            json_match = re.search(r'\\[.*\\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print(f"🔍 Found JSON match: {json_str[:100]}...")
                return json.loads(json_str)
            
            # Try parsing whole response as JSON
            print("🔍 Trying to parse entire response as JSON")
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"🔍 Response content: {response[:500]}")
            return []
        except Exception as e:
            print(f"❌ Unexpected parsing error: {e}")
            return []
    
    def analyze_job_with_ollama(self, job_data):
        """Analyze job relevance using Ollama"""
        analysis_prompt = f"""
        Analyze this job opportunity for a 2-person startup offering AI/ML development services:
        
        Job: {job_data.get('title', '')}
        Company: {job_data.get('company', '')}
        Description: {job_data.get('summary', '')}
        
        Provide analysis in EXACTLY this JSON format:
        {{
            "relevance_score": 7,
            "key_skills": ["Python", "AI", "ML"],
            "contact_priority": "High",
            "outreach_strategy": "Mention AI expertise and startup agility"
        }}
        
        Rate based on:
        - Relevance to AI/ML services (40%)
        - Company size/startup friendliness (30%) 
        - Technical skill match (30%)
        
        Score 1-10 where 10 = perfect match.
        """
        
        system_prompt = "You are an expert business analyst. Return ONLY valid JSON, no extra text."
        
        response = self.call_ollama(analysis_prompt, system_prompt)
        
        if not response:
            return self._fallback_analysis()
        
        try:
            json_match = re.search(r'\\{.*\\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except:
            print("⚠️  Could not parse job analysis, using fallback")
            return self._fallback_analysis()
    
    def _fallback_analysis(self):
        """Fallback analysis when AI parsing fails"""
        return {
            "relevance_score": 5,
            "key_skills": ["development"],
            "contact_priority": "Medium", 
            "outreach_strategy": "Professional introduction"
        }
    
    def generate_outreach_message(self, job_data, analysis):
        """Generate personalized outreach message"""
        outreach_prompt = f"""
        Write a professional outreach email for this job opportunity:
        
        Job: {job_data.get('title', '')}
        Company: {job_data.get('company', '')}
        Key Skills: {', '.join(analysis.get('key_skills', []))}
        
        Our startup offers:
        - AI/ML development and consulting
        - Custom software solutions
        - Rapid MVP development
        - 2 experienced developers
        
        Write a concise email (100-150 words) that:
        1. References their specific job posting
        2. Highlights relevant experience
        3. Proposes a brief call
        4. Professional but friendly tone
        
        Return just the email content, no subject line.
        """
        
        system_prompt = "You are an expert at writing professional outreach emails. Be concise and personalized."
        
        return self.call_ollama(outreach_prompt, system_prompt)
    
    async def run_job_hunt(self, search_terms=["AI developer", "python developer"], max_jobs=20):
        """Main job hunting pipeline"""
        print("🚀 Starting Job Hunt Pipeline...")
        
        try:
            # Initialize browser
            await self.init_browser()
            
            all_jobs = []
            
            # Scrape Indeed
            for term in search_terms[:2]:  # Limit to avoid rate limiting
                print(f"\\n🔍 Searching Indeed for: {term}")
                jobs = await self.extract_indeed_jobs(term)
                all_jobs.extend(jobs[:10])  # Limit per term
            
            # Scrape RemoteOK
            print("\\n🔍 Searching RemoteOK...")
            remote_jobs = await self.extract_remoteok_jobs()
            all_jobs.extend(remote_jobs[:10])
            
            print(f"\\n📊 Total jobs found: {len(all_jobs)}")
            
            if not all_jobs:
                print("⚠️  No jobs found. This could be due to:")
                print("  1. Website structure changes")
                print("  2. Anti-bot measures")
                print("  3. Ollama parsing issues")
                return None
            
            # Analyze jobs with Ollama
            analyzed_jobs = []
            for i, job in enumerate(all_jobs):
                print(f"\\n🤖 Analyzing job {i+1}/{len(all_jobs)}: {job.get('title', '')}")
                
                analysis = self.analyze_job_with_ollama(job)
                outreach_msg = self.generate_outreach_message(job, analysis)
                
                analyzed_job = {
                    **job,
                    'analysis': analysis,
                    'outreach_message': outreach_msg,
                    'processed_at': datetime.now().isoformat()
                }
                
                analyzed_jobs.append(analyzed_job)
            
            # Filter high priority jobs
            high_priority = [
                job for job in analyzed_jobs 
                if job['analysis']['relevance_score'] >= 7 or 
                job['analysis']['contact_priority'] == 'High'
            ]
            
            print(f"\\n🎯 High priority opportunities: {len(high_priority)}")
            
            return {
                'all_jobs': analyzed_jobs,
                'high_priority': high_priority,
                'summary': {
                    'total_found': len(all_jobs),
                    'total_analyzed': len(analyzed_jobs),
                    'high_priority_count': len(high_priority)
                }
            }
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return None
        
        finally:
            await self.close()
    
    def save_results(self, results, filename="job_opportunities"):
        """Save results to CSV and JSON"""
        if not results:
            return
        
        # Prepare data for CSV
        csv_data = []
        for job in results['all_jobs']:
            analysis = job.get('analysis', {})
            csv_row = {
                'title': job.get('title', ''),
                'company': job.get('company', ''),
                'platform': job.get('platform', ''),
                'location': job.get('location', ''),
                'relevance_score': analysis.get('relevance_score', 0),
                'priority': analysis.get('contact_priority', 'Medium'),
                'key_skills': ', '.join(analysis.get('key_skills', [])),
                'job_url': job.get('job_url', ''),
                'outreach_preview': job.get('outreach_message', '')[:100] + '...'
            }
            csv_data.append(csv_row)
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(f"{filename}.csv", index=False)
        
        # Save detailed JSON
        with open(f"{filename}_detailed.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {filename}.csv and {filename}_detailed.json")
        
        return df
    
    async def close(self):
        """Clean up resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

# Test Functions
def test_ollama_connection():
    """Test if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✅ Ollama is running!")
            print("Available models:", [m['name'] for m in models])
            return True
        else:
            print("❌ Ollama not accessible")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def test_ollama_generation():
    """Test Ollama text generation"""
    try:
        payload = {
            "model": "gemma3:4b",
            "prompt": "System: You are a helpful assistant.\\n\\nUser: Respond with only 'TEST SUCCESS'",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 10
            }
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        print(f"Ollama generation test - Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json().get('response', '')
            print(f"✅ Ollama generation successful: '{result.strip()}'")
            return True
        else:
            print(f"❌ Ollama generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ollama generation test failed: {e}")
        return False

# Usage Functions
async def quick_test():
    """Quick test with limited scope"""
    agent = SimpleJobAgent(ollama_model="gemma3:4b")
    
    try:
        # Test with just Indeed
        jobs = await agent.extract_indeed_jobs("python developer")
        print(f"\\nFound {len(jobs)} jobs in test")
        
        if jobs:
            # Test analysis on first job
            sample_job = jobs[0]
            print(f"\\n🤖 Testing analysis on: {sample_job.get('title', 'N/A')}")
            analysis = agent.analyze_job_with_ollama(sample_job)
            print(f"Sample analysis: {analysis}")
            
            # Test outreach generation
            print("\\n✉️ Testing outreach generation...")
            message = agent.generate_outreach_message(sample_job, analysis)
            print(f"Sample message (first 200 chars): {message[:200]}...")
        
        return jobs
        
    finally:
        await agent.close()

async def full_job_hunt():
    """Run complete job hunting pipeline"""
    agent = SimpleJobAgent(ollama_model="gemma3:4b")
    
    search_terms = ["AI developer", "python developer", "machine learning engineer"]
    
    results = await agent.run_job_hunt(search_terms, max_jobs=30)
    
    if results:
        # Save results
        df = agent.save_results(results)
        
        # Display summary
        print("\\n📊 JOB HUNT SUMMARY:")
        print(f"Total jobs found: {results['summary']['total_found']}")
        print(f"High priority opportunities: {results['summary']['high_priority_count']}")
        
        # Show top opportunities
        print("\\n🎯 TOP OPPORTUNITIES:")
        for i, job in enumerate(results['high_priority'][:5], 1):
            print(f"{i}. {job['title']} at {job['company']} (Score: {job['analysis']['relevance_score']})")
        
        return results
    
    return None

# Main execution function
async def main():
    """Main async function"""
    print("🤖 Simple Job Hunting Agent with Ollama")
    print("=====================================")
    
    # Test Ollama connection
    if not test_ollama_connection():
        return
    
    # Test Ollama generation
    print("\\n🧪 Testing Ollama generation...")
    if not test_ollama_generation():
        print("❌ Ollama generation test failed. Please check your model.")
        return
    
    # Choose mode
    print("\\n1. Quick test (limited results)")
    print("2. Full job hunt (complete pipeline)")
    print("3. Debug mode (test single Ollama call)")
    
    try:
        mode = input("Choose mode (1, 2, or 3): ").strip()
    except EOFError:
        print("No input available, defaulting to quick test")
        mode = "1"
    
    if mode == "1":
        print("\\n🧪 Starting quick test...")
        jobs = await quick_test()
        print(f"✅ Test completed with {len(jobs)} jobs found")
        
    elif mode == "2":
        print("\\n🚀 Starting full job hunt...")
        results = await full_job_hunt()
        if results:
            print("✅ Job hunt completed successfully!")
            print(f"📁 Check your current directory for:")
            print("  • job_opportunities.csv")
            print("  • job_opportunities_detailed.json")
        else:
            print("❌ Job hunt failed")
    
    elif mode == "3":
        print("\\n🔬 Debug mode - Testing single Ollama call...")
        agent = SimpleJobAgent(ollama_model="gemma3:4b")
        test_prompt = "List exactly 3 programming languages used for web development. Return as JSON array."
        result = agent.call_ollama(test_prompt, "You are a helpful assistant. Return only JSON.")
        print(f"\\nDebug result: {result}")
    
    else:
        print("❌ Invalid choice, running quick test instead")
        jobs = await quick_test()
        print(f"✅ Test completed with {len(jobs)} jobs found")

# Script entry point
def run_script():
    """Synchronous wrapper to run the async main function"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n👋 Script interrupted by user")
    except Exception as e:
        print(f"❌ Script error: {e}")

# This runs when script is executed directly
if __name__ == "__main__":
    run_script()




