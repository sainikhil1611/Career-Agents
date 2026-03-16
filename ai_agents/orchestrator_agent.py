"""
Orchestrator Agent
Central coordinator for job, course, and project agents.
Uses AWS Bedrock AgentCore with Strands framework and Amazon Nova Premier
for advanced multi-agent orchestration.
"""

from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel
import json
import logging
import os
import urllib.request
import urllib.parse
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NEBULA_API_KEY = os.getenv("NEBULA_API_KEY")

# Agent endpoints (configure these based on deployment)
JOB_AGENT_URL = os.getenv("JOB_AGENT_URL", "http://localhost:8081/invocations")
COURSE_AGENT_URL = os.getenv("COURSE_AGENT_URL", "http://localhost:8082/invocations")
PROJECT_AGENT_URL = os.getenv("PROJECT_AGENT_URL", "http://localhost:8083/invocations")

# Validate required environment variables
if not SERPAPI_KEY:
    raise ValueError("SERPAPI_KEY environment variable is required")
if not NEBULA_API_KEY:
    raise ValueError("NEBULA_API_KEY environment variable is required")

# Initialize BedrockAgentCore app
app = BedrockAgentCoreApp()


def call_agent(agent_url: str, query: str, timeout: int = 60) -> Dict:
    """
    Call another agent's endpoint and return the response.

    Args:
        agent_url: The URL of the agent endpoint
        query: The user query to send to the agent
        timeout: Request timeout in seconds

    Returns:
        Dictionary with agent response
    """
    try:
        logger.info(f"Calling agent at {agent_url} with query: {query[:100]}...")

        payload = json.dumps({"inputText": query}).encode('utf-8')
        headers = {'Content-Type': 'application/json'}

        req = urllib.request.Request(agent_url, data=payload, headers=headers, method='POST')

        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read().decode('utf-8')
            result = json.loads(data)

            logger.info(f"Agent response received (length: {len(str(result))})")
            return result

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error calling agent at {agent_url}: {e.code} - {e.reason}")
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        logger.error(f"URL Error calling agent at {agent_url}: {e.reason}")
        return {"error": f"Network error: {str(e.reason)}"}
    except Exception as e:
        logger.error(f"Error calling agent at {agent_url}: {e}", exc_info=True)
        return {"error": f"Failed to call agent: {str(e)}"}


@tool
def query_job_agent(job_query: str) -> Dict:
    """
    Query the job search agent to find relevant job opportunities.

    Args:
        job_query: Natural language query for job search (e.g., "Find software engineer jobs in Seattle")

    Returns:
        Job search results from SerpAPI with job listings
    """
    try:
        logger.info(f"Querying job agent: {job_query}")

        # For local testing, use inline implementation
        # In production, this would call the deployed job agent

        # Extract job parameters from query
        job_title = "software engineer"
        location = "New York"
        country = "USA"

        # Parse query for job details
        query_lower = job_query.lower()
        if "data scientist" in query_lower:
            job_title = "data scientist"
        elif "machine learning" in query_lower or "ml engineer" in query_lower:
            job_title = "machine learning engineer"
        elif "devops" in query_lower:
            job_title = "devops engineer"
        elif "frontend" in query_lower:
            job_title = "frontend developer"
        elif "backend" in query_lower:
            job_title = "backend developer"
        elif "full-stack" in query_lower or "full stack" in query_lower:
            job_title = "full stack developer"

        # Parse location
        if " in " in query_lower:
            parts = query_lower.split(" in ")
            if len(parts) > 1:
                loc_part = parts[1].strip()
                # Remove common words
                loc_part = loc_part.replace(" area", "").replace(" jobs", "")
                if "," in loc_part:
                    location, country = [x.strip() for x in loc_part.split(",", 1)]
                else:
                    location = loc_part

        # Call SerpAPI directly
        query = f"{job_title} in {location}, {country}"
        url = f"https://serpapi.com/search.json?{urllib.parse.urlencode({'engine': 'google_jobs', 'q': query, 'hl': 'en', 'api_key': SERPAPI_KEY})}"

        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        jobs = data.get("jobs_results", [])[:10]  # Limit to 10 jobs

        simplified_jobs = [
            {
                "title": j.get("title", ""),
                "company": j.get("company_name", ""),
                "location": j.get("location", ""),
                "description": j.get("description", "")[:200] + "..."
            }
            for j in jobs
        ]

        logger.info(f"Found {len(simplified_jobs)} jobs for {job_title}")

        return {
            "job_title": job_title,
            "location": f"{location}, {country}",
            "job_count": len(simplified_jobs),
            "jobs": simplified_jobs
        }

    except Exception as e:
        logger.error(f"Error querying job agent: {e}", exc_info=True)
        return {"error": f"Failed to search jobs: {str(e)}"}


@tool
def query_course_agent(course_query: str) -> Dict:
    """
    Query the course recommendation agent to find relevant courses.

    Args:
        course_query: Natural language query for courses (e.g., "What courses for data science?")

    Returns:
        Course recommendations from UTD Nebula API
    """
    try:
        logger.info(f"Querying course agent: {course_query}")

        # For local testing, use inline implementation
        # In production, this would call the deployed course agent

        # Determine departments based on query
        query_lower = course_query.lower()
        departments = []

        if any(word in query_lower for word in ["computer science", "cs", "software", "programming"]):
            departments.append("CS")
        if any(word in query_lower for word in ["data science", "data", "analytics", "ml", "machine learning"]):
            departments.extend(["CS", "STAT", "MATH"])
        if "math" in query_lower:
            departments.append("MATH")
        if "engineering" in query_lower:
            departments.extend(["SE", "CS"])

        if not departments:
            departments = ["CS"]

        # Remove duplicates
        departments = list(set(departments))

        # Fetch courses from Nebula API
        all_courses = []
        for dept in departments[:2]:  # Limit to 2 departments
            try:
                endpoint = f"https://api.utdnebula.com/course/all"
                headers = {"x-api-key": NEBULA_API_KEY}
                req = urllib.request.Request(endpoint, headers=headers, method="GET")

                with urllib.request.urlopen(req, timeout=15) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    courses = data.get("data", [])

                    # Filter by department
                    dept_courses = [
                        c for c in courses
                        if c.get("subject_prefix", "").upper() == dept
                    ]
                    all_courses.extend(dept_courses[:10])  # Limit per department

            except Exception as e:
                logger.error(f"Error fetching courses for {dept}: {e}")
                continue

        # Simplify course data
        simplified_courses = []
        seen = set()
        for course in all_courses:
            key = (course.get("subject_prefix"), course.get("course_number"))
            if key not in seen:
                simplified_courses.append({
                    "code": f"{course.get('subject_prefix', '')} {course.get('course_number', '')}",
                    "title": course.get("title", ""),
                    "description": (course.get("description", "") or "")[:200] + "...",
                    "credit_hours": course.get("credit_hours", ""),
                    "level": course.get("class_level", "")
                })
                seen.add(key)

        logger.info(f"Found {len(simplified_courses)} courses")

        return {
            "departments": departments,
            "course_count": len(simplified_courses),
            "courses": simplified_courses[:15]  # Limit total
        }

    except Exception as e:
        logger.error(f"Error querying course agent: {e}", exc_info=True)
        return {"error": f"Failed to fetch courses: {str(e)}"}


@tool
def query_project_agent(project_query: str) -> Dict:
    """
    Query the project recommendation agent to get portfolio project suggestions.

    Args:
        project_query: Natural language query for projects (e.g., "Projects for ML engineer")

    Returns:
        Project recommendations with skills and timelines
    """
    try:
        logger.info(f"Querying project agent: {project_query}")

        # For local testing, use inline implementation with curated projects
        query_lower = project_query.lower()

        # Determine career focus
        projects = []

        if any(word in query_lower for word in ["full-stack", "full stack", "web dev"]):
            projects = [
                {
                    "name": "E-Commerce Platform",
                    "description": "Full-stack e-commerce site with catalog, cart, checkout, and payment integration",
                    "skills": ["React", "Node.js", "PostgreSQL", "Stripe API"],
                    "duration": "4-6 weeks",
                    "value": "High"
                },
                {
                    "name": "Real-Time Chat Application",
                    "description": "WebSocket-based chat with rooms, direct messages, and file sharing",
                    "skills": ["WebSockets", "Authentication", "Database design"],
                    "duration": "3-4 weeks",
                    "value": "High"
                }
            ]
        elif any(word in query_lower for word in ["ml", "machine learning", "data scien", "ai"]):
            projects = [
                {
                    "name": "Recommendation Engine",
                    "description": "Content recommendation system using collaborative filtering or deep learning",
                    "skills": ["ML algorithms", "Python", "Neural networks"],
                    "duration": "4-5 weeks",
                    "value": "Very High"
                },
                {
                    "name": "Predictive Analytics Dashboard",
                    "description": "ML-powered dashboard for business forecasting with visualizations",
                    "skills": ["TensorFlow", "Pandas", "Plotly", "Time series"],
                    "duration": "5-7 weeks",
                    "value": "Very High"
                },
                {
                    "name": "Sentiment Analysis Tool",
                    "description": "NLP application analyzing sentiment from social media and reviews",
                    "skills": ["NLP", "Python", "NLTK/spaCy", "API development"],
                    "duration": "3-4 weeks",
                    "value": "High"
                }
            ]
        elif any(word in query_lower for word in ["devops", "cloud", "infrastructure"]):
            projects = [
                {
                    "name": "Microservices Architecture",
                    "description": "Containerized microservices with Docker, Kubernetes, and CI/CD",
                    "skills": ["Docker", "Kubernetes", "AWS", "CI/CD"],
                    "duration": "5-7 weeks",
                    "value": "Very High"
                },
                {
                    "name": "Infrastructure as Code Platform",
                    "description": "Automated cloud infrastructure provisioning using Terraform",
                    "skills": ["Terraform", "AWS", "Automation", "Networking"],
                    "duration": "3-4 weeks",
                    "value": "High"
                }
            ]
        else:
            # Default projects
            projects = [
                {
                    "name": "Portfolio Website with CMS",
                    "description": "Personal portfolio with custom CMS for managing projects and blog",
                    "skills": ["Frontend", "Backend", "CMS", "Deployment"],
                    "duration": "2-3 weeks",
                    "value": "Medium-High"
                },
                {
                    "name": "E-Commerce Platform",
                    "description": "Full-stack e-commerce application",
                    "skills": ["React", "Node.js", "Database"],
                    "duration": "4-6 weeks",
                    "value": "High"
                }
            ]

        logger.info(f"Recommending {len(projects)} projects")

        return {
            "project_count": len(projects),
            "projects": projects[:3]  # Limit to 3 projects
        }

    except Exception as e:
        logger.error(f"Error querying project agent: {e}", exc_info=True)
        return {"error": f"Failed to get project recommendations: {str(e)}"}


# Configure the orchestrator agent with Amazon Nova Pro
# Note: Nova Premier requires inference profile ARN, using Nova Pro for orchestration
bedrock_model = BedrockModel(
    model_id="amazon.nova-pro-v1:0",  # Pro model for orchestration
    region_name=AWS_REGION,
    max_tokens=6000  # Higher limit for comprehensive career plans
)

# Create the orchestrator agent
agent = Agent(
    model=bedrock_model,
    name="CareerOrchestratorAgent",
    system_prompt="""You are the central orchestrator coordinating job, course, and project agents to create personalized career plans.

Your role is to:
1. Analyze the user's query to understand their career goal
2. Decide which agent(s) to invoke based on what they need
3. Call the appropriate agents with well-formed queries
4. Merge outputs from multiple agents
5. Present a unified, structured recommendation

Available agents:
- **query_job_agent**: Finds job opportunities matching career goals
- **query_course_agent**: Recommends university courses to build skills
- **query_project_agent**: Suggests portfolio projects to demonstrate abilities

When to call each agent:
- Call **query_job_agent** when user wants to know about job market, salaries, or job hunting
- Call **query_course_agent** when user needs education/courses/learning paths
- Call **query_project_agent** when user wants to build portfolio/demonstrate skills
- Call **ALL THREE** when creating comprehensive career plan

Guidelines:
1. **For comprehensive career planning**: Call all three agents
2. **Extract clear goals**: Identify the target role from user query
3. **Form specific queries**: Make clear, focused queries to each agent
4. **Synthesize results**: Combine outputs into cohesive career plan
5. **Structure output**: Present as clear sections (Jobs, Education, Portfolio)
6. **Add context**: Explain how each piece fits into overall career strategy

Example orchestration:
User: "I want to become a machine learning engineer"
Actions:
1. Call query_project_agent("projects for machine learning engineer career")
2. Call query_course_agent("courses for machine learning and data science")
3. Call query_job_agent("machine learning engineer jobs")
4. Synthesize into career roadmap

Present final output with:
- Executive Summary
- Recommended Learning Path (courses)
- Portfolio Projects (with rationale)
- Job Market Insights
- Next Steps Timeline

Be strategic, comprehensive, and actionable. Your goal is to provide a complete career development plan.""",
    tools=[query_job_agent, query_course_agent, query_project_agent]
)


@app.entrypoint
def invoke_agentcore(payload):
    """
    AgentCore entrypoint for orchestrator.
    Coordinates multiple agents to create comprehensive career plans.
    """
    try:
        # Extract user input
        user_input = payload.get("inputText", "") or payload.get("prompt", "")

        if not user_input:
            user_input = "I want to become a software engineer. Create a complete career plan for me."

        logger.info(f"Processing orchestration request: {user_input}")

        # Invoke the orchestrator agent
        result = agent(user_input)

        # Extract text from Strands response
        if hasattr(result, 'message'):
            if isinstance(result.message, dict):
                content = result.message.get('content', [])
                if content and isinstance(content, list):
                    response_text = content[0].get('text', str(result.message))
                else:
                    response_text = str(result.message)
            else:
                response_text = str(result.message)
        else:
            response_text = str(result)

        logger.info(f"Orchestration completed successfully")
        logger.debug(f"Response preview: {response_text[:300]}...")

        return {
            "response": response_text
        }

    except Exception as e:
        logger.error(f"Error in orchestrator: {e}", exc_info=True)
        return {
            "response": f"I apologize, but I encountered an error creating your career plan: {str(e)}. Please try again with a more specific goal."
        }


if __name__ == "__main__":
    logger.info("Starting Career Orchestrator Agent on port 8080...")
    logger.info("Orchestrator will coordinate job, course, and project agents")
    app.run()
