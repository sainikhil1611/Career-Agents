"""
Course Recommendation Agent
Uses AWS Bedrock AgentCore with Strands framework to recommend university courses
based on a user's career goals.
"""

from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel
import json
import urllib.request
import urllib.error
import logging
import os
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
NEBULA_API_KEY = os.getenv("NEBULA_API_KEY")
NEBULA_BASE_URL = os.getenv("NEBULA_BASE_URL", "https://api.utdnebula.com")
MAX_DESC_LENGTH = 250
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Validate required environment variables
if not NEBULA_API_KEY:
    raise ValueError("NEBULA_API_KEY environment variable is required")

# Initialize BedrockAgentCore app
app = BedrockAgentCoreApp()


def truncate(text, length=MAX_DESC_LENGTH):
    """Truncate text to specified length with ellipsis"""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= length else text[:length].rstrip() + "..."


@tool
def get_courses_by_department(
    course_dept: str,
    course_level: str = ""
) -> dict:
    """
    Fetch and filter university courses by department and optional class level.

    Args:
        course_dept: Department code (e.g., 'CS', 'MATH', 'PHYS')
        course_level: Optional class level filter (e.g., 'Lower Division', 'Upper Division')

    Returns:
        Dictionary with count and list of relevant courses
    """
    try:
        if not NEBULA_API_KEY:
            logger.error("NEBULA_API_KEY not configured")
            return {
                "error": "API key not configured. Please set NEBULA_API_KEY environment variable."
            }

        # Fetch all courses from Nebula API
        logger.info(f"Fetching courses for department: {course_dept}, level: {course_level or 'all'}")

        endpoint = f"{NEBULA_BASE_URL}/course/all"
        headers = {"x-api-key": NEBULA_API_KEY}
        req = urllib.request.Request(endpoint, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read().decode("utf-8")
                parsed = json.loads(data)
                all_courses = parsed.get("data", [])
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP Error fetching courses: {e.code} - {e.reason}")
            return {"error": f"Failed to fetch courses: HTTP {e.code}"}
        except urllib.error.URLError as e:
            logger.error(f"URL Error fetching courses: {e.reason}")
            return {"error": f"Network error: {str(e.reason)}"}
        except Exception as e:
            logger.error(f"Unexpected error fetching courses: {e}")
            return {"error": f"Failed to fetch courses: {str(e)}"}

        # Simplify and deduplicate courses
        simplified = []
        seen = set()
        dept_code_upper = course_dept.upper()
        course_level_lower = course_level.lower() if course_level else ""

        for course in all_courses:
            dept = course.get("subject_prefix", "").upper()
            number = course.get("course_number", "")
            key = (dept, number)

            # Filter by department
            if dept != dept_code_upper:
                continue

            # Filter by class level if specified
            if course_level_lower and course.get("class_level", "").lower() != course_level_lower:
                continue

            # Deduplicate
            if key not in seen:
                simplified.append({
                    "title": course.get("title", ""),
                    "course_number": number,
                    "description": truncate(course.get("description", "")),
                    "credit_hours": course.get("credit_hours", ""),
                    "class_level": course.get("class_level", ""),
                    "school": course.get("school", ""),
                    "subject_prefix": dept
                })
                seen.add(key)

        # Limit results to prevent overwhelming the LLM
        max_results = 50
        result = {
            "count": len(simplified),
            "results": simplified[:max_results]
        }

        if len(simplified) > max_results:
            result["note"] = f"Showing first {max_results} of {len(simplified)} courses"

        logger.info(f"Found {len(simplified)} courses, returning {min(len(simplified), max_results)}")
        return result

    except Exception as e:
        logger.error(f"Error in get_courses_by_department: {e}", exc_info=True)
        return {"error": f"Failed to process course search: {str(e)}"}


@tool
def search_courses_by_keyword(keyword: str, max_results: int = 20) -> dict:
    """
    Search for courses by keyword in title or description.

    Args:
        keyword: Search term to find in course titles or descriptions
        max_results: Maximum number of results to return (default: 20)

    Returns:
        Dictionary with count and list of matching courses
    """
    try:
        if not NEBULA_API_KEY:
            return {"error": "API key not configured"}

        logger.info(f"Searching courses with keyword: {keyword}")

        endpoint = f"{NEBULA_BASE_URL}/course/all"
        headers = {"x-api-key": NEBULA_API_KEY}
        req = urllib.request.Request(endpoint, headers=headers, method="GET")

        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read().decode("utf-8")
            parsed = json.loads(data)
            all_courses = parsed.get("data", [])

        # Search in title and description
        keyword_lower = keyword.lower()
        matching = []
        seen = set()

        for course in all_courses:
            dept = course.get("subject_prefix", "").upper()
            number = course.get("course_number", "")
            key = (dept, number)

            title = course.get("title", "").lower()
            desc = course.get("description", "").lower()

            # Check if keyword matches
            if keyword_lower in title or keyword_lower in desc:
                if key not in seen:
                    matching.append({
                        "title": course.get("title", ""),
                        "course_number": number,
                        "description": truncate(course.get("description", "")),
                        "credit_hours": course.get("credit_hours", ""),
                        "class_level": course.get("class_level", ""),
                        "school": course.get("school", ""),
                        "subject_prefix": dept
                    })
                    seen.add(key)

                    if len(matching) >= max_results:
                        break

        logger.info(f"Found {len(matching)} courses matching '{keyword}'")
        return {
            "count": len(matching),
            "results": matching
        }

    except Exception as e:
        logger.error(f"Error in search_courses_by_keyword: {e}", exc_info=True)
        return {"error": f"Failed to search courses: {str(e)}"}


# Configure the Strands agent with Amazon Nova Pro
bedrock_model = BedrockModel(
    model_id="amazon.nova-pro-v1:0",
    region_name=AWS_REGION
)

# Create the course advisor agent
agent = Agent(
    model=bedrock_model,
    name="CourseAdvisorAgent",
    system_prompt="""You are an expert career advisor with access to real-time university course data from UTD (University of Texas at Dallas).

Your expertise lies in analyzing a user's ideal job or career goal to recommend the most relevant courses they should take.

When a user describes their career goal or ideal job:
1. Identify the key technical skills and knowledge areas required for that career
2. Determine which academic departments offer relevant courses (e.g., CS for software engineering, MATH for data science)
3. Use the get_courses_by_department tool to find relevant courses
4. Use the search_courses_by_keyword tool to find courses matching specific topics
5. Recommend a structured learning path with specific courses
6. Explain why each course is relevant to their career goal
7. Suggest both foundational (Lower Division) and advanced (Upper Division) courses

Common department codes:
- CS: Computer Science
- SE: Software Engineering
- MATH: Mathematics
- STAT: Statistics
- PHYS: Physics
- EECS: Electrical Engineering & Computer Science
- BIOL: Biology
- CHEM: Chemistry
- BCOM: Business Communication
- MECH: Mechanical Engineering

Be specific, actionable, and explain the connection between courses and career goals. If a user asks about a specific career, analyze it thoughtfully and provide a comprehensive course recommendation.""",
    tools=[get_courses_by_department, search_courses_by_keyword]
)


@app.entrypoint
def invoke_agentcore(payload):
    """
    AgentCore entrypoint using Strands framework.
    Handles course recommendation requests based on career goals.
    """
    try:
        # Extract user input
        user_input = payload.get("inputText", "") or payload.get("prompt", "")

        if not user_input:
            user_input = "What courses should I take to become a software engineer?"

        logger.info(f"Processing course recommendation request: {user_input}")

        # Invoke the Strands agent
        result = agent(user_input)

        # Extract text from Strands response
        if hasattr(result, 'message'):
            if isinstance(result.message, dict):
                # Extract text from structured response
                content = result.message.get('content', [])
                if content and isinstance(content, list):
                    response_text = content[0].get('text', str(result.message))
                else:
                    response_text = str(result.message)
            else:
                response_text = str(result.message)
        else:
            response_text = str(result)

        logger.info(f"Agent response generated successfully")
        logger.debug(f"Response preview: {response_text[:200]}...")

        return {
            "response": response_text
        }

    except Exception as e:
        logger.error(f"Error in agent invocation: {e}", exc_info=True)
        return {
            "response": f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question."
        }


if __name__ == "__main__":
    logger.info("Starting Course Advisor Agent on port 8080...")
    app.run()
