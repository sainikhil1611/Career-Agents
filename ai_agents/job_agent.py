from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel
import json
import urllib.request
import urllib.parse
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
MAX_DESC_LENGTH = 200

# Validate required environment variables
if not SERPAPI_KEY:
    raise ValueError("SERPAPI_KEY environment variable is required")

# Initialize BedrockAgentCore app
app = BedrockAgentCoreApp()


def truncate(text, length=MAX_DESC_LENGTH):
    """Truncate text to specified length"""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= length else text[:length].rstrip() + "..."


@tool
def search_jobs(job_title: str, location: str = "New York", country: str = "USA") -> dict:
    """
    Search for job listings using SerpAPI.

    Args:
        job_title: The job title or role to search for
        location: The city or region to search in
        country: The country to search in

    Returns:
        A dictionary containing job search results
    """
    query = f"{job_title} in {location}, {country}"
    url = f"https://serpapi.com/search.json?{urllib.parse.urlencode({'engine': 'google_jobs', 'q': query, 'hl': 'en', 'api_key': SERPAPI_KEY})}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Error fetching jobs: {e}")
        return {"error": f"Failed to fetch job listings: {str(e)}"}

    jobs = data.get("jobs_results", [])
    compact_jobs = [
        {
            "title": j.get("title", ""),
            "company": j.get("company_name", ""),
            "location": j.get("location", ""),
            "via": j.get("source", ""),
            "link": j.get("apply_options", [{}])[0].get("link", ""),
            "description": truncate(j.get("description", "")).replace("\n", " ")
        }
        for j in jobs
    ]

    return {
        "count": len(compact_jobs),
        "results": compact_jobs
    }


# Configure the Strands agent with Amazon Nova Pro
bedrock_model = BedrockModel(
    model_id="amazon.nova-pro-v1:0",
    region_name="us-east-1"
)

# Create the agent with tools
agent = Agent(
    model=bedrock_model,
    name="JobSearchAgent",
    system_prompt="""You are a helpful career advisor and job search assistant.
Your role is to help users find job opportunities based on their preferences.

When a user asks about jobs:
1. Extract the job title and location from their request
2. Use the search_jobs tool to find relevant positions
3. Present the results in a clear, helpful manner
4. If no specific details are provided, use sensible defaults (software engineer in New York, USA)

Be conversational, helpful, and provide actionable information.""",
    tools=[search_jobs]
)


@app.entrypoint
def invoke_agentcore(payload):
    """
    AgentCore entrypoint using Strands framework.
    Handles job search requests through an intelligent agent.
    """
    try:
        # Extract user input
        user_input = payload.get("inputText", "") or payload.get("prompt", "")

        if not user_input:
            user_input = "Find software engineer jobs in New York, USA"

        logger.info(f"Processing request: {user_input}")

        # Invoke the Strands agent
        result = agent(user_input)

        logger.info(f"Agent response: {result.message}")

        return {
            "response": result.message
        }

    except Exception as e:
        logger.error(f"Error in agent invocation: {e}")
        return {
            "response": f"Sorry, I encountered an error: {str(e)}"
        }


if __name__ == "__main__":
    app.run()
