"""
Project Recommendation Agent
Uses AWS Bedrock AgentCore with Strands framework to recommend portfolio-ready
projects and skills based on user's career goals.
"""

from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel
import json
import logging
import os
from typing import List, Dict

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Initialize BedrockAgentCore app
app = BedrockAgentCoreApp()


# Project database - curated project ideas organized by category
PROJECT_DATABASE = {
    "web_development": [
        {
            "name": "E-Commerce Platform",
            "description": "Full-stack e-commerce site with product catalog, shopping cart, checkout, and payment integration",
            "skills": ["React/Vue/Angular", "Node.js/Python/Java backend", "PostgreSQL/MongoDB", "Stripe/PayPal API", "Authentication", "RESTful APIs"],
            "difficulty": "Intermediate",
            "duration": "4-6 weeks",
            "portfolio_value": "High - Demonstrates full-stack skills and business logic"
        },
        {
            "name": "Social Media Dashboard",
            "description": "Analytics dashboard that aggregates data from multiple social media platforms",
            "skills": ["Frontend framework", "API integration", "Data visualization", "OAuth", "Real-time updates"],
            "difficulty": "Intermediate",
            "duration": "3-4 weeks",
            "portfolio_value": "High - Shows API integration and data handling"
        },
        {
            "name": "Real-Time Chat Application",
            "description": "WebSocket-based chat with rooms, direct messages, file sharing, and typing indicators",
            "skills": ["WebSockets", "Real-time communication", "Authentication", "File upload", "Database design"],
            "difficulty": "Intermediate",
            "duration": "3-4 weeks",
            "portfolio_value": "High - Demonstrates real-time technologies"
        }
    ],
    "mobile_development": [
        {
            "name": "Fitness Tracking App",
            "description": "Mobile app for tracking workouts, nutrition, and progress with data visualization",
            "skills": ["React Native/Flutter", "Local storage", "Charts/graphs", "Camera integration", "Health APIs"],
            "difficulty": "Intermediate",
            "duration": "4-5 weeks",
            "portfolio_value": "High - Shows mobile expertise and UX design"
        },
        {
            "name": "Expense Tracker",
            "description": "Personal finance app with budget tracking, categories, and spending analytics",
            "skills": ["Mobile development", "SQLite/Realm", "Data visualization", "Export features", "Authentication"],
            "difficulty": "Beginner-Intermediate",
            "duration": "2-3 weeks",
            "portfolio_value": "Medium-High - Practical app with data management"
        }
    ],
    "data_science": [
        {
            "name": "Predictive Analytics Dashboard",
            "description": "ML-powered dashboard for business forecasting with interactive visualizations",
            "skills": ["Python", "Scikit-learn/TensorFlow", "Pandas/NumPy", "Plotly/Dash", "Time series analysis"],
            "difficulty": "Advanced",
            "duration": "5-7 weeks",
            "portfolio_value": "Very High - Combines ML and visualization"
        },
        {
            "name": "Sentiment Analysis Tool",
            "description": "NLP application that analyzes sentiment from social media, reviews, or customer feedback",
            "skills": ["NLP", "Python", "NLTK/spaCy", "Data preprocessing", "API development", "Visualization"],
            "difficulty": "Intermediate-Advanced",
            "duration": "3-4 weeks",
            "portfolio_value": "High - Shows NLP and ML capabilities"
        },
        {
            "name": "Image Classification System",
            "description": "CNN-based image classifier for specific domain (medical, wildlife, products, etc.)",
            "skills": ["Deep learning", "TensorFlow/PyTorch", "Computer vision", "Data augmentation", "Model deployment"],
            "difficulty": "Advanced",
            "duration": "4-6 weeks",
            "portfolio_value": "Very High - Advanced ML project"
        }
    ],
    "machine_learning": [
        {
            "name": "Recommendation Engine",
            "description": "Content recommendation system using collaborative filtering or deep learning",
            "skills": ["ML algorithms", "Python", "Data processing", "Matrix factorization", "Neural networks"],
            "difficulty": "Advanced",
            "duration": "4-5 weeks",
            "portfolio_value": "Very High - Industry-relevant ML application"
        },
        {
            "name": "Fraud Detection System",
            "description": "ML model to detect fraudulent transactions with real-time scoring",
            "skills": ["Classification algorithms", "Feature engineering", "Imbalanced data handling", "Model evaluation", "API deployment"],
            "difficulty": "Advanced",
            "duration": "5-6 weeks",
            "portfolio_value": "Very High - Solves real business problem"
        }
    ],
    "cloud_devops": [
        {
            "name": "Microservices Architecture",
            "description": "Containerized microservices with Docker, Kubernetes, and CI/CD pipeline",
            "skills": ["Docker", "Kubernetes", "CI/CD", "AWS/Azure/GCP", "Monitoring", "Load balancing"],
            "difficulty": "Advanced",
            "duration": "5-7 weeks",
            "portfolio_value": "Very High - Shows modern DevOps practices"
        },
        {
            "name": "Infrastructure as Code Platform",
            "description": "Automated cloud infrastructure provisioning using Terraform/CloudFormation",
            "skills": ["Terraform", "AWS/Azure", "Automation", "Networking", "Security", "Documentation"],
            "difficulty": "Intermediate-Advanced",
            "duration": "3-4 weeks",
            "portfolio_value": "High - Demonstrates IaC expertise"
        }
    ],
    "cybersecurity": [
        {
            "name": "Security Vulnerability Scanner",
            "description": "Automated tool to scan web applications for common vulnerabilities (XSS, SQL injection, etc.)",
            "skills": ["Security testing", "Python", "Web scraping", "OWASP Top 10", "Reporting"],
            "difficulty": "Advanced",
            "duration": "4-5 weeks",
            "portfolio_value": "Very High - Shows security expertise"
        },
        {
            "name": "Password Manager",
            "description": "Secure password storage application with encryption and browser integration",
            "skills": ["Cryptography", "Security best practices", "Desktop/mobile dev", "Database encryption"],
            "difficulty": "Intermediate-Advanced",
            "duration": "3-4 weeks",
            "portfolio_value": "High - Demonstrates security focus"
        }
    ],
    "ai_llm": [
        {
            "name": "RAG-Based Chatbot",
            "description": "Retrieval-Augmented Generation chatbot using vector databases and LLMs",
            "skills": ["LangChain/LlamaIndex", "Vector databases", "OpenAI/Anthropic APIs", "Embeddings", "RAG architecture"],
            "difficulty": "Advanced",
            "duration": "4-6 weeks",
            "portfolio_value": "Very High - Cutting-edge AI application"
        },
        {
            "name": "AI Code Assistant",
            "description": "IDE plugin that helps with code completion, documentation, and refactoring using LLMs",
            "skills": ["LLM APIs", "IDE integration", "Prompt engineering", "Code parsing", "Testing"],
            "difficulty": "Advanced",
            "duration": "5-7 weeks",
            "portfolio_value": "Very High - Innovative AI tool"
        }
    ],
    "blockchain": [
        {
            "name": "NFT Marketplace",
            "description": "Decentralized marketplace for creating, buying, and selling NFTs",
            "skills": ["Solidity", "Web3.js/Ethers.js", "Smart contracts", "IPFS", "MetaMask integration"],
            "difficulty": "Advanced",
            "duration": "6-8 weeks",
            "portfolio_value": "Very High - Demonstrates blockchain expertise"
        },
        {
            "name": "DeFi Yield Aggregator",
            "description": "Platform that finds and optimizes yield farming opportunities across protocols",
            "skills": ["Smart contracts", "DeFi protocols", "Web3", "Financial calculations", "Security auditing"],
            "difficulty": "Advanced",
            "duration": "7-9 weeks",
            "portfolio_value": "Very High - Complex DeFi application"
        }
    ],
    "general": [
        {
            "name": "Portfolio Website with CMS",
            "description": "Personal portfolio with custom CMS for managing projects, blog, and contact",
            "skills": ["Frontend", "Backend", "CMS", "SEO", "Responsive design", "Deployment"],
            "difficulty": "Beginner-Intermediate",
            "duration": "2-3 weeks",
            "portfolio_value": "Medium - Essential for all developers"
        },
        {
            "name": "CLI Tool for Developers",
            "description": "Command-line utility that solves a specific developer workflow problem",
            "skills": ["Python/Go/Rust", "CLI frameworks", "Package management", "Documentation", "Testing"],
            "difficulty": "Intermediate",
            "duration": "2-3 weeks",
            "portfolio_value": "Medium-High - Shows practical problem-solving"
        }
    ]
}

# Skills database
SKILLS_DATABASE = {
    "frontend": ["React", "Vue.js", "Angular", "TypeScript", "Tailwind CSS", "Next.js", "State management", "Responsive design"],
    "backend": ["Node.js", "Python (Django/Flask)", "Java (Spring)", "Go", "RESTful APIs", "GraphQL", "Microservices"],
    "database": ["PostgreSQL", "MongoDB", "Redis", "Database design", "Query optimization", "Migrations"],
    "devops": ["Docker", "Kubernetes", "CI/CD", "AWS", "Azure", "Terraform", "Monitoring"],
    "ml_ai": ["Python ML libraries", "TensorFlow/PyTorch", "NLP", "Computer vision", "LLMs", "RAG", "MLOps"],
    "mobile": ["React Native", "Flutter", "iOS (Swift)", "Android (Kotlin)", "Mobile UI/UX"],
    "security": ["OWASP Top 10", "Cryptography", "Penetration testing", "Security auditing", "Authentication"],
    "blockchain": ["Solidity", "Web3", "Smart contracts", "DeFi", "Ethereum"],
    "soft_skills": ["Git/GitHub", "Agile", "Documentation", "Testing", "Code review", "Communication"]
}


@tool
def get_project_recommendations(career_goal: str, experience_level: str = "intermediate") -> Dict:
    """
    Get 3 portfolio-ready project recommendations based on career goal.

    Args:
        career_goal: Target career or role (e.g., 'full-stack developer', 'data scientist', 'ML engineer')
        experience_level: User's current level ('beginner', 'intermediate', 'advanced')

    Returns:
        Dictionary with 3 recommended projects and their details
    """
    try:
        logger.info(f"Getting project recommendations for: {career_goal} ({experience_level})")

        # Map career goals to project categories
        career_goal_lower = career_goal.lower()

        selected_categories = []

        # Match keywords to categories
        if any(word in career_goal_lower for word in ["web", "full-stack", "frontend", "backend"]):
            selected_categories.append("web_development")
        if any(word in career_goal_lower for word in ["mobile", "ios", "android", "app"]):
            selected_categories.append("mobile_development")
        if any(word in career_goal_lower for word in ["data scien", "analytics", "data analy"]):
            selected_categories.append("data_science")
        if any(word in career_goal_lower for word in ["machine learning", "ml engineer", "ai engineer", "deep learning"]):
            selected_categories.append("machine_learning")
            selected_categories.append("data_science")
        if any(word in career_goal_lower for word in ["devops", "sre", "cloud", "infrastructure"]):
            selected_categories.append("cloud_devops")
        if any(word in career_goal_lower for word in ["security", "cybersecurity", "penetration"]):
            selected_categories.append("cybersecurity")
        if any(word in career_goal_lower for word in ["llm", "gpt", "ai", "chatbot", "rag"]):
            selected_categories.append("ai_llm")
        if any(word in career_goal_lower for word in ["blockchain", "web3", "defi", "nft", "crypto"]):
            selected_categories.append("blockchain")

        # Default to general if no matches
        if not selected_categories:
            selected_categories = ["general", "web_development"]

        # Collect projects from selected categories
        all_projects = []
        for category in selected_categories:
            if category in PROJECT_DATABASE:
                all_projects.extend(PROJECT_DATABASE[category])

        # Filter by experience level if specified
        if experience_level.lower() != "all":
            filtered_projects = []
            for project in all_projects:
                proj_difficulty = project["difficulty"].lower()
                if experience_level.lower() in proj_difficulty or experience_level.lower() == "intermediate":
                    filtered_projects.append(project)

            if filtered_projects:
                all_projects = filtered_projects

        # Select top 3 projects (prioritize by portfolio value)
        value_order = {"Very High": 4, "High": 3, "Medium-High": 2, "Medium": 1}
        all_projects.sort(key=lambda x: value_order.get(x.get("portfolio_value", "Medium"), 0), reverse=True)

        recommended_projects = all_projects[:3]

        logger.info(f"Recommending {len(recommended_projects)} projects")

        return {
            "career_goal": career_goal,
            "experience_level": experience_level,
            "project_count": len(recommended_projects),
            "projects": recommended_projects
        }

    except Exception as e:
        logger.error(f"Error in get_project_recommendations: {e}", exc_info=True)
        return {"error": f"Failed to get project recommendations: {str(e)}"}


@tool
def get_skill_recommendations(career_goal: str, skill_categories: List[str] = None) -> Dict:
    """
    Get recommended skills to acquire for a specific career goal.

    Args:
        career_goal: Target career or role
        skill_categories: Optional list of categories to focus on (e.g., ['frontend', 'backend'])

    Returns:
        Dictionary with recommended skills organized by category
    """
    try:
        logger.info(f"Getting skill recommendations for: {career_goal}")

        career_goal_lower = career_goal.lower()

        # Auto-detect relevant skill categories if not provided
        if not skill_categories:
            skill_categories = []

            if any(word in career_goal_lower for word in ["frontend", "react", "vue", "angular"]):
                skill_categories.extend(["frontend", "soft_skills"])
            if any(word in career_goal_lower for word in ["backend", "api", "server"]):
                skill_categories.extend(["backend", "database", "soft_skills"])
            if any(word in career_goal_lower for word in ["full-stack", "full stack"]):
                skill_categories.extend(["frontend", "backend", "database", "devops", "soft_skills"])
            if any(word in career_goal_lower for word in ["data scien", "analytics", "ml", "ai"]):
                skill_categories.extend(["ml_ai", "database", "soft_skills"])
            if any(word in career_goal_lower for word in ["devops", "sre", "cloud"]):
                skill_categories.extend(["devops", "backend", "soft_skills"])
            if any(word in career_goal_lower for word in ["mobile", "ios", "android"]):
                skill_categories.extend(["mobile", "soft_skills"])
            if any(word in career_goal_lower for word in ["security", "cybersecurity"]):
                skill_categories.extend(["security", "backend", "soft_skills"])
            if any(word in career_goal_lower for word in ["blockchain", "web3"]):
                skill_categories.extend(["blockchain", "backend", "soft_skills"])

            # Default
            if not skill_categories:
                skill_categories = ["frontend", "backend", "soft_skills"]

        # Get skills from relevant categories
        recommended_skills = {}
        for category in skill_categories:
            if category in SKILLS_DATABASE:
                recommended_skills[category] = SKILLS_DATABASE[category]

        logger.info(f"Recommending skills from {len(recommended_skills)} categories")

        return {
            "career_goal": career_goal,
            "skill_categories": list(recommended_skills.keys()),
            "skills": recommended_skills
        }

    except Exception as e:
        logger.error(f"Error in get_skill_recommendations: {e}", exc_info=True)
        return {"error": f"Failed to get skill recommendations: {str(e)}"}


# Configure the Strands agent with Amazon Nova Pro
bedrock_model = BedrockModel(
    model_id="amazon.nova-pro-v1:0",
    region_name=AWS_REGION,
    max_tokens=4000  # Higher limit for detailed project recommendations
)

# Create the project advisor agent
agent = Agent(
    model=bedrock_model,
    name="ProjectAdvisorAgent",
    system_prompt="""You are an expert career advisor specializing in helping developers and tech professionals build impressive portfolios through strategic project selection.

Your role is to recommend 3 concrete, portfolio-ready projects and the skills users should acquire based on their career goals.

When a user describes their career goal or target role:
1. Analyze the key technical areas and skills required for that role
2. Use get_project_recommendations to find 3 relevant, portfolio-worthy projects
3. Use get_skill_recommendations to identify critical skills to develop
4. Present recommendations in a clear, actionable format
5. Explain WHY each project is valuable for their career goal
6. Provide realistic timelines and difficulty assessments
7. Suggest a learning path for acquiring the recommended skills

For each recommended project:
- Explain its relevance to the career goal
- Highlight what it demonstrates to employers
- Mention the key technologies involved
- Give a realistic time estimate
- Suggest how to make it stand out

For recommended skills:
- Organize by priority (must-have vs. nice-to-have)
- Suggest learning resources or approaches
- Explain how each skill applies to the role

Be specific, practical, and encouraging. Your recommendations should help users build a competitive portfolio that showcases their capabilities to potential employers.

Remember: Quality over quantity. 3 well-executed projects are better than 10 mediocre ones.""",
    tools=[get_project_recommendations, get_skill_recommendations]
)


@app.entrypoint
def invoke_agentcore(payload):
    """
    AgentCore entrypoint using Strands framework.
    Handles project and skill recommendations based on career goals.
    """
    try:
        # Extract user input
        user_input = payload.get("inputText", "") or payload.get("prompt", "")

        if not user_input:
            user_input = "I want to become a full-stack developer. What projects should I build?"

        logger.info(f"Processing project recommendation request: {user_input}")

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
    logger.info("Starting Project Advisor Agent on port 8080...")
    app.run()
