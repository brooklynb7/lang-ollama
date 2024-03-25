from dotenv import load_dotenv

from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, WebsiteSearchTool

load_dotenv()
search_tool = SerperDevTool()
web_search_tool = WebsiteSearchTool()

# Define your agents with roles and goals
researcher = Agent(
    role="Product Owner",
    goal="Provide the requirements of a product",
    backstory="""You work at a leading tech company.
  Your expertise lies in create the detail requirements for an application or product.""",
    verbose=True,
    allow_delegation=False,
    tools=[],
)
writer = Agent(
    role="SAP CAP developer",
    goal="Use the output from Product Owner to write codes to implement the product requirements by using SAP CAP program",
    backstory="""You are an expert of SAP CAP appliction. You can write down the code to complete the project requirements""",
    verbose=True,
    tools=[],
    allow_delegation=False,
)

# Create tasks for your agents
task1 = Task(
    description="""Create the requirements for the product: Book Management""",
    expected_output="Detailed requirements with user stories",
    agent=researcher,
)

task2 = Task(
    description="""Depends on the requirements, write down the code to complete the requirements.""",
    expected_output="SAP CAP application code delivered.",
    agent=writer,
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
