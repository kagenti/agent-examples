from crewai import Agent, Crew, Process, Task
from git_issue_agent.config import Settings
from git_issue_agent.data_types import RepositoryJudgement
from git_issue_agent.llm import CrewLLM
from git_issue_agent.prompts import REPO_ID_BACKSTORY, TOOL_CALL_PROMPT
class GitAgents():
    
    def __init__(self, config: Settings, issue_tools):
        self.llm = CrewLLM(config)

        ###################
        # Pre-requisitie validator
        # ##################
        self.prereq_identifier = Agent(
            role="GitHub Issue Analyst",
            goal="To determine whether a user has supplied enough information to find issues contained in Github repositories.",
            backstory=REPO_ID_BACKSTORY,
            verbose=True,
            llm=self.llm.llm
        )

        self.prereq_identifier_task = Task(
            description=(
                "User query: {request}"
            ),
            agent=self.prereq_identifier,
            output_pydantic=RepositoryJudgement,
            expected_output=(
                "A judgement on whether a request from a user successfully identifies both a repository owner/organization and repository name, accompanied by an explanation"
            ),
        )

        self.prereq_id_crew = Crew(
            agents=[self.prereq_identifier],
            tasks=[self.prereq_identifier_task],
            process=Process.sequential,
            verbose=True,           
        )

        ###################
        # Issue Researcher
        # ##################
        self.issue_researcher = Agent(
            role="GitHub Issue Analyst",
            goal=(
                "You are connected to GitHub's MCP server and specialize in exploring and summarizing repository issues. "
                "Prefer read-only operations. When querying, be explicit about repo owner/name and filters."
            ),
            backstory=TOOL_CALL_PROMPT,
            tools=issue_tools,
            verbose=True,
            llm=self.llm.llm,
            max_iter=3
        )
            
        # --- A generic task template -------------------------------------------------
        # The agent will use MCP tools to fulfill natural-language queries.
        self.issue_query_task = Task(
            description=(
                "Retrieve Github issues using tool calls in order to answer the user's query.\n"
                "Instructions:\n"
                "1) Identify the criteria they specify such as username, organization and/or repository name.\n"
                "2) If the user provides filters (e.g., state=open, label=bug, assignee=alice, or search text), apply them.\n"
                "3) Return a clean, numbered summary: issue number, title, state, labels, assignee(s), and direct URL.\n"
                "4) Prefer listing or searching for issues over the get_issues API unless the user gives you a specific issue number"
                "User query: {request}"
            ),
            agent=self.issue_researcher,
            expected_output=(
                "A concise report (Markdown allowed) listing matching issues with links and key metadata, "
                "or a brief explanation if nothing matches."
            ),
        )

        self.crew = Crew(
            agents=[self.issue_researcher],
            tasks=[self.issue_query_task],
            process=Process.sequential,
            verbose=True,
        )