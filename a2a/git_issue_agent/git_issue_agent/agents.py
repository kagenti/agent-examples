from crewai import Agent, Crew, Process, Task
from git_issue_agent.config import Settings
from git_issue_agent.llm import CrewLLM
from git_issue_agent.prompts import TOOL_CALL_PROMPT, INFO_PARSER_PROMPT


class GitAgents:
    def __init__(self, config: Settings, issue_tools):
        self.llm = CrewLLM(config)

        ###################
        # Pre-requisite validator
        # ##################
        self.prereq_identifier = Agent(
            role="Pre-requisite Extractor",
            goal="To extract the information about github artifacts that a user is looking for",
            backstory=INFO_PARSER_PROMPT,
            verbose=True,
            llm=self.llm.llm,
        )

        self.prereq_identifier_task = Task(
            description=("User query: {request}"),
            agent=self.prereq_identifier,
            expected_output=(
                'A JSON object with keys "owner", "repo", and "issue_numbers". '
                'Example: {"owner": "kagenti", "repo": "kagenti", "issue_numbers": null}'
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
                "Answer the user's query, using tools provided from MCP server. "
                "Prefer read-only operations. When querying, be explicit about repo owner/name and filters."
            ),
            backstory=TOOL_CALL_PROMPT,
            tools=issue_tools,
            verbose=True,
            llm=self.llm.llm,
            inject_date=True,
            max_iter=6,
            max_retry_limit=3,
            respect_context_window=True,
        )

        # --- A generic task template -------------------------------------------------
        # The agent will use MCP tools to fulfill natural-language queries.
        self.issue_query_task = Task(
            description=(
                "Retrieve Github issues using tool calls in order to answer the user's query.\n"
                "User query: {request}\n"
                "Identified repository: {repo} \n"
                "Identified owner: {owner} \n"
                "Identified issue numbers: {issues}"
            ),
            agent=self.issue_researcher,
            expected_output=(
                "A well formatted, detailed report, directly answering the user's query, citing the output of the tool to support your answer."
            ),
        )

        self.crew = Crew(
            agents=[self.issue_researcher],
            tasks=[self.issue_query_task],
            process=Process.sequential,
            verbose=True,
        )
