from datetime import datetime
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage, ToolMessage


class WriteAgent:
    def __init__(self, cfg, utils):
        self.cfg = cfg
        self.utils = utils

    async def run(self, state):
        report_title = f"{state.company.capitalize()} Company Report"
        report_date = datetime.now().strftime('%B %d, %Y')

        # Dynamically generate the "Documents to Base the Report On" section
        if state.clusters:
            # Use cluster-specific research data
            documents = "\n".join(
                f"- {state.research_data[key]}"
                for key in state.clusters[state.chosen_cluster].urls
                if key in state.research_data
            )
            documents_section = f"""
            ### Documents to Base the Report On:
            Use the following cluster-specific documents to write the report:
            {documents}"""
        else:
            # Use all available research data
            grounding_data_content = "\n".join(f"- {item}" for item in state.grounding_data.values())
            research_data_content = "\n".join(f"- {item}" for item in state.research_data.values())
            documents_section = f"""
            ### Documents to Base the Report On:
            #### Official Grounding Data:
            The following is official data sourced from the company's website and should be used as a primary reference:
            {grounding_data_content}

            #### Additional Research Data:
            Select and prioritize the most relevant sources to ensure alignment with the target company.
            {research_data_content}"""

        prompt = f"""
        You are an expert company researcher tasked with writing a fact-based report on recent developments for the company **{state.company}**. Write the report in Markdown format, but **do not include a title**. Each section must be written in well-structured paragraphs, not lists or bullet points.
        Ensure the report includes:
        - **Inline citations** as Markdown hyperlinks directly in the main sections (e.g., Company X is an innovative leader in AI ([LinkedIn](https://linkedin.com))).
        - A **Citations Section** at the end that lists all URLs used.

        ### Report Structure:
        1. **Executive Summary**:
            - High-level overview of the company, its services, location, employee count, and achievements.
            - Make sure to include the general information necessary to understand the company well including any notable achievements.

        2. **Leadership and Vision**:
            - Details on the CEO and key team members, their experience, and alignment with company goals.
            - Any personnel changes and their strategic impact.

        3. **Product and Service Overview**:
            - Summary of current products/services, features, updates, and market fit.
            - Include details from the company's website, tools, or new integrations.

        4. **Financial Performance**:
            - For public companies: key metrics (e.g., revenue, market cap).
            - For startups: funding rounds, investors, and milestones.

        5. **Recent Developments**:
            - New product enhancements, partnerships, competitive moves, or market entries.

        6. **Citations**:
            - Ensure every source cited in the report is listed in the text as Markdown hyperlinks.
            - Also include a list of all URLs as Markdown hyperlinks in this section.

        {documents_section}
        """
        prompt = prompt[:self.cfg.MAX_PROMPT_LENGTH]
        print(prompt)
        try:
            messages = [SystemMessage(content=prompt)]
            response = await self.cfg.model.ainvoke(messages)
            report = f"# {report_title}\n\n*{report_date}*\n\n{response.content}"
            return {"report": report}
        except Exception as e:
            msg = f"🚫 Error generating report: {str(e)}"
            return {"messages": msg}
