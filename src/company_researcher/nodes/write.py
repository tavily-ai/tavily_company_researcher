from datetime import datetime
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage, ToolMessage


class WriteAgent:
    def __init__(self, cfg, utils):
        self.cfg = cfg
        self.utils = utils

    async def run(self, state):
        report_title = f"{state.company} Company Report"
        report_date = datetime.now().strftime('%B %d, %Y')

        prompt = (
            f"You are an expert company researcher tasked with writing a fact-based report on recent developments for the company **{state.company}**. "
            f"Write the report in Markdown format. DO NOT change the titles. Each section must be written in well-structured paragraphs, not lists or bullet points.\n"
            f"Ensure the report includes:\n"
            f"- **Inline citations** as Markdown hyperlinks directly in the main sections (e.g., Company X is an innovative leader in AI ([LinkedIn](https://linkedin.com))).\n"
            f"- A **Citations Section** at the end that lists all URLs used.\n\n"
            f"### Strict Guidelines:\n"
            f"- You must only use the information provided in the documents listed below.\n"
            f"- Do not make up or infer any details that are not explicitly stated in the provided sources.\n"
            f"- If a required data point (e.g., employee count, financial figures) is not available in the provided documents, state that it is unavailable.\n"
            f"- As of today, **{report_date}**, prioritize the most recent and updated source in cases where conflicting data points or metrics are found.\n"
        )

        if state.include:
            prompt += (
                f"- Ensure the report includes the following user-requested information, if available: "
                f"{', '.join(state.include)}.\n"
            )

        prompt += (
            "- Make sure to support specific data points and metrics included in the report with in-text Markdown hyperlink citations.\n\n"
            f"### Report Structure:\n"
            f"Title: {report_title}\n"
            f"Date: {report_date}\n"
            f"1. **Executive Summary**:\n"
            f"    - High-level overview of the company, its services, location, employee count, and achievements.\n"
            f"    - Make sure to include the general information necessary to understand the company well, including any notable achievements.\n\n"
            f"2. **Leadership and Vision**:\n"
            f"    - Details on the CEO and key team members, their experience, and alignment with company goals.\n"
            f"    - Any personnel changes and their strategic impact.\n\n"
            f"3. **Product and Service Overview**:\n"
            f"    - Summary of current products/services, features, updates, and market fit.\n"
            f"    - Include details from the company's website, tools, or new integrations.\n\n"
            f"4. **Financial Performance**:\n"
            f"    - For public companies: key metrics (e.g., revenue, market cap).\n"
            f"    - For startups: funding rounds, investors, and milestones.\n\n"
            f"5. **Recent Developments**:\n"
            f"    - New product enhancements, partnerships, competitive moves, or market entries.\n\n"
            f"6. **Competitive Landscape**:\n"
            f"    - Overview of major competitors and their positioning in the market.\n"
            f"    - Compare key differentiators, market share, pricing, and product/service features.\n"
            f"    - Include relevant competitor developments that impact the company’s strategy.\n\n"
        )
        if state.include:
            prompt += (
            f"7. (Optional) **Additional Information**:\n"
            f"    - Attempt to fit the user-requested information into the predefined sections above, where relevant.\n"
            f"    - ONLY if the information does not fit into ANY section, include that unfitted information here.\n"
            f"    - AVOID including user-requested information in multiple sections. For example, if the user requests that report includes the company CEO's name, it should be mentioned ONLY in the **Leadership and Vision** section and not repeated here."
            f"    - Present the information in well-structured paragraphs, not lists or bullet points.\n\n"
            )

        prompt += (
            f"{'8' if state.include else '7'}. **Citations**:\n"
            f"    - Ensure every source cited in the report is listed in the text as Markdown hyperlinks.\n"
            f"    - Also include a list of all URLs as Markdown hyperlinks in this section.\n\n"
        )

        # Dynamically generate the "Documents to Base the Report On" section
        if state.clusters:
            # Use cluster-specific research data
            documents = "\n".join(
                f"- {state.research_data[key]}"
                for key in state.clusters[state.chosen_cluster].urls
                if key in state.research_data
            )
            prompt += (
                f"### Documents to Base the Report On:\n"
                f"Use the following cluster-specific documents to write the report:\n"
                f"{documents}"
            )
        else:
            # Use all available research data
            grounding_data_content = "\n".join(f"- {item}" for item in state.grounding_data.values())
            research_data_content = "\n".join(f"- {item}" for item in state.research_data.values())
            prompt += (
                f"### Documents to Base the Report On:\n"
                f"#### Official Grounding Data:\n"
                f"The following is official data sourced from the company's website and should be used as a primary reference:\n"
                f"{grounding_data_content}\n\n"
                f"#### Additional Research Data:\n"
                f"Select and prioritize the most relevant sources to ensure alignment with the target company.\n"
                f"{research_data_content}"
            )

        prompt = prompt[:self.cfg.MAX_PROMPT_LENGTH]
        if self.cfg.DEBUG:
            print(prompt)

        try:
            messages = [SystemMessage(content=prompt)]
            response = await self.cfg.FACTUAL_LLM.ainvoke(messages)
            report = response.content
            return {"report": report}
        except Exception as e:
            msg = f"🚫 Error generating report: {str(e)}"
            if self.cfg.DEBUG:
                print(msg)
            return {"messages": msg}
