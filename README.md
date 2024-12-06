# Tavily Company Research Agent

This agent automates company research by leveraging Tavily to retrieve accurate, up-to-date data. It requires a company name and official website URL, where the URL serves as grounding data to ensure credible results. You can optionally use the "include" argument to specify a list of strings to include in the report, tailoring it to your specific needs. For example, you can request details such as "Company's CEO", "Location of Headquarters", or other specific information.

## Key Steps

1. **🔗 Grounding**: Establishes the website URL as a trusted baseline for all research efforts.
2. **🔎 Searching**: Collects a wide range of relevant data from various online sources.  
3. **📊 Clustering**: Organizes the collected data into clusters, picking the most relevant one. This is especially handy for companies with similar names or limited online visibility.  
4. **🚀 Extraction**: Enriches documents in the chosen cluster.  
5. **📝 Generation**: Creates a detailed company report.  

Our goal is to provide you with a practical tool that helps you effortlessly and efficiently gather meaningful insights on any company.