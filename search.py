# -*- coding: utf-8 -*-
"""
End-to-end Research Pipeline using OpenAI's GPT-4o
This script implements an agentic research system using OpenAI's APIs for all sub-agent roles,
with real-time web search using Tavily.
"""

import openai
import os
import logging
import requests
from typing import List, Dict

openai.api_key = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Parameters
MAX_TOKENS = 8192
BUDGET = 2
MAX_QUERIES = 2
MAX_SOURCES = 10
MODEL = "gpt-4o"

# System Prompts
PROMPTS = {
    "planning": "You are a strategic research planner. Generate 3 focused search queries to break down the research topic.",
    "plan_parsing": "Extract only the list of queries from the previous response in JSON format.",
    "summarizer": "Extract and synthesize only the content relevant to the research topic from the following text.",
    "evaluation": "Evaluate the following research evidence for completeness. List any gaps or follow-up queries if needed.",
    "evaluation_parsing": "Extract follow-up search queries. If none, return an empty list.",
    "filtering": "Rank the following sources by relevance to the research topic. Return top 5 as a list of source IDs.",
    "answer": "Create a markdown research report including title, intro, analysis sections with citations [Ref X], and conclusion. Do not use bullets."
}

# === Helpers ===
def call_openai(messages: List[Dict], max_tokens=1024) -> str:
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9
    )
    return response["choices"][0]["message"]["content"].strip()

def tavily_search(query: str) -> str:
    logging.info(f"[Search] Tavily querying: {query}")
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": True
    }
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    if "results" in data:
        return "\n\n".join([f"Title: {r.get('title')}\nContent: {r.get('content', '')}" for r in data["results"]])
    return f"No results for: {query}"

# === Agents ===
def planning_agent(question: str) -> List[str]:
    logging.info("[Planner] Generating search queries...")
    messages = [
        {"role": "system", "content": PROMPTS["planning"]},
        {"role": "user", "content": question}
    ]
    response = call_openai(messages)
    logging.info("[Planner Output]\n%s", response)
    return response.split('\n')[:MAX_QUERIES]

def summarizer_agent(content: str, topic: str) -> str:
    logging.info("[Summarizer] Summarizing content for relevance...")
    messages = [
        {"role": "system", "content": PROMPTS["summarizer"]},
        {"role": "user", "content": f"Topic: {topic}\nContent:\n{content}"}
    ]
    response = call_openai(messages)
    return response

def evaluator_agent(summaries: List[str], topic: str) -> List[str]:
    logging.info("[Evaluator] Checking completeness of evidence...")
    combined = '\n'.join(summaries)
    messages = [
        {"role": "system", "content": PROMPTS["evaluation"]},
        {"role": "user", "content": f"Topic: {topic}\nEvidence:\n{combined}"}
    ]
    response = call_openai(messages)
    logging.info("[Evaluator Output]\n%s", response)
    return response.split('\n')

def filter_agent(sources: List[str]) -> List[int]:
    logging.info("[Filter] Ranking sources...")
    indexed = [f"[{i+1}] {src}" for i, src in enumerate(sources)]
    joined = '\n'.join(indexed)
    messages = [
        {"role": "system", "content": PROMPTS["filtering"]},
        {"role": "user", "content": joined}
    ]
    response = call_openai(messages)
    return [int(s.strip()) for s in response if s.strip().isdigit()][:MAX_SOURCES]

def answer_agent(sources: List[str], topic: str) -> str:
    logging.info("[Writer] Generating final report...")
    refs = '\n'.join([f"[Ref {i+1}] {src[:60]}..." for i, src in enumerate(sources)])
    joined = '\n'.join(sources)
    messages = [
        {"role": "system", "content": PROMPTS["answer"]},
        {"role": "user", "content": f"Research Topic: {topic}\nSources:\n{joined}\nReferences:\n{refs}"}
    ]
    return call_openai(messages, max_tokens=MAX_TOKENS)

# === Orchestrator ===
def run_research_pipeline(topic: str):
    queries = planning_agent(topic)
    all_sources = []

    for cycle in range(BUDGET + 1):
        logging.info(f"[Cycle {cycle}] Searching and summarizing...")
        search_results = [tavily_search(q) for q in queries]
        summaries = [summarizer_agent(r, topic) for r in search_results]
        all_sources.extend(summaries)

        new_queries = evaluator_agent(summaries, topic)
        if not any(new_queries):
            break
        queries = new_queries

    filtered = filter_agent(all_sources)
    selected_sources = [all_sources[i-1] for i in filtered if i-1 < len(all_sources)]
    report = answer_agent(selected_sources, topic)
    print("\n===== Final Report =====\n")
    print(report)

# === Main ===
if __name__ == "__main__":
    topic = input("Enter your research question: ")
    run_research_pipeline(topic)

