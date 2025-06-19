# Deep Search Research Pipeline

End-to-end research pipeline using OpenAI's GPT-4o and Tavily search API to conduct comprehensive research on any topic.

## Prerequisites

- Python 3.7+
- OpenAI API key
- Tavily API key

## Required API Keys

Set the following environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export TAVILY_API_KEY="your_tavily_api_key_here"
```

### Getting API Keys

1. **OpenAI API Key**: Sign up at [OpenAI Platform](https://platform.openai.com) and create an API key
2. **Tavily API Key**: Sign up at [Tavily](https://tavily.com) to get your search API key

## Installation

Install required dependencies:

```bash
pip install openai requests
```

## Usage

Run the research pipeline:

```bash
python search.py
```

Enter your research question when prompted, and the system will:

1. Generate focused search queries
2. Search the web using Tavily
3. Summarize and evaluate results
4. Filter most relevant sources
5. Generate a comprehensive markdown research report

## Configuration

You can modify these parameters in `search.py`:

- `MAX_TOKENS`: Maximum tokens for final report (default: 8192)
- `BUDGET`: Number of research cycles (default: 2)
- `MAX_QUERIES`: Maximum queries per cycle (default: 2)
- `MAX_SOURCES`: Maximum sources in final report (default: 10)
- `MODEL`: OpenAI model to use (default: "gpt-4o")