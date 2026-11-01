# LangGraph Lab

## Project overview

This project is an intelligent assistant developed using Python, LangGraph, and LangChain. It can classify questions into different categories and route them to specialized agents to generate appropriate responses. The assistant can also use external tools for calculations, date and time information, and Wikipedia searches. It supports conversational memory to maintain context within the same conversation.

## Main features

- **Five categories:** Courses, Programming, Mathematics, Administration, and General.
- **Conditional routing:** Selects an agent based on the question.
- **Tools:** Calculator, current date/time, and Wikipedia search.
- **Conversation memory:** Uses `MemorySaver` and `thread_id` to keep conversations separate.
- **Fallback handling:** Responds to empty questions and certain processing errors.

## Technologies

- Python and Jupyter Notebook
- LangGraph and LangChain
- OpenRouter (LLM access)
- Wikipedia API (via `requests`)

## Setup

1. Install the required packages in your Python environment:

   ```bash
   pip install langgraph langchain langchain-openai langchain-community langchain-openrouter python-dotenv requests
   ```

2. Configure your OpenRouter API key in your local environment before running the notebook. Do not share your API key publicly.
3. Open `langgraph_lab_Statement.ipynb` in Jupyter Notebook or VS Code and select the environment where you installed the packages.

## How to run

1. Open `langgraph_lab_Statement.ipynb`.
2. Run the cells **in order**, starting with imports and model configuration.
3. Run the classification, tools, and memory examples.
4. Run the final Intelligent University Assistant demonstration and Wikipedia test.

The Wikipedia extension uses a custom `@tool` function that calls the Wikipedia API with `requests`. The tool must be added to `tools` **before** `llm.bind_tools(tools)` and the final graph is compiled.

## Design choices

- A **classifier node** assigns each question to one of five categories.
- A **specialized agent** answers according to the category.
- A shared **tools node** executes tool calls and returns results to the agent.
- **MemorySaver** preserves conversation history for each `thread_id`.
- A **fallback node** handles empty input and classification errors.

## Testing

The notebook includes tests for:

- Question classification and routing (6/6 sample cases classified correctly).
- Calculator execution: `(15 + 27) * 3 = 126`.
- Current date/time tool calls.
- Memory: remembering a name in one thread without sharing it with another.
- Wikipedia retrieval: searching for Alan Turing.

These are demonstration tests, not a complete reliability evaluation.

## Architecture

See `assistant_architecture.png` for the exported LangGraph diagram.

## Project files

- `langgraph_lab_Statement.ipynb` — executed notebook with code and outputs.
- `assistant_architecture.png` — graph diagram.
- `README.md` — project overview and instructions.
- Additional source code, screenshots, and a short report are included as required by the lab checklist.
