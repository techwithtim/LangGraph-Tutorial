# LangGraph-Tutorial

Welcome to the LangGraph-Tutorial! This project is a hands-on guide to building a conversational AI chatbot using [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) and [LangChain](https://python.langchain.com/docs/introduction/). The chatbot intelligently classifies user messages as "emotional" or "logical" and responds with empathy or facts accordingly.

## Features

- **Message Classification:** Automatically detects if a user message is emotional or logical.
- **Adaptive Responses:** Provides empathetic replies to emotional messages and factual answers to logical ones.
- **Powered by Claude 3.5 Sonnet:** Utilizes Anthropic Claude 3.5 Sonnet via LangChain for advanced language understanding.

## Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```sh copy
git clone https://github.com/techwithtim/LangGraph-Tutorial.git
cd LangGraph-Tutorial
```

### 2. Create a Virtual Environment (Recommended)

Using `venv`:

```sh copy
uv venv
source .venv/bin/activate  
# On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```sh copy
uv pip install -r pyproject.toml
```

### 4. Set Up API Keys

This project requires access to Anthropic's Claude 3.5 Sonnet. Set your API key as an environment variable:

```sh copy
export ANTHROPIC_API_KEY=your_api_key_here
```

### 5. Run the Chatbot

```sh copy
python main.py
```

## Usage

Once running, type your messages into the terminal. The chatbot will classify and respond to each message appropriately.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.
