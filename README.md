---
sdk: streamlit
python_version: 3.11
---

# ViettelPay Chatbot with RAG

A Vietnamese intelligent chatbot system built for ViettelPay customer support using Retrieval-Augmented Generation (RAG) with contextual enhancement and ensemble retrieval.

This is a mini project for Viettel Digital Talent 2025 with the mentors from Viettel Digital Services.

## Demo
Try the live demo on Hugging Face Spaces: [ViettelPay Chatbot Demo](https://huggingface.co/spaces/minhan6559/viettelpay-chatbot)

## 🚀 Features

- **Interactive Web Interface**: Full Streamlit-based Vietnamese chat interface with conversation management
- **Contextual RAG**: Enhanced document retrieval using contextual processing with LLM enhancement
- **Ensemble Retrieval**: Combines BM25 and semantic search (ChromaDB) for optimal results
- **Vietnamese NLP**: Specialized Vietnamese text processing and tokenization
- **Multi-modal Support**: Processes Word documents (.doc/.docx) and CSV files
- **Intent Classification**: Intelligent intent recognition for user queries
- **Multi-turn Conversations**: Support for conversational context and follow-up questions
- **Reranking**: Cohere reranking for improved result relevance
- **Real-time Analytics**: Intent confidence, response metrics, and system health monitoring
- **Comprehensive Evaluation**: Built-in evaluation framework for performance testing

## 📋 Requirements

- Python 3.11+
- API Keys:
  - Google API Key (for Gemini 2.0 Flash)
  - Cohere API Key (for reranking)
  - OpenAI API Key (optional, for alternative LLM)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Chatbot-ViettelPay
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Note: The project includes Streamlit for the web interface, so make sure it's installed:
```bash
pip install streamlit
```

3. **Set up environment variables**
Create a `.env` file or set environment variables:
```bash
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
```

4. **Build the knowledge base**
```bash
python src/scripts/build_database.py
```

## 🚀 Quick Start

### Demo
Try the live demo on Hugging Face Spaces: [ViettelPay Chatbot Demo](https://huggingface.co/spaces/minhan6559/viettelpay-chatbot)

### Running the Streamlit Web Interface
```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501` and provides:
- 💬 Interactive chat interface in Vietnamese
- 🧵 Multi-turn conversation support
- 📊 Real-time intent classification and confidence metrics
- 🏥 Health check for system components
- 📜 Conversation history management
- 🗑️ Clear conversation and start new sessions

### Building Knowledge Base Programmatically
```python
from src.knowledge_base.viettel_knowledge_base import ViettelKnowledgeBase

# Initialize knowledge base
kb = ViettelKnowledgeBase()

# Build from documents
kb.build_knowledge_base(
    documents_folder="./viettelpay_docs/raw",
    persist_dir="./knowledge_base",
    reset=True
)

# Search
results = kb.search("lỗi 606", top_k=5)
```

### Using the Agent
```python
from src.agent.viettelpay_agent import ViettelPayAgent

# Initialize agent
agent = ViettelPayAgent()

# Process query
response = agent.process_query("Tôi không nạp được tiền vào tài khoản")
print(response)
```

## 📁 Project Structure

```
Chatbot-ViettelPay/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── src/
│   ├── agent/                      # Chatbot agent logic
│   │   ├── viettelpay_agent.py    # Main agent implementation
│   │   ├── nodes.py               # Agent processing nodes
│   │   ├── prompts.py             # System prompts
│   │   └── scripts.py             # Agent utility scripts
│   ├── knowledge_base/            # Knowledge base management
│   │   └── viettel_knowledge_base.py
│   ├── llm/                       # LLM client abstractions
│   │   └── llm_client.py
│   ├── processor/                 # Text processing utilities
│   │   ├── contextual_word_processor.py
│   │   └── text_utils.py
│   ├── evaluation/                # Evaluation framework
│   │   ├── intent_classification.py
│   │   ├── multi_turn_retrieval.py
│   │   └── single_turn_retrieval.py
│   ├── scripts/                   # Build and utility scripts
│   │   └── build_database.py
│   └── utils/                     # Configuration utilities
│       └── config.py
├── viettelpay_docs/               # Document sources
│   ├── raw/                       # Original documents
│   └── processed/                 # Processed documents
├── knowledge_base/                # Vector database (ChromaDB)
└── evaluation_data/               # Evaluation datasets and results
    ├── datasets/
    └── results/
```

## ⚙️ Configuration

The system uses multiple embedding models and LLMs:

- **Default Vietnamese Embedding**: `dangvantuan/vietnamese-document-embedding`
- **Contextual Enhancement**: Gemini 2.0 Flash
- **Reranking**: Cohere Rerank v3.5
- **Vietnamese Tokenization**: underthesea library

Configure these in `src/utils/config.py` or via environment variables.

## 📊 Evaluation

The project includes comprehensive evaluation capabilities:

### Intent Classification
```bash
python src/evaluation/intent_classification.py
```

### Single-turn Retrieval
```bash
python src/evaluation/single_turn_retrieval.py
```

### Multi-turn Conversations
```bash
python src/evaluation/multi_turn_retrieval.py
```

Evaluation results are saved in `evaluation_data/results/`.

## 🔧 Key Components

### ViettelKnowledgeBase
- **Contextual Processing**: Documents are enhanced with contextual information using LLMs
- **Ensemble Retrieval**: Combines keyword (BM25) and semantic (vector) search
- **Vietnamese Optimization**: Specialized for Vietnamese language processing

### ViettelPayAgent
- **Multi-node Processing**: Intent classification → retrieval → response generation
- **Context Awareness**: Maintains conversation history
- **Configurable**: Easy to adjust prompts and behavior

### ContextualWordProcessor
- **Document Enhancement**: Adds context to document chunks using LLMs
- **Multi-format Support**: Handles various document types
- **Intelligent Chunking**: Context-aware document segmentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the evaluation results in `evaluation_data/results/`
- Review the configuration in `src/utils/config.py`
- Ensure all API keys are properly configured

## 🔮 Future Enhancements

- [ ] Real-time learning from user feedback
- [ ] Multi-language support beyond Vietnamese
- [ ] Advanced conversation analytics and dashboards
- [ ] Integration with ViettelPay APIs
- [ ] Voice input/output capabilities
- [ ] Mobile-responsive design improvements