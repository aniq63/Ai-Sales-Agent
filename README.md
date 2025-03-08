# AI Sales Agent 🤖💼

## Overview
The **AI Sales Agent** is a virtual assistant designed to automate the role of a salesperson by efficiently selling a company's products. It leverages the **Groq LLM model llama-3.1-8b-instant** to interact with customers professionally and politely, ensuring a seamless sales experience. The agent is customized with the company's branding, product details, and unique persona to enhance user engagement.

## 🌐 Live Demo  
Access the hosted version: https://huggingface.co/spaces/Aniq-63/Ai_Sales_Agent

Live Video Demo : 

## Features
- **Conversational AI**: Engages in natural, human-like conversations.
- **Product Knowledge**: Equipped with deep knowledge of company products,in this csv file are use as database of products
- **Customer Interaction**: Understands customer needs and provides tailored recommendations.
- **Objection Handling**: Addresses customer concerns effectively.
- **Secure Transactions**: Generates payment links (Stripe, etc.) upon customer agreement.
- **Retrieval-Based Responses**: Uses vector embeddings for precise document retrieval.

## Tech Stack
- **Language Model**: Groq's `mixtral-8x7b-32768`
- **Embeddings**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
- **Document Processing**: `langchain_community.document_loaders.CSVLoader`
- **Vector Storage**: `InMemoryVectorStore`
- **Payment Integration**: Stripe (or other payment gateways)
- **Cloud Platform**: Google Colab (for development/testing)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ai-sales-agent.git
   cd ai-sales-agent
   ```
2. Install dependencies:
   ```sh
   pip install langchain langchain_groq langchain_huggingface pandas
   ```
3. Set up your Groq API key in `userdata` (Google Colab):
   ```python
   from google.colab import userdata
   key = userdata.get('groq')
   ```
4. Update the `file_path` in the script with your product CSV file.

## Usage
Run the Python script to start the AI Sales Agent:
```sh
python main.py
```
### Example Interaction:
```
User: Hi, how can you assist me today?
Agent: Hello! I'm Alex, your AI Sales Assistant from TechElectronics. How can I help you with our latest electronics?
```

## Configuration
You can customize the agent's details by modifying the following variables in the script:
```python
company_name = "TechElectronics"
company_business = "Consumer Electronics Retailer"
agent_name = "Alex"
key_features = "Cutting-edge technology, Competitive pricing, Excellent customer service"
```

## Contribution
Feel free to contribute by submitting pull requests or reporting issues.

**Author:** Aniq Ramzan

