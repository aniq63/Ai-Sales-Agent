# AI Sales Agent 🤖💼

## Overview
The **AI Sales Agent** is a virtual assistant designed to automate the role of a salesperson by efficiently selling a company's products. It leverages the **Groq LLM model llama-3.1-8b-instant** to interact with customers professionally and politely, ensuring a seamless sales experience. The agent is customized with the company's branding, product details, and unique persona to enhance user engagement.

## 🌐 Live Demo  
Access the hosted version: https://huggingface.co/spaces/Aniq-63/Ai_Sales_Agent

Live Video Demo : 


## Features ✨

- **CSV Data Integration**: Process customer/product data from CSV files
- **Dynamic Conversational Flow**: 7-step sales process implementation
- **Real-time Document Retrieval**: Semantic search through uploaded data
- **Customizable Agent Identity**: Configure agent name and company details
- **Secure Payment Integration**: Automatic payment link generation via Stripe e.t.c, the Company integrate its Payment link

## Technologies Used 🛠️

- **Groq API**: LLM inference with Llama-3.1-8b-instant
- **LangChain**: Agent orchestration and tool integration
- **Streamlit**: Web interface and user interaction
- **Hugging Face**: Sentence Transformers for embeddings
- **In-Memory Vector Store**: Real-time document retrieval

## Installation ⚙️

1. Clone repository:
```bash
git clone https://github.com/yourusername/ai-sales-assistant.git
cd ai-sales-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export GROQ_API_KEY=your_api_key_here
```

## Usage 🚀

1. Start the application:
```bash
streamlit run app.py
```

2. In the sidebar:
   - Upload CSV file with company data
   - Configure agent details
   - Enter GROQ API key

3. Chat with the AI assistant through the main interface

## CSV Format Requirements 📋

Sample CSV structure:
```csv
product_name,description,price,features
"SmartPhone X","Flagship smartphone",799,"5G, 128GB storage, 48MP camera"
```

## Configuration Options ⚡

- **Agent Identity**:
  - Name
  - Company details
  - Key features

- **Model Settings**:
  - Temperature (via code)
  - Chunk size/overlap (text processing)

## Conversation Flow 🔄

1. Initial greeting and introduction
2. Customer qualification
3. Needs assessment
4. Product recommendation
5. Feature highlighting
6. Purchase confirmation
7. Payment link delivery

## Future Improvements 🚀

Here are some potential enhancements and features that could be added to the AI Sales Assistant:

### 1. **Enhanced Data Integration**
   - **Support for Multiple File Formats**: Extend support to Excel, JSON, and other data formats.
   - **Database Integration**: Allow direct connection to databases (e.g., SQL, MongoDB) for real-time data retrieval.
   - **API Integration**: Enable integration with external APIs to fetch live product data or customer information.

### 2. **Advanced Natural Language Processing**
   - **Sentiment Analysis**: Detect customer sentiment during conversations to tailor responses.
   - **Multilingual Support**: Add support for multiple languages to cater to a global audience.
   - **Contextual Memory**: Improve long-term memory to maintain context across multiple sessions.


### 4. **Advanced Agent Capabilities**
   - **Multi-Agent Systems**: Implement multiple agents specializing in different areas (e.g., sales, support, technical).
   - **Role-Based Access Control**: Allow different levels of access and capabilities based on user roles.
   - **Automated Follow-ups**: Schedule and send follow-up emails or messages based on conversation outcomes.


### 5. **Mobile and Cross-Platform Support**
   - **Mobile App**: Develop mobile applications for iOS and Android.
   - **Cross-Platform Compatibility**: Ensure seamless operation across different devices and platforms.
   - **Offline Mode**: Implement offline capabilities for uninterrupted service.

---

These improvements aim to enhance the functionality, usability, and scalability of the AI Sales Assistant, making it a more powerful tool for businesses and users alike.


**Note**: Requires valid GROQ API key for operation. CSV file processing done in temporary memory - uploaded files are not stored permanently.
