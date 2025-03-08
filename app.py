import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

# Set up page configuration
st.set_page_config(page_title="AI Sales Assistant", page_icon="🤖")
st.title("🤖 AI Sales Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    # Company and agent details
    agent_name = st.text_input("Agent Name", value="Alex")
    company_name = st.text_input("Company Name", value="TechElectronics")
    company_business = st.text_input("Company Business", value="Consumer Electronics Retailer")
    key_features = st.text_area("Key Features", 
                              value="Cutting-edge technology, Competitive pricing, Excellent customer service")
    
    # API key input
    groq_api_key = st.text_input("GROQ API Key", type="password")

# Check for API key
if not groq_api_key:
    st.info("Please enter your GROQ API key in the sidebar to continue.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# Process CSV and initialize system if file is uploaded
if uploaded_file is not None:
    # Save uploaded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Load and process CSV
    try:
        loader = CSVLoader(file_path=tmp_file_path)
        docs = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        splits = text_splitter.split_documents(docs)
        
        # Create vector store
        @st.cache_resource
        def load_embeddings():
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        embeddings = load_embeddings()
        vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # Define retrieval tool
        def retrieve_query(query: str):
            docs = retriever.get_relevant_documents(query)
            return docs

        tool = Tool(
            name="retriever",
            func=retrieve_query,
            description="Useful for retrieving documents related to a query"
        )
        
        # System prompt template
        system_prompt = """
        You are {agent_name}, the AI Sales Assistant for {company_name} ({company_business}).

        Company Profile:
        - Company Name: {company_name}
        - Business: {company_business}
        - Key Features: {key_features}

        Conversation Flow:
        1. Introduction
        2. Qualification
        3. Understanding Needs
        4. Needs Analysis
        5. Solution Presentation
        6. Confirmation
        7. If the prospect agrees to purchase, thank them and provide the payment link: https://www.example.com/payment 

        Guidelines:
        - Maintain natural, professional conversations
        - Follow company policies
        - Be helpful and polite
        """

        # Format system prompt
        formatted_system_prompt = system_prompt.format(
            agent_name=agent_name,
            company_name=company_name,
            company_business=company_business,
            key_features=key_features
        )

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Initialize LLM
        llm = ChatGroq(
            temperature=0.1,
            model_name="llama-3.1-8b-instant",
        )

        # Create agent
        tools = [tool]
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
    finally:
        os.unlink(tmp_file_path)  # Clean up temporary file

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Convert chat history to LangChain messages
        chat_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        # Get AI response
        try:
            response = agent_executor.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            ai_response = response["output"]
        except Exception as e:
            ai_response = f"Sorry, I encountered an error: {str(e)}"

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)

else:
    st.info("Please upload a CSV file in the sidebar to get started.")
