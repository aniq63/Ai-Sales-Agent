import os
import sqlite3
import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain.docstore.document import Document

# --- Database Setup ---
@st.cache_resource
def init_db():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  previous_chat_history TEXT,
                  previous_products_bought TEXT)''')
    
    # Company settings table
    c.execute('''CREATE TABLE IF NOT EXISTS company_settings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  business TEXT NOT NULL,
                  agent_name TEXT NOT NULL,
                  key_features TEXT NOT NULL)''')
    
    # Products table with inventory
    c.execute('''CREATE TABLE IF NOT EXISTS products
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  category TEXT NOT NULL,
                  price REAL NOT NULL,
                  description TEXT NOT NULL,
                  features TEXT NOT NULL,
                  stock INTEGER NOT NULL DEFAULT 0)''')
    
    # Check and update schema if needed
    c.execute("PRAGMA table_info(products)")
    columns = [column[1] for column in c.fetchall()]
    if 'stock' not in columns:
        c.execute('ALTER TABLE products ADD COLUMN stock INTEGER NOT NULL DEFAULT 0')
    
    # Insert default company settings if empty
    c.execute('SELECT COUNT(*) FROM company_settings')
    if c.fetchone()[0] == 0:
        c.execute('''INSERT INTO company_settings 
                     (name, business, agent_name, key_features)
                     VALUES (?, ?, ?, ?)''',
                  ('TechElectronics', 
                   'Consumer Electronics Retailer',
                   'Alex',
                   'Cutting-edge technology, Competitive pricing, Excellent customer service'))
    
    conn.commit()
    return conn

conn = init_db()

# --- Admin Classes ---

class Company:
    @staticmethod
    def get_settings():
        c = conn.cursor()
        c.execute('SELECT * FROM company_settings LIMIT 1')
        return c.fetchone()

    @staticmethod
    def update_settings(name, business, agent_name, key_features):
        c = conn.cursor()
        c.execute('''UPDATE company_settings 
                     SET name=?, business=?, agent_name=?, key_features=?
                     WHERE id=1''',
                   (name, business, agent_name, key_features))
        conn.commit()

class Product:
    @staticmethod
    def get_all():
        c = conn.cursor()
        c.execute('SELECT * FROM products')
        return c.fetchall()

    @staticmethod
    def add(name, category, price, description, features, stock):
        c = conn.cursor()
        c.execute('''INSERT INTO products 
                     (name, category, price, description, features, stock)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (name, category, price, description, features, stock))
        conn.commit()

    @staticmethod
    def delete(product_id):
        c = conn.cursor()
        c.execute('DELETE FROM products WHERE id=?', (product_id,))
        conn.commit()

    @staticmethod
    def update_stock(product_id, new_stock):
        c = conn.cursor()
        c.execute('UPDATE products SET stock=? WHERE id=?', (new_stock, product_id))
        conn.commit()

# --- User Class ---
class User:
    def __init__(self, id, username, password, chat_history=None, products_bought=None):
        self.id = id
        self.username = username
        self.password = password
        self.chat_history = chat_history or []
        self.products_bought = products_bought or []

    @classmethod
    def create(cls, username, password):
        hashed_pw = generate_password_hash(password)
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        return cls(user_id, username, hashed_pw)

    @classmethod
    def get_by_username(cls, username):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        if user:
            return cls(user[0], user[1], user[2], 
                      eval(user[3]) if user[3] else [],
                      eval(user[4]) if user[4] else [])
        return None

    def update_chat_history(self, new_messages):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        updated_history = self.chat_history + new_messages
        c.execute('UPDATE users SET previous_chat_history = ? WHERE id = ?',
                 (str(updated_history), self.id))
        conn.commit()
        conn.close()

    def update_products_bought(self, new_products):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        updated_products = self.products_bought + new_products
        c.execute('UPDATE users SET previous_products_bought = ? WHERE id = ?',
                 (str(updated_products), self.id))
        conn.commit()
        conn.close()

# --- AI Setup ---
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    api_key=st.secrets["GROQ_API_KEY"],
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_data():
    products = Product.get_all()
    docs = []
    for p in products:
        content = f"Name: {p[1]}\nCategory: {p[2]}\nPrice: {p[3]}\nDescription: {p[4]}\nFeatures: {p[5]}\nStock: {p[6]}"
        metadata = {"id": p[0], "name": p[1], "category": p[2], "price": p[3], "stock": p[6]}
        docs.append(Document(page_content=content, metadata=metadata))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = load_data()

def retrieve_query(query: str):
    docs = retriever.get_relevant_documents(query)
    st.session_state.last_retrieved_docs = docs  # Store retrieved documents
    return docs

tool = Tool(
    name="product_retriever",
    func=retrieve_query,
    description="Useful for retrieving product information including current stock levels"
)

# --- Admin Dashboard ---
def admin_dashboard():
    st.header("Admin Dashboard")
    
    # Company Settings
    with st.expander("Company Settings"):
        current_settings = Company.get_settings()
        with st.form("Company Settings Form"):
            name = st.text_input("Company Name", value=current_settings[1])
            business = st.text_input("Business", value=current_settings[2])
            agent_name = st.text_input("Agent Name", value=current_settings[3])
            key_features = st.text_area("Key Features", value=current_settings[4])
            
            if st.form_submit_button("Update Settings"):
                Company.update_settings(name, business, agent_name, key_features)
                st.success("Settings updated!")
    
    # Product Management
    with st.expander("Product Management"):
        # Add Product
        with st.form("Add Product"):
            st.subheader("Add New Product")
            name = st.text_input("Product Name")
            category = st.text_input("Category")
            price = st.number_input("Price", min_value=0.0)
            description = st.text_area("Description")
            features = st.text_area("Features")
            stock = st.number_input("Initial Stock", min_value=0, value=0)
            
            if st.form_submit_button("Add Product"):
                Product.add(name, category, price, description, features, stock)
                st.success("Product added!")
                load_data.clear()
        
        # Manage Products
        st.subheader("Manage Inventory")
        products = Product.get_all()
        if products:
            for p in products:
                cols = st.columns([3,2,2,1])
                cols[0].write(f"**{p[1]}** ({p[2]})")
                cols[1].write(f"Price: ${p[3]}")
                new_stock = cols[2].number_input(
                    "Stock", 
                    min_value=0, 
                    value=p[6],
                    key=f"stock_{p[0]}"
                )
                if new_stock != p[6]:
                    Product.update_stock(p[0], new_stock)
                    st.rerun()
                if cols[3].button("❌", key=f"del_{p[0]}"):
                    Product.delete(p[0])
                    st.rerun()
        else:
            st.info("No products found in database")

# --- Main App ---
def main():
    company_settings = Company.get_settings()
    company_name = company_settings[1]

    st.title("AI Sales Assistant 🤖")
    
    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.chat_history = []
        st.session_state.admin_mode = False
        st.session_state.last_retrieved_docs = []
    
    # Authentication
    if not st.session_state.user and not st.session_state.admin_mode:
        st.header("Login/Register Admin")
        tab1, tab2, tab3 = st.tabs(["Login", "Register", "Admin"])
        
        with tab1:
            with st.form("Login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    user = User.get_by_username(username)
                    if user and check_password_hash(user.password, password):
                        st.session_state.user = user
                        st.session_state.chat_history = user.chat_history
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("Register"):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                if st.form_submit_button("Register"):
                    if User.get_by_username(new_user):
                        st.error("Username already exists")
                    else:
                        user = User.create(new_user, new_pass)
                        st.session_state.user = user
                        st.session_state.chat_history = []
                        st.rerun()
        
        with tab3:
            with st.form("Admin Login"):
                admin_pin = st.text_input("Admin PIN", type="password")
                if st.form_submit_button("Admin Login"):
                    if admin_pin == st.secrets["ADMIN_PIN"]:
                        st.session_state.admin_mode = True
                        st.rerun()
                    else:
                        st.error("Invalid Admin PIN")

    elif st.session_state.admin_mode:
        admin_dashboard()
        if st.button("Exit Admin Mode"):
            st.session_state.admin_mode = False
            st.rerun()

    else:
        # Chat Interface
        st.header(f"Welcome to {company_name}, {st.session_state.user.username} 😊!")
        st.subheader("Chat with our AI Sales Assistant")

        # Display Chat History
        for msg in st.session_state.chat_history:
            if msg["type"] == "human":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])

        # System Prompt
        company_settings = Company.get_settings()
        current_products = "\n".join([f"- {p[1]} (${p[3]}, Stock: {p[6]})" for p in Product.get_all()])
        system_prompt = f"""
        You are {company_settings[3]}, the AI Sales Assistant for {company_settings[1]} ({company_settings[2]}).

        Company Profile:
        - Company Name: {company_settings[1]}
        - Business: {company_settings[2]}
        - Key Features: {company_settings[4]}

        Product Availability:
        - Only recommend products with stock greater than zero
        - Always check stock levels before making recommendations
        - Suggest alternatives if requested product is out of stock
        - Never recommend discontinued or unavailable products

        Current Inventory:
        {current_products}

        Sales Process:
        1. Greet and understand customer needs
        2. Qualify requirements
        3. Recommend in-stock products
        4. Provide detailed product information
        5. Handle objections
        6. Close sale with payment link

        Payment Link: https://www.example.com/payment
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        tools = [tool]
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        if prompt_input := st.chat_input("Type your message here..."):
            with st.chat_message("user"):
                st.write(prompt_input)

            with st.chat_message("assistant"):
                response = agent_executor.invoke({
                    "input": prompt_input,
                    "chat_history": [HumanMessage(content=msg["content"]) if msg["type"] == "human" else AIMessage(content=msg["content"]) 
                                   for msg in st.session_state.chat_history]
                })["output"]
                st.write(response)
                
                # Handle inventory update on purchase
                if "https://www.example.com/payment" in response:
                    if st.session_state.last_retrieved_docs:
                        product_doc = st.session_state.last_retrieved_docs[0]
                        product_id = product_doc.metadata.get("id")
                        product_name = product_doc.metadata.get("name")
                        
                        if product_id:
                            # Update stock and user purchase history
                            Product.update_stock(product_id, product_doc.metadata["stock"] - 1)
                            st.session_state.user.update_products_bought([product_name])
                            st.success(f"Purchased {product_name}! Stock updated.")
                            load_data.clear()  # Refresh product data
                    else:
                        st.error("No product selected for purchase")

            new_messages = [
                {"type": "human", "content": prompt_input},
                {"type": "ai", "content": response}
            ]
            st.session_state.user.update_chat_history(new_messages)
            st.session_state.chat_history += new_messages

if __name__ == "__main__":
    main()