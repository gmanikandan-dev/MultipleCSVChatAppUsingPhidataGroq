import os
import streamlit as st
import pandas as pd
from phi.assistant import Assistant
from phi.llm.groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="CSV Chat App", layout="wide")

# App title and description
st.title("🤖 Multiple CSV Chat App")
st.markdown("Upload multiple CSV files and chat with your data")

# Get API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# Sidebar for model selection and file uploads
with st.sidebar:
    st.header("Configuration")
    
    # Display API key status
    if groq_api_key:
        st.success("GROQ_API_KEY loaded from .env file")
    else:
        # Fallback to manual input if not in environment
        groq_api_key = st.text_input("GROQ_API_KEY not found in .env. Enter it here:", type="password")
    
    model_name = st.selectbox(
        "Select Groq Model:",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    )
    
    st.header("Upload CSV Files")
    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} files uploaded")

# Function to process uploaded files
def process_uploaded_files(files):
    dataframes = {}
    for uploaded_file in files:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            # Store the dataframe with filename as key
            dataframes[uploaded_file.name] = df
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
    return dataframes

# Initialize or get chat history from session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize or get dataframes from session state
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}

# Process uploaded files and store in session state
if uploaded_files:
    st.session_state.dataframes = process_uploaded_files(uploaded_files)

# Display uploaded CSV data
if st.session_state.dataframes:
    st.header("Uploaded Data")
    
    # Create tabs for each CSV file
    tabs = st.tabs([file_name for file_name in st.session_state.dataframes.keys()])
    
    # Display each dataframe in its respective tab
    for i, (file_name, df) in enumerate(st.session_state.dataframes.items()):
        with tabs[i]:
            st.write(f"File: {file_name}, Shape: {df.shape}")
            st.dataframe(df)

# Function to create a system prompt based on the dataframes
def create_system_prompt():
    prompt = "You are a helpful assistant that can analyze CSV data. "
    
    if st.session_state.dataframes:
        prompt += "The user has uploaded the following CSV files:\n\n"
        
        for file_name, df in st.session_state.dataframes.items():
            prompt += f"File: {file_name}\n"
            prompt += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            prompt += f"Columns: {', '.join(df.columns.tolist())}\n"
            
            # Add sample data (first 5 rows)
            prompt += f"Sample data (first 5 rows):\n"
            prompt += df.head(5).to_string() + "\n\n"
    
    prompt += "When answering questions about the data, be specific about which file you're referring to. "
    prompt += "You can perform aggregation tasks like counting, summing, averaging, and finding the maximum, minimum, and average values of columns."
    return prompt

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get response from Groq
def get_groq_response(user_prompt):
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar or add it to your .env file")
        return "Please provide a Groq API key to continue."
    
    try:
        # Initialize Groq
        llm = Groq(
            model=model_name,
            api_key=groq_api_key,
        )
        
        # Create assistant with system prompt based on uploaded data
        assistant = Assistant(
            llm=llm,
            system_prompt=create_system_prompt()
        )
        
        # Get response - collect full response from generator
        response_generator = assistant.chat(message=user_prompt)
        
        # Convert the generator to a full response
        # If response_generator is an object with a content attribute
        if hasattr(response_generator, 'content'):
            return response_generator.content
        
        # If response_generator is a generator, collect the response chunks
        full_response = ""
        try:
            for chunk in response_generator:
                if isinstance(chunk, str):
                    full_response += chunk
                elif hasattr(chunk, 'content'):
                    full_response += chunk.content
                elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                    full_response += chunk.delta.content
        except Exception as e:
            # If it's not actually a generator (might be a direct string response)
            if isinstance(response_generator, str):
                full_response = response_generator
            else:
                st.error(f"Error processing response chunks: {e}")
                return f"Error processing response: {str(e)}"
        
        return full_response
    except Exception as e:
        st.error(f"Error communicating with Groq: {e}")
        return f"Error: {str(e)}"

# Chat input and response
if prompt := st.chat_input("Ask about your CSV data..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant thinking
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if we have uploaded files
            if not st.session_state.dataframes:
                response = "Please upload at least one CSV file to begin analyzing data."
            else:
                # Get response from Groq
                response = get_groq_response(prompt)
            
            # Display response
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add instructions at the bottom
st.markdown("---")
st.markdown("""
### How to use this app:
1. Create a `.env` file in the same directory with `GROQ_API_KEY=your_api_key_here`
2. Upload one or more CSV files
3. Chat with the assistant about your data
4. Ask questions like:
   - "Summarize the data in file1.csv"
   - "What's the average value of column X?"
   - "Compare the distributions in file1.csv and file2.csv"
   - "Find correlations between columns in file3.csv"
""")