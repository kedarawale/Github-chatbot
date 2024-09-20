import os
import time
import streamlit as st
from github import Github
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, AgentOutputParser
from langchain.agents import LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
import re
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Set up GitHub access token and Google API key
github_token = os.getenv("GITHUB_TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not github_token or not google_api_key:
    st.error("Please set the GITHUB_TOKEN and GOOGLE_API_KEY environment variables.")
    st.stop()

# Base directory for storing vector stores
VECTOR_STORE_BASE_DIR = "vector_stores"

# SentenceTransformerEmbeddings class (unchanged)
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

    def __call__(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)

# Helper functions (get_repository_content, create_or_load_vector_store) remain unchanged

@st.cache_data
def get_repository_content(repo_url):
    # Initialize the Github client
    g = Github(github_token)
    
    # Get the repository
    repo = g.get_repo(repo_url.split('github.com/')[-1])
    
    # Get all files in the repository
    all_files = []
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            all_files.append(file_content)
    
    return all_files

@st.cache_resource(ttl=3600)
def create_or_load_vector_store(_all_files, repo_name):
    vector_store_path = os.path.join(VECTOR_STORE_BASE_DIR, repo_name)
    
    if os.path.exists(vector_store_path):
        # Load existing vector store
        print(f"Loading existing vector store for {repo_name}")
        vector_store = FAISS.load_local(vector_store_path, SentenceTransformerEmbeddings(),allow_dangerous_deserialization=True)
    else:
        # Create new vector store
        print(f"Creating new vector store for {repo_name}")
        
        # Process and split the content of all files
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = []
        for file in _all_files:
            try:
                content = file.decoded_content.decode('utf-8')
                chunks = text_splitter.split_text(content)
                texts.extend(chunks)
            except Exception as e:
                print(f"Error processing file {file.path}: {str(e)}")
        
        # Create and save the vector store
        vector_store = FAISS.from_texts(texts, SentenceTransformerEmbeddings())
        vector_store.save_local(vector_store_path)
    
    return vector_store

class RepositoryData:
    def __init__(self, name, url, files, vector_store):
        self.name = name
        self.url = url
        self.files = files
        self.vector_store = vector_store

    def get_file_list(self):
        return [file.path for file in self.files]

    def get_file_content(self, file_path):
        for file in self.files:
            if file.path == file_path:
                return file.decoded_content.decode('utf-8')
        return None

def setup_qa_chain(vector_store):
    template = """You are an AI assistant specialized in analyzing GitHub repositories. Your task is to provide specific and detailed information based solely on the content of the repository.

    Repository Context:
    {context}

    Based on the above context, please answer the following question about the repository. Your answer should be specific and detailed, using exact information from the repository content.

    Question: {question}

    Detailed Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=google_api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

def repo_info_tool(query: str) -> str:
    """Answers questions about the repository details such as structure and contents etc"""
    if hasattr(st.session_state, 'repo_data') and hasattr(st.session_state, 'qa_chain'):
        repo_data = st.session_state.repo_data
        repo_info = f"Repository Name: {repo_data.name}\n"
        repo_info += f"Repository URL: {repo_data.url}\n"
        repo_info += f"File List:\n{', '.join(repo_data.get_file_list())}\n\n"
        
        response = st.session_state.qa_chain.run(repo_info + query)
        return response
    else:
        return "Please analyze a repository first."

def code_generator(prompt: str) -> str:
    """Generates code based on the given prompt and repository context."""
    if hasattr(st.session_state, 'repo_data'):
        repo_data = st.session_state.repo_data
        context = f"Repository: {repo_data.name}\nFiles: {', '.join(repo_data.get_file_list())}\n\n"
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        return llm.predict(f"{context}Generate code for: {prompt}")
    else:
        return "Please analyze a repository first."

def concept_explainer(concept: str) -> str:
    """Explains a given concept or tech stack in the context of the repository."""
    if hasattr(st.session_state, 'repo_data'):
        repo_data = st.session_state.repo_data
        context = f"In the context of the repository {repo_data.name} with files: {', '.join(repo_data.get_file_list())}\n\n"
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        return llm.predict(f"{context}Explain the following concept or tech stack in simple terms: {concept}")
    else:
        return "Please analyze a repository first."

tools = [
    Tool(
        name="RepoInfoTool",
        func=repo_info_tool,
        description="Useful for answering all the questions about repository structure, contents, and other repository-specific information."
    ),
    Tool(
        name="CodeGenerator",
        func=code_generator,
        description="Useful for generating code based on a given prompt and the repository as the context."
    ),
    Tool(
        name="ConceptExplainer",
        func=concept_explainer,
        description="Useful for explaining concepts or tech stacks in simple terms."
    )
]

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        
        # Add repository context
        if hasattr(st.session_state, 'repo_data'):
            repo_data = st.session_state.repo_data
            kwargs["repo_context"] = f"Repository: {repo_data.name}\nURL: {repo_data.url}\nFiles: {', '.join(repo_data.get_file_list())}\n\n"
        else:
            kwargs["repo_context"] = "No repository analyzed yet.\n\n"
        return self.template.format(**kwargs)
prompt = CustomPromptTemplate(
template="""You are an AI assistant specialized in analyzing and answering questions about GitHub repositories. Your primary task is to provide information about the repository that has been analyzed, including its structure, file list, and contents. If a question is not related to the repository or its contents, politely decline to answer.

Repository Context:
{repo_context}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (a question or request for the tool)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When using the RepoInfoTool, always provide a clear question or request as the Action Input. This tool can provide information about the repository structure, file list, and contents.

Begin!

Question: {input}
{agent_scratchpad}""",
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": "I couldn't understand how to proceed. Could you please rephrase your question?"},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def main():
    global prompt
    

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <style>
                .stSidebar {
                    background-color: #1e1e1e;
                    border-right: 1px solid #333;
                }
                .stSidebar h1, .stSidebar h2 {
                    margin-bottom: 1rem;
                    padding-top: 1rem;
                    border-top: 1px solid #ddd;
                    color: #ffffff;
                }
                .stSidebar p {
                    font-size: 0.9rem;
                    line-height: 1.5;
                    color: #cccccc;
                }
                .stSidebar a {
                    color: #66d9ef;
                    text-decoration: none;
                }
                .stSidebar a:hover {
                    color: #ffffff;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.title("(ℹ️) About")
        st.write("This app analyzes GitHub repositories and answers questions about them using AI.")
        
        st.title("📚 Instructions")
        st.write("1. Enter a GitHub repository URL in the input box.")
        st.write("2. Wait for the repository to be analyzed.")
        st.write("3. Use the chat interface to ask questions about the repository.")

        st.title("🛠️ Tools")
        st.write("• RepoInfoTool: Provides information about repository structure and contents")
        st.write("• CodeGenerator: Generates code based on prompts and repository context")
        st.write("• ConceptExplainer: Explains concepts and tech stacks in simple terms")


    st.title("GitHub Repository Analyzer")

    # Get repository URL from user
    repo_url = st.text_input("Enter a GitHub repository URL:", key="repo_url_input")

    if repo_url:
        # Extract repository name from URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")

        # Get repository content
        try:
            all_files = get_repository_content(repo_url)
           
        
  
        except Exception as e:
            st.error(f"Error accessing the repository: {str(e)}")
            return

        with st.spinner("Analyzing repository..."):
            vector_store = create_or_load_vector_store(all_files, repo_name)

        # Create RepositoryData object
        repo_data = RepositoryData(repo_name, repo_url, all_files, vector_store)

        # Store RepositoryData in session state
        st.session_state.repo_data = repo_data

        # Set up QA chain
        qa_chain = setup_qa_chain(vector_store)

        # Store QA chain in session state
        st.session_state.qa_chain = qa_chain

        # Create the LLM
        st_callback = StreamlitCallbackHandler(st.empty())
        stream_handler = StreamingStdOutCallbackHandler()

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            google_api_key=google_api_key,
            callbacks=[stream_handler, st_callback]
        )

        # Create the agent
        agent = LLMSingleActionAgent(
            llm_chain=LLMChain(llm=llm, prompt=prompt),  # Now this should work
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in tools]
        )

        # Create the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, 
            tools=tools, 
            verbose=True
        )

        st.session_state.agent_executor = agent_executor

    # Chat interface
    if "agent_executor" in st.session_state:
        # Display chat history
        for message in st.session_state.messages:
            col1, col2 = st.columns([1, 1])  # Create two equal-width columns
            
            if message["role"] == "user":
                with col2:  # Right column for user messages
                    st.chat_message("user").markdown(message["content"])
            elif message["role"] == "assistant":
                with col1:  # Left column for AI messages
                    st.chat_message("assistant").markdown(message["content"])

        user_question = st.chat_input("Ask a question about the repository:", key="user_question_input")
        if user_question:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Force a rerun to display the user's message immediately
            st.rerun()

        # Process the AI response only if there's a new user message
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
            try:
                # Use the agent to answer the question
                user_question = st.session_state.messages[-1]["content"]
                response = st.session_state.agent_executor.run(user_question)
                
                # Process the response
                if "Final Answer:" in response:
                    full_response = response.split("Final Answer:")[-1].strip()
                else:
                    full_response = response
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_message = f"Error processing your question: {str(e)}"
                # Add error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_message})

            # Force another rerun to display the AI's response
            st.rerun()

if __name__ == "__main__":
    os.makedirs(VECTOR_STORE_BASE_DIR, exist_ok=True)
    main()
