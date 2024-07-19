# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import os
# load_dotenv()

# ## load the GROQ And OpenAI API KEY 
# groq_api_key=os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# st.title("Gemma Model Document Q&A")

# llm=ChatGroq(groq_api_key=groq_api_key,
#              model_name="Llama3-8b-8192")

# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )

# def vector_embedding():

#     if "vectors" not in st.session_state:

#         st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#         st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings





# prompt1=st.text_input("Enter Your Question From Doduments")


# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# import time



# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")









# import streamlit as st
# import os
# import fitz  # PyMuPDF
# from io import BytesIO
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time

# # Define a custom Document class with metadata
# class Document:
#     def __init__(self, page_content, metadata=None):
#         self.page_content = page_content
#         self.metadata = metadata or {}  # Default to an empty dictionary if no metadata is provided

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# st.title("Gemma Model Document Q&A")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# prompt = ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}
# """
# )

# def load_pdf_from_bytes(pdf_bytes):
#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# def vector_embedding(uploaded_files):
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
#         # Load documents from uploaded files
#         texts = []
#         for file in uploaded_files:
#             if file.type == "application/pdf":
#                 pdf_bytes = BytesIO(file.read())
#                 text = load_pdf_from_bytes(pdf_bytes)
#                 texts.append(text)
        
#         # Convert plain text to Document objects
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = []
#         for text in texts[:20]:  # Limiting to the first 20 texts
#             chunks = st.session_state.text_splitter.split_text(text)
#             st.session_state.final_documents.extend([Document(page_content=chunk) for chunk in chunks])
        
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# # Sidebar file uploader
# uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# if st.sidebar.button("Process Documents"):
#     if uploaded_files:
#         vector_embedding(uploaded_files)
#         st.write("Vector Store DB Is Ready")
#     else:
#         st.warning("Please upload at least one PDF file.")

# prompt1 = st.text_input("Enter Your Question From Documents")

# if st.button("Get Answer"):
#     if prompt1 and "vectors" in st.session_state:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': prompt1})
#         st.write("Response time:", time.process_time() - start)
#         st.write(response['answer'])

#         # With a streamlit expander
#         with st.expander("Document Similarity Search"):
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc.page_content)  # Ensure `doc` has `page_content` attribute
#                 st.write("--------------------------------")
#     else:
#         st.warning("Please process documents first and enter a question.")


# import streamlit as st
# import os
# import fitz  # PyMuPDF
# from io import BytesIO
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time

# # Set up page configuration
# st.set_page_config(
#     page_title="SBA INFO SOLUTION",
#     page_icon="sba_info_solutions_logo.jpg",  # Path to your icon
#     layout="wide",  # Wide layout
# )

# # Add markdown for branding
# st.markdown('# :white[SBA INFO SOLUTION]', unsafe_allow_html=True)
# st.markdown('## :white[Search Engine]', unsafe_allow_html=True)

# # Sidebar content
# st.sidebar.image("sba_info_solutions_logo.jpg", width=200, use_column_width=False)
# st.sidebar.markdown("---")

# # Define a custom Document class with metadata
# class Document:
#     def __init__(self, page_content, metadata=None):
#         self.page_content = page_content
#         self.metadata = metadata or {}  # Default to an empty dictionary if no metadata is provided

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# st.title("Gemma Model Document Q&A")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# prompt = ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}
# """
# )

# def load_pdf_from_bytes(pdf_bytes):
#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# def vector_embedding(uploaded_files):
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
#         # Load documents from uploaded files
#         texts = []
#         for file in uploaded_files:
#             if file.type == "application/pdf":
#                 pdf_bytes = BytesIO(file.read())
#                 text = load_pdf_from_bytes(pdf_bytes)
#                 texts.append(text)
        
#         # Convert plain text to Document objects
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = []
#         for text in texts[:20]:  # Limiting to the first 20 texts
#             chunks = st.session_state.text_splitter.split_text(text)
#             st.session_state.final_documents.extend([Document(page_content=chunk) for chunk in chunks])
        
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# # Sidebar file uploader
# uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# if st.sidebar.button("Process Documents"):
#     if uploaded_files:
#         vector_embedding(uploaded_files)
#         st.write("Vector Store DB Is Ready")
#     else:
#         st.warning("Please upload at least one PDF file.")

# prompt1 = st.text_input("Enter Your Question From Documents")

# if st.button("Get Answer"):
#     if prompt1 and "vectors" in st.session_state:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': prompt1})
#         st.write("Response time:", time.process_time() - start)
#         st.write(response['answer'])

#         # With a streamlit expander
#         with st.expander("Document Similarity Search"):
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc.page_content)  # Ensure `doc` has `page_content` attribute
#                 st.write("--------------------------------")
#     else:
#         st.warning("Please process documents first and enter a question.")

import streamlit as st
import os
import fitz  # PyMuPDF
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Set up page configuration
st.set_page_config(
    page_title="SBA INFO SOLUTION",
    page_icon="sba_info_solutions_logo.jpg",  # Path to your icon
    layout="wide",  # Wide layout
)

# Add markdown for branding
st.markdown('# :white[SBA INFO SOLUTION]', unsafe_allow_html=True)
st.markdown('## :white[Search Engine]', unsafe_allow_html=True)

# Sidebar content
st.sidebar.image("sba_info_solutions_logo.jpg", width=200, use_column_width=False)
st.sidebar.markdown("---")

# Define a custom Document class with metadata
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}  # Default to an empty dictionary if no metadata is provided

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

def load_pdf_from_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load documents from uploaded files
        texts = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                pdf_bytes = BytesIO(file.read())
                text = load_pdf_from_bytes(pdf_bytes)
                texts.append(text)
        
        # Convert plain text to Document objects
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = []
        for text in texts[:20]:  # Limiting to the first 20 texts
            chunks = st.session_state.text_splitter.split_text(text)
            st.session_state.final_documents.extend([Document(page_content=chunk) for chunk in chunks])
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Sidebar file uploader
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        vector_embedding(uploaded_files)
        st.write("Now ready")
    else:
        st.warning("Please upload at least one PDF file.")

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Get Answer"):
    if prompt1 and "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)  # Ensure `doc` has `page_content` attribute
                st.write("--------------------------------")
    else:
        st.warning("Please process documents first and enter a question.")
