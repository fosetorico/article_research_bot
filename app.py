import os
import streamlit as st
import pickle
import faiss
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)


st.title("Article/News Research Tool")
st.sidebar.title("Article URLs...")

# Initialize session state for Q&A history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Ask the user how many URLs they want to input
num_urls = st.sidebar.number_input("How many URLs do you want to process?", min_value=1, max_value=10, value=3)

urls = []
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

process_url_clicked = st.sidebar.button("Process Article URLs")
# file_path = "faiss_store_openai.pkl"
#
main_placeholder = st.empty()
llm = OpenAI(temperature=0.5, max_tokens=500)

index_path = "faiss_index.bin"
docs_path = "docs.pkl"
index_to_docstore_id_path = "index_to_docstore_id.pkl"

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Initiated...")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        # separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        # chunk_overlap=200
    )

    main_placeholder.text("Text Splitter...Initiated...")
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    embedding_dimension = 1536
    docstore_dict = {str(i): doc for i, doc in enumerate(docs)}
    docstore = InMemoryDocstore(docstore_dict)

    # Create FAISS vector index
    index = faiss.IndexFlatL2(embedding_dimension)

    # Initialize the FAISS vector store with a correct mapping
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}
    vector_store = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

    # Add documents to the FAISS index
    vector_store.add_documents(docs)
    main_placeholder.text("Embedding Vector Building Initiated...")

    # Save the FAISS index and documents separately
    # index_path = "faiss_index.bin"
    faiss.write_index(vector_store.index, index_path)
    # docs_path = "docs.pkl"
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)

    # Save the index_to_docstore_id mapping
    # index_to_docstore_id_path = "index_to_docstore_id.pkl"
    with open(index_to_docstore_id_path, "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)


query = main_placeholder.text_input("Question: ")
if query:
    # Load the FAISS index and documents
    if os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(index_to_docstore_id_path):
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        with open(index_to_docstore_id_path, "rb") as f:
            index_to_docstore_id = pickle.load(f)
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
        # print(f"Loaded document store keys: {list(docstore._dict.keys())[:10]}")  # Debug output
        embeddings = OpenAIEmbeddings()  # Recreate embeddings object
        vector_store = FAISS(embedding_function=embeddings, index=index, docstore=docstore,
                             index_to_docstore_id=index_to_docstore_id)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
        result = chain.invoke({"question": query}, return_only_outputs=True)

        # Extract and display the result
        answer = result.get("answer", "No answer found.")
        sources = result.get("sources", "No sources available.")
        
        # Add to session state history
        st.session_state.qa_history.append({"question": query, "answer": answer, "sources": sources})

        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.subheader("Response:")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)

# Display all questions and answers from the session
if st.session_state.qa_history:
    st.write("---------------------------------------------------------------------")
    st.subheader("History:")
    for entry in st.session_state.qa_history:
        st.write(f"**Q:** {entry['question']}")
        st.write(f"**A:** {entry['answer']}")
        st.write(f"**Sources:** {entry['sources']}")





