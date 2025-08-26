import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
import time
from collections import defaultdict
import pandas as pd

# Basic setup
st.set_page_config(page_title="Multi-PDF Chat with Model Comparison")
st.title("Chat with Multiple PDFs (Model Comparison)")

# Initialize ALL session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.vector_stores = defaultdict(dict)
    st.session_state.uploaded_pdfs = []
    st.session_state.model_metrics = defaultdict(dict)
    st.session_state.current_model = "all-MiniLM-L6-v2"
    st.session_state.query_metrics = []

# Define embedding models to compare
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Faster but lower dimension (384d)"
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "description": "Slower but higher quality (768d)"
    }
}

# Load LLM
groq_api_key = "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"
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=groq_api_key, temperature=0.4)

# PDF upload and processing
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Load and split PDF
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                
                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(pages)
                
                # Process with both embedding models
                for model_name, model_info in EMBEDDING_MODELS.items():
                    start_time = time.time()
                    embeddings = HuggingFaceEmbeddings(model_name=model_info["name"])
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    processing_time = time.time() - start_time
                    
                    # Store metrics
                    st.session_state.model_metrics[uploaded_file.name][model_name] = {
                        "processing_time": processing_time,
                        "num_chunks": len(chunks),
                        "embedding_dim": embeddings.client.get_sentence_embedding_dimension()
                    }
                    st.session_state.vector_stores[model_name][uploaded_file.name] = vector_store
                
                st.session_state.uploaded_pdfs.append(uploaded_file.name)
                st.success(f"Processed: {uploaded_file.name} with both models")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)

# Model selection
current_model = st.selectbox(
    "Select Embedding Model",
    list(EMBEDDING_MODELS.keys()),
    index=0,
    format_func=lambda x: f"{x} ({EMBEDDING_MODELS[x]['description']})"
)

# Show uploaded PDFs and metrics
if st.session_state.uploaded_pdfs:
    st.subheader("Uploaded PDFs")
    for pdf in st.session_state.uploaded_pdfs:
        with st.expander(f"{pdf}", expanded=False):
            cols = st.columns(2)
            for idx, model_name in enumerate(EMBEDDING_MODELS.keys()):
                if pdf in st.session_state.model_metrics and model_name in st.session_state.model_metrics[pdf]:
                    metrics = st.session_state.model_metrics[pdf][model_name]
                    cols[idx].metric(
                        label=model_name,
                        value=f"{metrics['processing_time']:.2f}s",
                        help=f"Dimensions: {metrics['embedding_dim']} | Chunks: {metrics['num_chunks']}"
                    )

# Chat interface
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

def get_relevant_chunks(query, model_name, k_per_pdf=2):
    """Get top chunks from each PDF for a specific model"""
    relevant_chunks = []
    scores = []
    for pdf_name, vector_store in st.session_state.vector_stores[model_name].items():
        docs_and_scores = vector_store.similarity_search_with_score(query, k=k_per_pdf)
        for doc, score in docs_and_scores:
            relevant_chunks.append(f"From {pdf_name}:\n{doc.page_content}")
            scores.append(score)
    return relevant_chunks, scores

if prompt := st.chat_input("Ask about your PDFs"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("Analyzing..."):
        try:
            if st.session_state.vector_stores:
                # Get relevant chunks
                start_time = time.time()
                relevant_chunks, similarity_scores = get_relevant_chunks(prompt, current_model)
                retrieval_time = time.time() - start_time
                
                if relevant_chunks:
                    context = "\n\n".join(relevant_chunks)
                    
                    full_prompt = f"""
                    You are an assistant analyzing resumes. Provide:
                    1. ATS score (0-100)
                    2. Key strengths
                    3. Areas for improvement
                    4. Job role recommendations
                    
                    Context:
                    {context}
                    
                    Question: {prompt}
                    Answer:"""
                    
                    # Get response
                    response = llm.invoke(full_prompt)
                    answer = response.content
                    
                    # Store query metrics
                    st.session_state.query_metrics.append({
                        "model": current_model,
                        "query": prompt,
                        "retrieval_time": retrieval_time,
                        "avg_similarity": sum(similarity_scores)/len(similarity_scores) if similarity_scores else 0
                    })
                    
                    # Add metrics to answer
                    answer += f"\n\n🔍 Performance Metrics ({current_model}):"
                    answer += f"\n- ⏱️ Retrieval Time: {retrieval_time:.2f}s"
                    if similarity_scores:
                        answer += f"\n- 📊 Avg Similarity: {sum(similarity_scores)/len(similarity_scores):.2f}"
                else:
                    answer = "No relevant information found in the uploaded PDFs."
            else:
                answer = "Please upload PDF files first."
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Model comparison analysis
if st.session_state.uploaded_pdfs and st.session_state.query_metrics:
    st.sidebar.header("Model Performance Analysis")
    
    # Create comparison dataframe
    query_df = pd.DataFrame(st.session_state.query_metrics)
    
    # Calculate averages
    comparison_data = []
    for model_name in EMBEDDING_MODELS:
        model_data = query_df[query_df["model"] == model_name]
        if not model_data.empty:
            comparison_data.append({
                "Model": model_name,
                "Avg Retrieval Time (s)": model_data["retrieval_time"].mean(),
                "Avg Similarity Score": model_data["avg_similarity"].mean()
            })
    
    if comparison_data:
        st.sidebar.write("### Performance Comparison")
        st.sidebar.dataframe(pd.DataFrame(comparison_data).set_index("Model"), 
                           use_container_width=True)
        
        # Recommendation
        if len(comparison_data) == 2:
            time_ratio = comparison_data[1]["Avg Retrieval Time (s)"] / comparison_data[0]["Avg Retrieval Time (s)"]
            similarity_diff = comparison_data[1]["Avg Similarity Score"] - comparison_data[0]["Avg Similarity Score"]
            
            st.sidebar.write("### Recommendation")
            st.sidebar.write(f"- MPNet is {time_ratio:.1f}x slower than MiniLM")
            st.sidebar.write(f"- MPNet has {similarity_diff:.2f} better similarity score")
            if similarity_diff > 0.1:
                st.sidebar.success("Recommend MPNet for better accuracy")
            else:
                st.sidebar.info("Recommend MiniLM for faster performance")