import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from uuid import uuid4
import time
import json
from typing import Any
from serpapi import GoogleSearch

# Custom JSON encoder to handle Pinecone ScoredVector and sets
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, '__dict__'):
            return {key: self.default(value) for key, value in obj.__dict__.items()}
        return super().default(obj)

# Set up logging
st.set_page_config(page_title="ReAct Pipeline", layout="wide")
log = st.empty()

def log_info(message):
    log.info(message)
    time.sleep(0.1)  # Small delay to ensure logs are displayed in order

def log_success(message):
    log.success(message)
    time.sleep(0.1)

def log_error(message):
    log.error(message)
    time.sleep(0.1)

def log_data(title, data):
    try:
        json_data = json.dumps(data, indent=2, cls=CustomJSONEncoder)
        log.info(f"{title}:\n```json\n{json_data}\n```")
    except Exception as e:
        log_error(f"Error logging data: {str(e)}")
        log.info(f"{title}: Unable to serialize data")
    time.sleep(0.1)

# Initialize Pinecone
log_info("Initializing Pinecone...")
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

index_name = "document-embeddings"
if index_name not in pc.list_indexes().names():
    log_info(f"Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    log_success(f"Pinecone index '{index_name}' created successfully")
else:
    log_info(f"Using existing Pinecone index: {index_name}")

index = pc.Index(index_name)

# Initialize OpenAI
log_info("Initializing OpenAI client...")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
log_success("OpenAI client initialized")

def split_text(text, chunk_size=1000):
    log_info(f"Splitting text into chunks (chunk size: {chunk_size})...")
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        if current_size + len(word) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for the space
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    log_success(f"Text split into {len(chunks)} chunks")
    log_data("First chunk", {"text": chunks[0]})
    return chunks

def get_embedding(text):
    log_info(f"Generating embedding for text: {text[:50]}...")
    start_time = time.time()
    embedding = client.embeddings.create(input=text, model="text-embedding-ada-002").data[0].embedding
    end_time = time.time()
    log_success(f"Embedding generated in {end_time - start_time:.2f} seconds")
    log_data("Embedding sample", {"embedding": embedding[:5]})  # Log first 5 values of embedding
    return embedding

def web_search(query, num_results=3):
    log_info(f"Performing web search for: {query}")
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": st.secrets["SERPAPI_API_KEY"],
            "num": num_results
        })
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        
        search_results = []
        for result in organic_results[:num_results]:
            search_results.append({
                'title': result.get('title', 'No title'),
                'link': result.get('link', 'No link'),
                'snippet': result.get('snippet', 'No snippet')
            })
        
        log_success(f"Web search completed. Found {len(search_results)} results.")
        log_data("Web search results", search_results)
        return search_results
    except Exception as e:
        log_error(f"Error during web search: {str(e)}")
        return []

# Streamlit UI
st.title("ReAct Pipeline: Document + Web Search")

# Upload document
uploaded_file = st.file_uploader("Upload a text document", type=["txt"])
if uploaded_file is not None:
    log_info(f"Document uploaded: {uploaded_file.name}")
    document_text = uploaded_file.read().decode("utf-8")
    st.write("**Document Content:**")
    st.write(document_text[:1000] + "...")  # Show only first 1000 characters

    # Split document into chunks
    chunks = split_text(document_text)
    
    # Generate embeddings and store in Pinecone
    log_info("Generating embeddings and storing in Pinecone...")
    start_time = time.time()
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        unique_id = f"{uploaded_file.name}-{i}-{uuid4()}"
        index.upsert(vectors=[(unique_id, embedding, {"text": chunk})])
        log_info(f"Chunk {i+1}/{len(chunks)} processed and stored")
        if i == 0:
            log_data("Sample upsert to Pinecone", {
                "id": unique_id,
                "metadata": {"text": chunk[:100] + "..."},  # Log first 100 characters of chunk
                "embedding_sample": embedding[:5]  # Log first 5 values of embedding
            })
    end_time = time.time()
    log_success(f"All chunks processed and stored in {end_time - start_time:.2f} seconds")

# Ask questions
question = st.text_input("Ask a question")
if question:
    log_info(f"Processing question: {question}")
    
    # 1. Retrieve from document
    log_info("Retrieving information from document...")
    query_embedding = get_embedding(question)
    doc_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    st.subheader("Document Retrieval:")
    doc_context = ""
    if doc_results['matches']:
        for i, match in enumerate(doc_results['matches']):
            st.write(f"Chunk {i+1} (Similarity: {match.score:.4f}):")
            st.text(match.metadata['text'])
            doc_context += match.metadata['text'] + "\n"
        log_success(f"Retrieved {len(doc_results['matches'])} relevant chunks from document")
        log_data("Document retrieval results", [
            {
                "score": match.score,
                "metadata": match.metadata,
                "id": match.id
            } for match in doc_results['matches']
        ])
    else:
        st.write("No relevant information found in the document.")
        log_info("No relevant information found in the document")
    
    # 2. Web search
    log_info("Performing web search...")
    web_results = web_search(question)
    st.subheader("Web Search Results:")
    web_context = ""
    for i, result in enumerate(web_results):
        st.write(f"Result {i+1}:")
        st.write(f"Title: {result['title']}")
        st.write(f"Link: {result['link']}")
        st.write(f"Snippet: {result['snippet']}")
        web_context += f"{result['title']}\n{result['snippet']}\n"
    log_success(f"Web search completed. Found {len(web_results)} results.")
    
    # 3. Combine and generate response
    log_info("Generating final response using GPT-4...")
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant that combines information from a document and web search to answer questions. Clearly distinguish between information from the document and from the web."},
            {"role": "user", "content": f"""
            Question: {question}
            
            Document context:
            {doc_context}
            
            Web search context:
            {web_context}
            
            Please provide a comprehensive answer, clearly indicating which parts of the information come from the document and which come from the web search. Then, provide a summary of your total learning from both sources.
            """}
        ],
        max_tokens=500
    )
    end_time = time.time()
    
    answer = response.choices[0].message.content.strip()
    st.subheader("Final Answer:")
    st.write(answer)
    log_success(f"Final response generated in {end_time - start_time:.2f} seconds")
    log_data("GPT-4 response", {"response": answer})

log_info("Processing complete")