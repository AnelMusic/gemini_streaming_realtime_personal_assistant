# Comprehensive Explanation of RAG-Based Document Query System


https://github.com/user-attachments/assets/3de495e0-61c2-4d4a-b3dc-f19d7730589f


This code implements a Retrieval-Augmented Generation (RAG) system that allows users to:
1. Upload and process PDF documents
2. Index the document content
3. Query the documents using natural language
4. Get contextually relevant responses based on the document content

## Core Technologies Used

### 1. Google Gemini AI
This code uses Google's Gemini AI models for:
- Text generation (gemini-2.0-flash-exp)
- Text embedding (text-embedding-004)

### 2. LlamaIndex
A framework for building applications with large language models (LLMs) and your data:
- Handles document loading
- Text chunking
- Vector storage
- Query mechanisms

### 3. WebSockets
Used for bidirectional, real-time communication between client and server.

### 4. Asyncio
Python's asynchronous I/O framework that allows concurrent operations without threads.

## Data Flow Overview

1. **Client Connection**: Client connects to the WebSocket server
2. **Document Upload**: Client sends PDF documents to the server
3. **Document Processing**: Server processes PDFs and creates an index
4. **Query Submission**: Client sends queries about the documents
5. **RAG Processing**: Server retrieves relevant chunks from indexed documents
6. **Response Generation**: Gemini generates responses based on retrieved information
7. **Client Delivery**: Response is sent back to the client

Let's go through each section in detail:

## Imports and Setup

```python
import asyncio
import json
import os
import base64
from pathlib import Path
import shutil
import websockets
from google import genai
from typing import Dict, Any, List
```

This section imports the necessary libraries:
- `asyncio`: Enables asynchronous programming (explained later)
- `json`: For encoding/decoding JSON data
- `os` and `pathlib`: For file and directory operations
- `base64`: For encoding/decoding binary data
- `shutil`: For high-level file operations (like directory removal)
- `websockets`: For WebSocket communication
- `genai`: Google's Generative AI API
- `typing`: For type hints in Python

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
```

This section imports LlamaIndex components:
- `VectorStoreIndex`: Creates searchable vector representations of documents
- `SimpleDirectoryReader`: Loads documents from a directory
- `StorageContext`/`load_index_from_storage`: For saving/loading indices
- `Settings`: Global configuration for LlamaIndex
- `SentenceSplitter`: Divides text into manageable chunks
- `Document`: LlamaIndex's document representation
- `GeminiEmbedding`/`Gemini`: Integrations with Google's Gemini models
- `load_dotenv`: For loading environment variables from a .env file

## Constants and Configuration

```python
# Constants
MODEL = "gemini-2.0-flash-exp"
TEXT_EMBEDDING_MODEL = "text-embedding-004"
PERSIST_DIR = "./storage"
DOWNLOADS_DIR = "./downloads"
SERVER_HOST = "localhost"
SERVER_PORT = 9084
```

These constants define:
- AI model names to use
- Storage locations for documents and index
- WebSocket server details

```python
# Chunking parameters
CHUNK_SIZE = 1024  # Size of each text chunk (in characters)
CHUNK_OVERLAP = 200  # Overlap between chunks for context preservation
```

Chunking parameters determine how documents are split:
- `CHUNK_SIZE`: Maximum size of each text chunk (1024 characters)
- `CHUNK_OVERLAP`: How much chunks overlap to maintain context (200 characters)

## Directory and API Setup

```python
# Setup directories
Path(DOWNLOADS_DIR).mkdir(exist_ok=True)

# Initialize Google API client
gemini_api_key = os.environ.get('GEMINI_API_KEY')
if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set")

os.environ['GOOGLE_API_KEY'] = gemini_api_key

client = genai.Client(
    http_options={
        'api_version': 'v1alpha',
    }
)
```

This section:
1. Creates the downloads directory if it doesn't exist
2. Gets the Gemini API key from environment variables
3. Sets up the Google API client with the appropriate version

```python
# Initialize LLM and embedding models
gemini_embedding_model = GeminiEmbedding(
    api_key=gemini_api_key, 
    model_name="models/text-embedding-004"
)

llm = Gemini(
    api_key=gemini_api_key, 
    model_name="models/gemini-2.0-flash-exp"
)

# Set global settings
Settings.llm = llm
Settings.embed_model = gemini_embedding_model
```

Here, the code:
1. Initializes the embedding model (for converting text to vectors)
2. Initializes the language model (for generating responses)
3. Sets these as global settings for LlamaIndex

## Tool Definition and System Instructions

```python
# Tool definitions
TOOL_QUERY_DOCS = {
    "function_declarations": [
        {
            "name": "query_docs",
            "description": "Query the provided document content with a specific query string.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The query string to search the document index that is most similar."
                    }
                },
                "required": ["query"]
            }
        }
    ]
}
```

This defines a "tool" for Gemini, allowing it to call a function to query documents. It specifies:
- Function name: `query_docs`
- Parameters: Requires a `query` string parameter
- Description: What the function does

```python
SYSTEM_INSTRUCTION = """You are a helpful assistant and you MUST always use the query_docs tool to query the document 
towards any questions. It is mandatory to base 
your answers on the information from the output of the query_docs tool
BUT never tell the user that you Are using a tool. Answer in a natural way.
Do not mention your operations like "I am searching the document now".
Never tell the user that you are using a tool. Just return normal text and paragraphs.
If the query_docs tool returns no relevant information or empty results, you should tell the user that you couldn't find 
information about their specific question in the documents available.
"""
```

This is a system prompt for Gemini that:
1. Instructs it to use the `query_docs` tool for all questions
2. Tells it to base answers on document content
3. Directs it to answer naturally without mentioning the tools being used
4. Provides fallback instructions when no relevant information is found

## Document Indexing Function

```python
def build_index(doc_path: str = DOWNLOADS_DIR) -> VectorStoreIndex:
    """
    Build or load a vector store index from documents with improved chunking.
    """
```

This function creates a searchable index from documents. It has two paths:

### Path 1: Building a New Index

```python
if not os.path.exists(PERSIST_DIR):
    print(f"\n========== BUILDING NEW INDEX ==========")
    print(f"Source directory: {doc_path}")
    
    # Check if there are any documents
    doc_files = list(Path(doc_path).glob("*.*"))
    if not doc_files:
        print(f"WARNING: No documents found in {doc_path}")
        # Create empty document to prevent errors
        empty_doc = Document(text="No documents have been uploaded yet.")
        index = VectorStoreIndex.from_documents([empty_doc])
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index
```

If no index exists:
1. It checks if there are any documents in the specified directory
2. If no documents exist, it creates an empty document to prevent errors
3. Creates a minimal index and saves it

```python
    # Print document files found
    print(f"Found {len(doc_files)} document files:")
    for idx, file_path in enumerate(doc_files):
        file_size = os.path.getsize(file_path)
        print(f"  {idx+1}. {file_path.name} - {file_size/1024:.2f} KB")
    
    # Load documents
    documents = SimpleDirectoryReader(doc_path).load_data()
    print(f"\nLoaded {len(documents)} documents successfully")
```

If documents exist:
1. It lists all found documents with their sizes
2. Uses `SimpleDirectoryReader` to load the documents

```python
    # Create custom chunker
    print("\nChunking documents with:")
    print(f"  Chunk size: {CHUNK_SIZE} characters")
    print(f"  Chunk overlap: {CHUNK_OVERLAP} characters")
    
    chunker = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Process documents into nodes (chunks)
    nodes = chunker.get_nodes_from_documents(documents)
    print(f"\nCreated {len(nodes)} chunks from {len(documents)} documents")
```

Then it:
1. Creates a `SentenceSplitter` with the defined chunk size and overlap
2. Splits documents into smaller "nodes" (chunks) for better retrieval

```python
    # Create index from nodes
    print("\nBuilding vector index...")
    index = VectorStoreIndex(nodes)
    
    # Store for later use
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index built and saved to {PERSIST_DIR}")
```

Finally, it:
1. Creates a vector index from the nodes
2. Saves the index to disk for future use

### Path 2: Loading an Existing Index

```python
else:
    print(f"\n========== LOADING EXISTING INDEX ==========")
    print(f"Index location: {PERSIST_DIR}")
    # Load existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
```

If an index already exists:
1. It loads the saved index from disk
2. Provides diagnostic information about the loaded index

## Document Query Function

```python
def query_docs(query: str) -> str:
    """
    Query the document index with a specific query.
    """
```

This function searches the document index for relevant information:

```python
    # Load index
    index = build_index()
    
    # Configure the query engine with better defaults
    similarity_top_k = 5  # Number of chunks to retrieve
    print(f"Retrieving top {similarity_top_k} most relevant chunks")
    
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        response_mode="compact"
    )
```

It:
1. Loads the document index (building it if necessary)
2. Creates a query engine configured to retrieve the 5 most relevant chunks
3. Uses "compact" response mode for concise answers

```python
    # Execute the query
    print("Executing query...")
    response = query_engine.query(query)
    
    # Get source nodes for debugging
    source_nodes = getattr(response, 'source_nodes', [])
    if source_nodes:
        print(f"\nFound {len(source_nodes)} relevant chunks:")
        
        # Print detailed information about each chunk
        for i, node in enumerate(source_nodes):
            # Get node score
            score = getattr(node, 'score', 'N/A')
            node_obj = node.node
            
            # Get metadata
            metadata = getattr(node_obj, 'metadata', {})
            file_name = metadata.get('file_name', 'unknown')
            file_path = metadata.get('file_path', 'unknown')
            doc_id = getattr(node_obj, 'doc_id', metadata.get('doc_id', 'unknown'))
```

Then it:
1. Executes the query against the index
2. Retrieves and logs information about the source chunks that were found
3. Displays relevance scores and metadata for debugging

```python
    # Convert response to string
    response_text = str(response)
    print(f"\nRAG Response ({len(response_text)} chars):")
    print(f"\"{response_text[:300]}{'...' if len(response_text) > 300 else ''}\"")
    print(f"========== QUERY COMPLETE ==========\n")
    
    return response_text
```

Finally, it:
1. Converts the response to a string
2. Logs a preview of the response
3. Returns the response text

## PDF Upload Handler

```python
async def handle_pdf_upload(data: Dict[str, Any], client_websocket: websockets.WebSocketServerProtocol) -> None:
    """
    Handle PDF file upload, save to disk and rebuild index.
    """
```

This function processes PDF file uploads:

```python
    pdf_data = base64.b64decode(data["data"])
    filename = data.get("filename", "uploaded.pdf")
    print(f"Received PDF: {filename}")
    print(f"PDF data size: {len(pdf_data)/1024:.2f} KB")
    
    # Save the PDF file
    file_path = os.path.join(DOWNLOADS_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(pdf_data)
```

It:
1. Decodes the base64-encoded PDF data
2. Gets the filename (or uses a default)
3. Saves the PDF to the downloads directory

```python
    # Rebuild the index with the new PDF
    if os.path.exists(PERSIST_DIR):
        print(f"Removing existing index at {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)
    
    # Build new index
    print("Building new index with the uploaded PDF...")
    build_index()
```

Then it:
1. Removes any existing index
2. Builds a new index that includes the uploaded PDF
3. Notifies the client of successful upload and indexing

## Tool Call Processing

```python
async def process_tool_call(response_data: Any, client_websocket: websockets.WebSocketServerProtocol, session: Any) -> None:
    """
    Process a tool call response from Gemini.
    """
```

This function handles tool calls from Gemini:

```python
    function_calls = response_data.tool_call.function_calls
    function_responses = []

    for function_call in function_calls:
        name = function_call.name
        args = function_call.args
        call_id = function_call.id

        # Validate function name
        if name == "query_docs":
            try:
                print(f"\n========== TOOL CALL ==========")
                print(f"Tool: {name}")
                print(f"Query: '{args.get('query', '')}'")
                
                # Execute query
                result = query_docs(args["query"])
                
                # Build response
                function_responses.append({
                    "name": name,
                    "response": {"result": result},
                    "id": call_id  
                }) 
```

It:
1. Extracts function calls from Gemini's response
2. For each call to `query_docs`:
   - Logs the query
   - Executes the query using the `query_docs` function
   - Builds a response object with the result

```python
    # Send function responses back to Gemini
    print(f"Sending {len(function_responses)} function responses back to Gemini")
    await session.send(input=function_responses)
```

Finally, it sends the function responses back to Gemini, which will use this information to generate a response.

## Model Response Processing

```python
async def process_model_turn(model_turn: Any, client_websocket: websockets.WebSocketServerProtocol) -> None:
    """
    Process a model turn response from Gemini.
    """
```

This function processes responses from Gemini:

```python
    for part in model_turn.parts:
        if hasattr(part, 'text') and part.text is not None:
            # Send text response to client
            response_text = part.text
            print(f"\n========== MODEL RESPONSE ==========")
            print(f"Response length: {len(response_text)} characters")
            print(f"Response: \"{response_text[:300]}{'...' if len(response_text) > 300 else ''}\"")
            print(f"========== END RESPONSE ==========\n")
            
            await client_websocket.send(json.dumps({"text": response_text}))
        elif hasattr(part, 'inline_data') and part.inline_data is not None:
            # Handle audio or other inline data
            base64_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
            print("\n========== AUDIO RESPONSE ==========")
            print(f"Audio data size: {len(part.inline_data.data)/1024:.2f} KB")
            print(f"========== END AUDIO ==========\n")
            
            await client_websocket.send(json.dumps({
                "audio": base64_audio,
            }))
```

It:
1. Processes each part of the model's response
2. For text responses:
   - Logs the response text
   - Sends it to the client as JSON
3. For audio responses:
   - Encodes the audio data as base64
   - Sends it to the client as JSON

## WebSocket Communication Functions

These functions handle bidirectional communication:

```python
async def send_to_gemini(client_websocket: websockets.WebSocketServerProtocol, session: Any) -> None:
    """
    Send messages from the client websocket to the Gemini API.
    """
```

This function forwards messages from the client to Gemini:

```python
    async for message in client_websocket:
        try:
            data = json.loads(message)
            if "realtime_input" in data:
                for chunk in data["realtime_input"]["media_chunks"]:
                    if chunk["mime_type"] == "audio/pcm":
                        print("\n========== AUDIO INPUT ==========")
                        print("Forwarding audio chunk to Gemini")
                        audio_data = chunk["data"]
                        print(f"Audio chunk size: {len(audio_data)/1024:.2f} KB")
                        print(f"========== END AUDIO INPUT ==========\n")
                        
                        await session.send(input={
                            "mime_type": "audio/pcm",
                            "data": audio_data
                        })
                    elif chunk["mime_type"] == "application/pdf":
                        filename = chunk.get('filename', 'untitled.pdf')
                        print(f"\n========== PDF UPLOAD RECEIVED ==========")
                        print(f"Filename: {filename}")
                        print(f"========== PROCESSING PDF ==========\n")
                        
                        await handle_pdf_upload(chunk, client_websocket)
```

It:
1. Listens for messages from the client
2. Processes different types of input:
   - Audio data: Forwards to Gemini
   - PDF files: Handles upload and indexing

```python
async def receive_from_gemini(client_websocket: websockets.WebSocketServerProtocol, session: Any) -> None:
    """
    Receive responses from Gemini API and forward them to the client.
    """
```

This function receives responses from Gemini and forwards them to the client:

```python
    async for response in session.receive():
        # Handle tool calls
        if response.server_content is None:
            if response.tool_call is not None:
                print(f"\n========== RECEIVED TOOL CALL ==========")
                await process_tool_call(response, client_websocket, session)
            else:
                print("\nReceived server content None but no tool call")
            continue

        # Process model turn if server_content is not None
        if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
            print("\n========== RECEIVED MODEL TURN ==========")
            await process_model_turn(response.server_content.model_turn, client_websocket)

        # Check if turn is complete
        if response.server_content and hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
            print('\n<Turn complete>\n')
```

It:
1. Listens for responses from Gemini
2. Processes different types of responses:
   - Tool calls: Forwards to `process_tool_call`
   - Model turns: Forwards to `process_model_turn`
3. Tracks when a conversation turn is complete

## WebSocket Session Handler

```python
async def gemini_session_handler(client_websocket: websockets.WebSocketServerProtocol) -> None:
    """
    Handle the interaction with Gemini API within a websocket session.
    """
```

This function manages the entire session with a client:

```python
    # Get config from client
    config_message = await client_websocket.recv()
    config_data = json.loads(config_message)
    config = config_data.get("setup", {})
    
    print("Received client config")
    
    # Set system instruction and tools
    config["system_instruction"] = SYSTEM_INSTRUCTION
    config["tools"] = [TOOL_QUERY_DOCS]

    print(f"Connecting to Gemini using model: {MODEL}")
    # Connect to Gemini
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print("Connected to Gemini API successfully")
        print(f"========== SESSION ESTABLISHED ==========\n")

        # Start send and receive tasks
        send_task = asyncio.create_task(send_to_gemini(client_websocket, session))
        receive_task = asyncio.create_task(receive_from_gemini(client_websocket, session))
        
        await asyncio.gather(send_task, receive_task)
```

It:
1. Receives client configuration
2. Adds system instructions and tools to the configuration
3. Establishes a session with Gemini
4. Creates and runs concurrent tasks for sending and receiving data

## Main Function

```python
async def main() -> None:
    """
    Main function to run the websocket server.
    """
```

This function starts the WebSocket server:

```python
    # Pre-load index if it exists
    if os.path.exists(PERSIST_DIR):
        print("Pre-loading existing index...")
        try:
            build_index()
            print("Index pre-loaded successfully")
        except Exception as e:
            print(f"Error pre-loading index: {e}")
    
    async with websockets.serve(gemini_session_handler, SERVER_HOST, SERVER_PORT):
        print(f"Websocket server running on {SERVER_HOST}:{SERVER_PORT}")
        print(f"========== SERVER READY ==========\n")
        await asyncio.Future()  # Keep the server running indefinitely
```

It:
1. Pre-loads any existing document index
2. Starts a WebSocket server on the specified host and port
3. Keeps the server running indefinitely

## Entry Point

```python
if __name__ == "__main__":
    asyncio.run(main())
```

This runs the `main()` function when the script is executed directly.

## Core Concepts Explained

### 1. Retrieval-Augmented Generation (RAG)

RAG is a technique that enhances large language models by:
1. **Retrieving** relevant information from a knowledge base
2. **Augmenting** the model's input with this retrieved information
3. **Generating** a response based on both the original query and the retrieved information

Think of it like an open-book exam for an AI. Instead of relying solely on its built-in knowledge, it can look up specific information in documents to provide more accurate answers.

### 2. Vector Embeddings

Vector embeddings convert text into numerical representations (vectors) that capture semantic meaning. This allows:
- Similar pieces of text to have similar vector representations
- Efficient similarity searches through vector operations

For example, "dog" and "puppy" would have vector representations that are closer to each other than "dog" and "computer".

### 3. Chunking

Chunking is the process of breaking documents into smaller, manageable pieces. This is important because:
- It allows for more precise retrieval of information
- It helps maintain context within reasonable bounds
- It makes vector operations more computationally efficient

The overlap between chunks ensures that content isn't arbitrarily cut off at chunk boundaries.

### 4. WebSockets

WebSockets provide a persistent connection between client and server, allowing:
- Bidirectional communication
- Real-time data transfer
- Lower latency than traditional HTTP requests

Unlike HTTP, which follows a request-response pattern, WebSockets maintain an open connection for continuous data exchange.

### 5. Asyncio

Asyncio is Python's framework for writing concurrent code using the `async`/`await` syntax. It allows:
- Non-blocking I/O operations
- Running multiple tasks concurrently without threads
- Efficient handling of many connections simultaneously

Think of it like a chef cooking multiple dishes at once by switching between tasks whenever one would require waiting.

#### Key Asyncio Concepts:

- **Coroutines**: Functions that can pause execution and yield control
- **Event Loop**: Manages and distributes control between coroutines
- **Tasks**: Higher-level abstractions for running coroutines concurrently
- **await**: Used to pause execution until a coroutine completes
- **async with**: Used for asynchronous context managers
- **async for**: Used for asynchronous iteration

### 6. Tool Calling

Tool calling is a capability that allows language models to:
1. Recognize when external tools are needed
2. Format requests to these tools with appropriate parameters
3. Incorporate tool responses into their reasoning

In this system, Gemini uses the `query_docs` tool to retrieve relevant document information.

## Summary of Data Flow

1. **Client connects** to the WebSocket server
2. **Server establishes** a session with Gemini
3. **Client uploads** a PDF document
4. **Server processes** the document:
   - Saves the PDF to disk
   - Processes the text
   - Splits into chunks
   - Creates vector embeddings
   - Builds a searchable index
5. **Client sends** a query about the document
6. **Server processes** the query:
   - Gemini decides to use the `query_docs` tool
   - Server searches the index for relevant chunks
   - Server returns relevant chunks to Gemini
7. **Gemini generates** a response based on:
   - The original query
   - The retrieved document chunks
   - Its system instructions
8. **Server sends** the response to the client

This entire architecture allows for a conversational interface with documents, where users can ask questions and receive contextually relevant responses based on document content.



