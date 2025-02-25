import asyncio
import json
import os
import base64
from pathlib import Path
import shutil
import websockets
from google import genai
from typing import Dict, Any, List

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

# Load environment variables
load_dotenv(override=True)

# Constants
MODEL = "gemini-2.0-flash-exp"
TEXT_EMBEDDING_MODEL = "text-embedding-004"
PERSIST_DIR = "./storage"
DOWNLOADS_DIR = "./downloads"
SERVER_HOST = "localhost"
SERVER_PORT = 9084

# Chunking parameters
CHUNK_SIZE = 512  # Size of each text chunk (in characters)
CHUNK_OVERLAP = 200  # Overlap between chunks for context preservation

# Track speech state
audio_from_client_active = False
audio_to_client_active = False

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

SYSTEM_INSTRUCTION = """You are a helpful assistant and you MUST always use the query_docs tool to query the document 
towards any questions. It is mandatory to base 
your answers on the information from the output of the query_docs tool
BUT never tell the user that you Are using a tool. Answer in a natural way.
Do not mention your operations like "I am searching the document now".
Never tell the user that you are using a tool. Just return normal text and paragraphs.
If the query_docs tool returns no relevant information or empty results, you should tell the user that you couldn't find 
information about their specific question in the documents available.
"""


def build_index(doc_path: str = DOWNLOADS_DIR) -> VectorStoreIndex:
    """
    Build or load a vector store index from documents with improved chunking.
    
    Args:
        doc_path: Directory containing documents to index
        
    Returns:
        The vector store index
    """
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
            
        # Print document files found
        print(f"Found {len(doc_files)} document files:")
        for idx, file_path in enumerate(doc_files):
            file_size = os.path.getsize(file_path)
            print(f"  {idx+1}. {file_path.name} - {file_size/1024:.2f} KB")
        
        # Load documents
        documents = SimpleDirectoryReader(doc_path).load_data()
        print(f"\nLoaded {len(documents)} documents successfully")
        
        # Print document details
        for idx, doc in enumerate(documents):
            filename = doc.metadata.get('file_name', 'unknown')
            file_type = doc.metadata.get('file_type', 'unknown')
            print(f"\nDocument {idx+1}: {filename} ({file_type})")
            print(f"  Length: {len(doc.text)} characters")
            print(f"  Metadata: {doc.metadata}")
            print(f"  First 150 chars: {doc.text[:150]}...")
        
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
        
        # Print chunk details
        print("\nSample chunks:")
        samples = min(5, len(nodes))
        for i in range(samples):
            node = nodes[i]
            doc_id = node.metadata.get("doc_id", "unknown")
            filename = node.metadata.get("file_name", "unknown")
            print(f"\nChunk {i+1}/{samples} (from {filename}, doc_id: {doc_id}):")
            print(f"  Length: {len(node.text)} characters")
            print(f"  Content: {node.text[:150]}...")
            
        # Create index from nodes
        print("\nBuilding vector index...")
        index = VectorStoreIndex(nodes)
        
        # Store for later use
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"Index built and saved to {PERSIST_DIR}")
        print(f"========== INDEX BUILDING COMPLETE ==========\n")
    else:
        print(f"\n========== LOADING EXISTING INDEX ==========")
        print(f"Index location: {PERSIST_DIR}")
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        
        # Print some diagnostics
        if hasattr(index, 'docstore') and hasattr(index.docstore, 'docs'):
            print(f"Loaded index with {len(index.docstore.docs)} chunks")
            
            # Print some sample chunks from the loaded index
            print("\nSample chunks from loaded index:")
            docs = list(index.docstore.docs.values())
            samples = min(3, len(docs))
            for i in range(samples):
                doc = docs[i]
                if hasattr(doc, 'text'):
                    doc_id = getattr(doc, 'doc_id', "unknown")
                    print(f"\nChunk {i+1}/{samples} (doc_id: {doc_id}):")
                    print(f"  Length: {len(doc.text)} characters")
                    print(f"  Content: {doc.text[:150]}...")
        
        print(f"========== INDEX LOADING COMPLETE ==========\n")
        
    return index


def query_docs(query: str) -> str:
    """
    Query the document index with a specific query.
    
    Args:
        query: The query string to search for
        
    Returns:
        Response text from the query engine
    """
    print(f"\n========== QUERY EXECUTION ==========")
    print(f"Query: '{query}'")
    
    # Load index
    index = build_index()
    
    # Configure the query engine with better defaults
    similarity_top_k = 5  # Number of chunks to retrieve
    print(f"Retrieving top {similarity_top_k} most relevant chunks")
    
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        response_mode="compact"
    )
    
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
            
            # Print node information with clear formatting
            print(f"\n----- Chunk {i+1}/{len(source_nodes)} -----")
            print(f"Score: {score:.4f}" if isinstance(score, float) else f"Score: {score}")
            print(f"Document: {file_name}")
            print(f"Path: {file_path}")
            print(f"Doc ID: {doc_id}")
            
            # Print text sample with character count
            text = node_obj.text
            text_len = len(text)
            print(f"Text length: {text_len} characters")
            
            # Show more text for higher-ranked results
            preview_len = min(300 if i < 2 else 150, text_len)
            preview_text = text[:preview_len]
            print(f"Text preview: \"{preview_text}{'...' if text_len > preview_len else ''}\"")
    else:
        print("\nWARNING: No relevant chunks found for query")
    
    # Convert response to string
    response_text = str(response)
    print(f"\nRAG Response ({len(response_text)} chars):")
    print(f"\"{response_text[:300]}{'...' if len(response_text) > 300 else ''}\"")
    print(f"========== QUERY COMPLETE ==========\n")
    
    return response_text


async def handle_pdf_upload(data: Dict[str, Any], client_websocket: websockets.WebSocketServerProtocol) -> None:
    """
    Handle PDF file upload, save to disk and rebuild index.
    
    Args:
        data: The PDF chunk data
        client_websocket: The websocket connection to the client
    """
    try:
        print(f"\n========== PDF UPLOAD ==========")
        pdf_data = base64.b64decode(data["data"])
        filename = data.get("filename", "uploaded.pdf")
        print(f"Received PDF: {filename}")
        print(f"PDF data size: {len(pdf_data)/1024:.2f} KB")
        
        # Save the PDF file
        file_path = os.path.join(DOWNLOADS_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(pdf_data)
        
        print(f"Saved PDF file to {file_path}")
        
        # Check if PDF was saved correctly
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"Saved PDF file size: {file_size/1024:.2f} KB")
            if file_size == 0:
                print("WARNING: Saved PDF file is empty")
                await client_websocket.send(json.dumps({
                    "text": f"Warning: The uploaded PDF file appears to be empty."
                }))
        
        # Rebuild the index with the new PDF
        if os.path.exists(PERSIST_DIR):
            print(f"Removing existing index at {PERSIST_DIR}")
            shutil.rmtree(PERSIST_DIR)
        
        # Build new index
        print("Building new index with the uploaded PDF...")
        build_index()
        
        print(f"========== PDF UPLOAD COMPLETE ==========\n")
        await client_websocket.send(json.dumps({
            "text": f"PDF file '{filename}' has been uploaded and indexed successfully."
        }))
    except Exception as e:
        error_msg = f"Error handling PDF upload: {str(e)}"
        print(error_msg)
        await client_websocket.send(json.dumps({
            "text": f"Error processing PDF: {str(e)}"
        }))


async def process_tool_call(response_data: Any, client_websocket: websockets.WebSocketServerProtocol, session: Any) -> None:
    """
    Process a tool call response from Gemini.
    
    Args:
        response_data: The response containing tool calls
        client_websocket: The websocket connection to the client
        session: The Gemini session
    """
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
                
                # Let the user know we're processing their query
                await client_websocket.send(json.dumps({"text": f"Searching for: {args['query']}..."}))
                print(f"Function executed - response length: {len(result)} characters")
                print(f"========== TOOL CALL COMPLETE ==========\n")
            except Exception as e:
                error_msg = f"Error executing function: {str(e)}"
                print(f"\nERROR: {error_msg}")
                import traceback
                traceback.print_exc()
                
                # Send error information to help debugging
                function_responses.append({
                    "name": name,
                    "response": {"result": f"Error processing query: {str(e)}"},
                    "id": call_id
                })
                print(f"========== TOOL CALL ERROR ==========\n")
                continue

    # Send function responses back to Gemini
    print(f"Sending {len(function_responses)} function responses back to Gemini")
    await session.send(input=function_responses)


async def process_model_turn(model_turn: Any, client_websocket: websockets.WebSocketServerProtocol) -> None:
    """
    Process a model turn response from Gemini.
    
    Args:
        model_turn: The model turn data
        client_websocket: The websocket connection to the client
    """
    for part in model_turn.parts:
        if hasattr(part, 'text') and part.text is not None:
            # Send text response to client
            response_text = part.text
            print(f"Response: \"{response_text[:300]}{'...' if len(response_text) > 300 else ''}\"")
            
            await client_websocket.send(json.dumps({"text": response_text}))
        elif hasattr(part, 'inline_data') and part.inline_data is not None:
            # Handle audio or other inline data - don't log anything here
            base64_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
            
            await client_websocket.send(json.dumps({
                "audio": base64_audio,
            }))


async def send_to_gemini(client_websocket: websockets.WebSocketServerProtocol, session: Any) -> None:
    """
    Send messages from the client websocket to the Gemini API.
    
    Args:
        client_websocket: The websocket connection to the client
        session: The Gemini session
    """
    global audio_from_client_active
    
    try:
        print("Starting message forwarding from client to Gemini")
        async for message in client_websocket:
            try:
                data = json.loads(message)
                if "realtime_input" in data:
                    # Check if there are any audio chunks in this message
                    has_audio = any(chunk["mime_type"] == "audio/pcm" 
                                   for chunk in data["realtime_input"]["media_chunks"])
                    
                    # Print only when audio starts (transition from no audio to having audio)
                    if has_audio and not audio_from_client_active:
                        print("\n========== AUDIO FROM CLIENT STARTED ==========")
                        audio_from_client_active = True
                    
                    # Process all chunks
                    for chunk in data["realtime_input"]["media_chunks"]:
                        if chunk["mime_type"] == "audio/pcm":
                            audio_data = chunk["data"]
                            await session.send(input={
                                "mime_type": "audio/pcm",
                                "data": audio_data
                            })
                        elif chunk["mime_type"] == "application/pdf":
                            filename = chunk.get('filename', 'untitled.pdf')
                            print(f"\n========== PDF UPLOAD RECEIVED ==========")
                            print(f"Filename: {filename}")
                            print(f"========== PROCESSING PDF ==========\n")
                            
                            # Reset audio flag since we're processing a PDF
                            audio_from_client_active = False
                            
                            await handle_pdf_upload(chunk, client_websocket)
                
                # If we received a non-audio message, mark audio as inactive
                elif audio_from_client_active:
                    print("\n========== AUDIO FROM CLIENT ENDED ==========")
                    audio_from_client_active = False
            except Exception as e:
                print(f"Error sending to Gemini: {e}")
                import traceback
                traceback.print_exc()
        print("Client connection closed (send)")
    except Exception as e:
        print(f"Error sending to Gemini: {e}")
    finally:
        print("send_to_gemini task closed")


async def receive_from_gemini(client_websocket: websockets.WebSocketServerProtocol, session: Any) -> None:
    """
    Receive responses from Gemini API and forward them to the client.
    
    Args:
        client_websocket: The websocket connection to the client
        session: The Gemini session
    """
    global audio_to_client_active
    
    try:
        print("Starting response handling from Gemini to client")
        while True:
            try:
                print("Awaiting response from Gemini")
                audio_detected_in_turn = False
                
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
                        # Only print "RECEIVED MODEL TURN" if we have meaningful content and not just audio continuation
                        has_text = any(hasattr(part, 'text') and part.text 
                                      for part in response.server_content.model_turn.parts)
                        
                        if has_text:
                            print("\n========== RECEIVED MODEL TURN ==========")
                                
                        # Check for audio in this turn
                        has_audio = any(hasattr(part, 'inline_data') and part.inline_data is not None 
                                       for part in response.server_content.model_turn.parts)
                                
                        # Mark if we detected audio in this entire response stream
                        if has_audio:
                            audio_detected_in_turn = True
                            
                            # Print only when audio starts
                            if not audio_to_client_active:
                                print("\n========== AUDIO TO CLIENT STARTED ==========")
                                audio_to_client_active = True
                                
                        await process_model_turn(response.server_content.model_turn, client_websocket)

                    # Check if turn is complete
                    if response.server_content and hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                        print('\n<Turn complete>\n')
                        
                        # If we've completed a turn and had no audio, reset the audio flag
                        if not audio_detected_in_turn and audio_to_client_active:
                            print("\n========== AUDIO TO CLIENT ENDED ==========")
                            audio_to_client_active = False
                            
            except websockets.exceptions.ConnectionClosedOK:
                print("Client connection closed normally (receive)")
                break
            except Exception as e:
                print(f"Error receiving from Gemini: {e}")
                import traceback
                traceback.print_exc()
                break

    except Exception as e:
        print(f"Error receiving from Gemini: {e}")
    finally:
        print("Gemini connection closed (receive)")


async def gemini_session_handler(client_websocket: websockets.WebSocketServerProtocol) -> None:
    """
    Handle the interaction with Gemini API within a websocket session.
    
    Args:
        client_websocket: The websocket connection to the client
    """
    try:
        print("\n========== NEW CLIENT CONNECTION ==========")
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

    except Exception as e:
        print(f"\n========== SESSION ERROR ==========")
        print(f"Error in Gemini session: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"========== END ERROR ==========\n")
        
        # Try to send error message to client
        try:
            await client_websocket.send(json.dumps({
                "text": f"Error connecting to Gemini: {str(e)}"
            }))
        except:
            pass
    finally:
        print("\n========== SESSION CLOSED ==========\n")


async def main() -> None:
    """
    Main function to run the websocket server.
    """
    print(f"\n========== SERVER STARTUP ==========")
    print(f"Host: {SERVER_HOST}")
    print(f"Port: {SERVER_PORT}")
    print(f"Document storage directory: {DOWNLOADS_DIR}")
    print(f"Index storage directory: {PERSIST_DIR}")
    print(f"Chunk size: {CHUNK_SIZE}, Chunk overlap: {CHUNK_OVERLAP}")
    
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


if __name__ == "__main__":
    asyncio.run(main())
