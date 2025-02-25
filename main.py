##
## pip install -u google-genai==0.5.0 llama-index==0.12.11 llama-index-llms-gemini==0.4.3 llama-index-embeddings-gemini==0.3.1 websockets
##
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
            "description": "Query the document content with a specific query string.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The query string to search the document index."
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
"""


def build_index(doc_path: str = DOWNLOADS_DIR) -> VectorStoreIndex:
    """
    Build or load a vector store index from documents.
    
    Args:
        doc_path: Directory containing documents to index
        
    Returns:
        The vector store index
    """
    if not os.path.exists(PERSIST_DIR):
        # Load documents and create new index
        documents = SimpleDirectoryReader(doc_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        # Store for later use
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    return index


def query_docs(query: str) -> str:
    """
    Query the document index with a specific query.
    
    Args:
        query: The query string to search for
        
    Returns:
        Response text from the query engine
    """
    index = build_index()
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    
    # Convert response to string
    response_text = str(response)
    print(f"RAG response: {response_text}")
    return response_text


async def handle_pdf_upload(data: Dict[str, Any], client_websocket: websockets.WebSocketServerProtocol) -> None:
    """
    Handle PDF file upload, save to disk and rebuild index.
    
    Args:
        data: The PDF chunk data
        client_websocket: The websocket connection to the client
    """
    pdf_data = base64.b64decode(data["data"])
    filename = data.get("filename", "uploaded.pdf")
    
    # Save the PDF file
    file_path = os.path.join(DOWNLOADS_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(pdf_data)
    
    print(f"Saved PDF file to {file_path}")
    
    # Rebuild the index with the new PDF
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    
    build_index()
    
    await client_websocket.send(json.dumps({
        "text": f"PDF file {filename} has been uploaded and indexed successfully."
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
                result = query_docs(args["query"])
                function_responses.append({
                    "name": name,
                    "response": {"result": result},
                    "id": call_id  
                }) 
                await client_websocket.send(json.dumps({"text": f"Searching for: {args['query']}..."}))
                print("Function executed")
            except Exception as e:
                print(f"Error executing function: {e}")
                continue

    # Send function responses back to Gemini
    print(f"Function responses: {function_responses}")
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
            await client_websocket.send(json.dumps({"text": part.text}))
        elif hasattr(part, 'inline_data') and part.inline_data is not None:
            base64_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
            await client_websocket.send(json.dumps({
                "audio": base64_audio,
            }))
            print("Audio received")


async def send_to_gemini(client_websocket: websockets.WebSocketServerProtocol, session: Any) -> None:
    """
    Send messages from the client websocket to the Gemini API.
    
    Args:
        client_websocket: The websocket connection to the client
        session: The Gemini session
    """
    try:
        async for message in client_websocket:
            try:
                data = json.loads(message)
                if "realtime_input" in data:
                    for chunk in data["realtime_input"]["media_chunks"]:
                        if chunk["mime_type"] == "audio/pcm":
                            await session.send(input={
                                "mime_type": "audio/pcm",
                                "data": chunk["data"]
                            })
                        elif chunk["mime_type"] == "application/pdf":
                            await handle_pdf_upload(chunk, client_websocket)
                            
            except Exception as e:
                print(f"Error sending to Gemini: {e}")
        print("Client connection closed (send)")
    except Exception as e:
        print(f"Error sending to Gemini: {e}")
    finally:
        print("send_to_gemini closed")


async def receive_from_gemini(client_websocket: websockets.WebSocketServerProtocol, session: Any) -> None:
    """
    Receive responses from Gemini API and forward them to the client.
    
    Args:
        client_websocket: The websocket connection to the client
        session: The Gemini session
    """
    try:
        while True:
            try:
                print("Receiving from Gemini")
                async for response in session.receive():
                    # Handle tool calls
                    if response.server_content is None:
                        if response.tool_call is not None:
                            print(f"Tool call received: {response.tool_call}")
                            await process_tool_call(response, client_websocket, session)
                        continue

                    # Process model turn if server_content is not None
                    if hasattr(response.server_content, 'model_turn') and response.server_content.model_turn:
                        await process_model_turn(response.server_content.model_turn, client_websocket)

                    # Check if turn is complete
                    if response.server_content and hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                        print('\n<Turn complete>')
                            
            except websockets.exceptions.ConnectionClosedOK:
                print("Client connection closed normally (receive)")
                break
            except Exception as e:
                print(f"Error receiving from Gemini: {e}")
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
        # Get config from client
        config_message = await client_websocket.recv()
        config_data = json.loads(config_message)
        config = config_data.get("setup", {})
        
        # Set system instruction and tools
        config["system_instruction"] = SYSTEM_INSTRUCTION
        config["tools"] = [TOOL_QUERY_DOCS]

        # Connect to Gemini
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("Connected to Gemini API")

            # Start send and receive tasks
            send_task = asyncio.create_task(send_to_gemini(client_websocket, session))
            receive_task = asyncio.create_task(receive_from_gemini(client_websocket, session))
            
            await asyncio.gather(send_task, receive_task)

    except Exception as e:
        print(f"Error in Gemini session: {e}")
    finally:
        print("Gemini session closed.")


async def main() -> None:
    """
    Main function to run the websocket server.
    """
    async with websockets.serve(gemini_session_handler, SERVER_HOST, SERVER_PORT):
        print(f"Running websocket server {SERVER_HOST}:{SERVER_PORT}...")
        await asyncio.Future()  # Keep the server running indefinitely


if __name__ == "__main__":
    asyncio.run(main())