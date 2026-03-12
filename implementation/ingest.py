# Standard library: filesystem and path utilities.
import os
# Standard library: glob pattern matching for file paths.
import glob
# Standard library: object-oriented path handling.
from pathlib import Path
# LangChain: load documents from directories and from plain text files.
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pydantic import BaseModel, Field
# LangChain: Chroma vector store for embeddings.
from langchain_chroma import Chroma
# LangChain: OpenAI embeddings and chat.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# LangChain: message types
from langchain_core.messages import SystemMessage, HumanMessage
# LangChain: message types
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm
import uuid


# Load environment variables from .env (e.g. API keys).
from dotenv import load_dotenv

# Load .env and override any existing env vars.
load_dotenv(override=True)

MODEL = "gpt-5-nano"

# Path to the Chroma vector DB directory (project root / vector_db).
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
# Path to the knowledge base directory (project root / knowledge-base).
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Embedding model used to turn text into vectors (OpenAI text-embedding-3-small by default).
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



AVERAGE_CHUNK_SIZE = 500

# LLM for structured output (chunking)
llm = ChatOpenAI(temperature=0, model_name=MODEL)

class Result(BaseModel):
    page_content: str
    metadata: dict
    id: str | None = None  # Chroma.from_documents expects .id on each doc



    # A class to perfectly represent a chunk

class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")

    def as_result(self, document):
        metadata = {"source": document.metadata.get("source"), "type": document.metadata.get("doc_type")}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
            id=str(uuid.uuid4()),
        )


class Chunks(BaseModel):
    chunks: list[Chunk]



def fetch_documents():
    # List all top-level folders under the knowledge base.
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    # Accumulate loaded documents.
    documents = []
    for folder in folders:
        # Use folder name as document type (e.g. "docs", "faq").
        doc_type = os.path.basename(folder)
        # Loader: all .md files under this folder, UTF-8 text loader.
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        # Load documents from this folder.
        folder_docs = loader.load()
        for doc in folder_docs:
            # Tag each doc with its folder (doc_type).
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents

def make_prompt(document):
    how_many = (len(document.page_content) // AVERAGE_CHUNK_SIZE) + 1
    type = document.metadata.get("doc_type")
    source = document.metadata.get("source")
    return f"""
    You take a document and you split the document into overlapping chunks for a KnowledgeBase.

    The document is from the shared drive of a company called Insurellm.
    The document is of type: {type}
    The document has been retrieved from: {source}

    A chatbot will use these chunks to answer questions about the company.
    You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
    This document should probably be split into {how_many} chunks, but you can have more or less as appropriate with focus on completeness and relevance.
    There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

    For each chunk, you should provide a headline, a summary, and the original text of the chunk.
    Together your chunks should represent the entire document with overlap.

    Here is the document:

    {document.page_content}

    Respond with the chunks.
    """


def process_document(document):
    messages = [
        HumanMessage(content=make_prompt(document))
    ]
    # Create structured LLM for Chunks response format
    chunk_llm = llm.with_structured_output(Chunks)
    chunks_result = chunk_llm.invoke(messages)
    return [chunk.as_result(document) for chunk in chunks_result.chunks]

def create_chunks(documents):
    chunks = []
    for doc in tqdm(documents):
        chunks.extend(process_document(doc))
    return chunks

def create_embeddings(chunks):
    # If DB already exists, delete its collection so we rebuild from scratch.
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    # Build a new Chroma store from chunks using our embedding model.
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    # Access Chroma's underlying collection for count and sample.
    collection = vectorstore._collection
    count = collection.count()

    # Get one embedding to report vector dimensions.
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore

if __name__ == "__main__":
    # Load all .md files from knowledge-base subfolders.
    documents = fetch_documents()
    # Chunk each document via LLM (headline, summary, original_text per chunk).
    chunks = create_chunks(documents)
    # Embed chunks and persist to vector_db (Chroma).
    create_embeddings(chunks)
    print("Ingestion complete")
