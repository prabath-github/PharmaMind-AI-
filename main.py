import os
import re
import hashlib
import numpy as np
from dotenv import load_dotenv

# =========================
# Imports
# =========================
import pymupdf4llm
from huggingface_hub import hf_hub_download
from sentence_transformers import CrossEncoder
from langchain_text_splitters import (MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatLlamaCpp

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
model_path = hf_hub_download(
    repo_id="DhruvalLabs/Qwen3-4B-Instruct-2507-GGUF",
    filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)

# =========================
# Loading and chunking PDF using pymupdf4llm
# =========================

# FIXED: Added 'r' to properly handle Windows file paths
load_dotenv()
file_path = os.getenv(
    "PHARMA_PDF_PATH",
    r"V:\PROJECTS\pharma book AI\pharma_book.pdf"
)
if not os.path.isfile(file_path):
    raise FileNotFoundError(
        f"PDF not found: {file_path}"
    )
COLLECTION_NAME = "pharma_test"
PERSIST_DIR = "chroma_test_db"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

FETCH_K = 30
MMR_FETCH_K = 60
FINAL_K = 5
SIMILARITY_THRESHOLD = 0.40

# =========================
#  LOAD PDF
# =========================
# 1. convert pdf to markdown
print("Converting PDF to Markdown...")
md_text = pymupdf4llm.to_markdown(file_path, use_ocr=False)
print(f"Markdown characters: {len(md_text)}")

# =========================
#  MARKDOWN HEADER SPLITTING
# =========================
# 2. Define your Markdown headers
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)

sections = header_splitter.split_text(md_text)

print(f"Sections: {len(sections)}")

# =========================
# Recursive chunking
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True
)

chunks = text_splitter.split_documents(sections)

print(f"Chunks before deduplication: {len(chunks)}")

# =========================
# REMOVE DUPLICATE CHUNKS
# =========================

def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def prepare_chunks(documents):
    unique_chunks = []
    seen = set()

    for doc in documents:
        normalized = normalize_text(doc.page_content)

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)

        # Stable ID based on normalized text
        chunk_id = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        doc.metadata["chunk_id"] = chunk_id

        unique_chunks.append(doc)

    return unique_chunks


chunks = prepare_chunks(chunks)

print(f"Chunks after deduplication: {len(chunks)}")

# =========================
# Create embeddings
# =========================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# =========================
# 5. Create vector database
# =========================

print("Creating or loading vector database...")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIR
)

# Deterministic IDs prevent duplicate insertion
# of the same chunks on subsequent runs.

# =========================
# BATCH INSERT INTO CHROMA
# =========================

BATCH_SIZE = 500

print(f"Total chunks: {len(chunks)}")
print("Inserting chunks into ChromaDB...")

for start in range(0, len(chunks), BATCH_SIZE):
    batch_chunks = chunks[start:start + BATCH_SIZE]

    batch_ids = [
        doc.metadata["chunk_id"]
        for doc in batch_chunks
    ]

    vectorstore.add_documents(
        documents=batch_chunks,
        ids=batch_ids
    )

    print(
        f"Inserted {min(start + BATCH_SIZE, len(chunks))}"
        f" / {len(chunks)} chunks"
    )

print("Vector database ready!")

# =========================
# LOAD CROSS-ENCODER
# =========================

print("Loading reranker...")

reranker = CrossEncoder(RERANKER_ID)

# =========================
# RETRIEVAL FUNCTION
# =========================

def retrieve_chunks(question):

    # A. MMR retrieval
    # Retrieve a diverse candidate pool.

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": FETCH_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": 0.65
        }
    )

    candidates = retriever.invoke(question)

    print(f"\nMMR candidates: {len(candidates)}")

    # B. Deduplicate retrieved documents

    unique_candidates = []
    seen_ids = set()
    seen_text = set()

    for doc in candidates:
        chunk_id = doc.metadata.get("chunk_id")
        normalized = normalize_text(doc.page_content)

        if chunk_id and chunk_id in seen_ids:
            continue

        if normalized in seen_text:
            continue

        if chunk_id:
            seen_ids.add(chunk_id)

        seen_text.add(normalized)
        unique_candidates.append(doc)

    # C. Cosine similarity filtering

    if not unique_candidates:
        return []

    query_embedding = np.asarray(
        embedding_model.embed_query(question),
        dtype=np.float32
    )

    document_embeddings = np.asarray(
        embedding_model.embed_documents(
            [doc.page_content for doc in unique_candidates]
        ),
        dtype=np.float32
    )

    query_norm = np.linalg.norm(query_embedding)
    document_norms = np.linalg.norm(
        document_embeddings, axis=1
    )

    similarities = (
        document_embeddings @ query_embedding
    ) / np.maximum(query_norm * document_norms, 1e-12)

    filtered = [
        (doc, float(score))
        for doc, score in zip(
            unique_candidates, similarities
        )
        if score >= SIMILARITY_THRESHOLD
    ]

    print(f"After similarity filtering: {len(filtered)}")

    if not filtered:
        return []

    # D. Cross-encoder reranking

    pairs = [
        (question, doc.page_content)
        for doc, _ in filtered
    ]

    rerank_scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    ranked = sorted(
        zip(filtered, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    # E. Select final five

    final_results = []

    for (doc, cosine_score), rerank_score in ranked[:FINAL_K]:
        doc.metadata["cosine_similarity"] = round(
            cosine_score, 4
        )
        doc.metadata["rerank_score"] = round(
            float(rerank_score), 4
        )

        final_results.append(doc)

    return final_results

# =========================
# 10. LLM
# =========================

llm = ChatLlamaCpp(
    model_path=model_path,
    n_ctx=8192,
    n_threads=4,
    n_batch=256,
    temperature=0.1,
    max_tokens=1024,
    verbose=False
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a pharmaceutical reference assistant.

Your task is to answer the user's question using
the supplied reference sources.

Follow these instructions carefully:

1. Read ALL the reference sources before answering.
2. Identify every piece of information relevant
   to the question.
3. Combine relevant information from different
   sources when necessary.
4. Do not ignore relevant facts, figures, or
   explanations present in the sources.
5. Preserve drug names, dosages, units and
   technical terminology exactly as provided.
6. Do not introduce facts from your own knowledge.
7. Do not make assumptions or invent missing details.
8. If the sources do not contain sufficient
   information, explicitly state what is missing.
9. Support important statements with source numbers.
10. Answer the exact question asked. Do not replace
    a request for drug names with an explanation
    of enzymes or metabolic pathways.
11. If the question asks "Drugs Whose Metabolism
    Is Enhanced", identify and list those drugs
    explicitly from the reference sources.
12. Do not substitute general information about
    CYP enzymes for the requested drug names.
13. If the requested drug names are not explicitly
    available in the reference sources, say so.

First identify the relevant evidence, then formulate
a clear and complete answer.

Reference material:
{context}"""
    ),
    ("human", "{question}")
])



# =========================
# INTERACTIVE CHAT LOOP
# =========================
'''
while True:
    question = input(
        "\nEnter your question (or type 'exit'): "
    ).strip()

    if question.lower() == "exit":
        print("Exiting Pharma AI chatbot.")
        break

    if not question:
        continue

    # Retrieve relevant chunks
    results = retrieve_chunks(question)

    # Display retrieved chunks
    print("\n========== RETRIEVED CHUNKS ==========")
    print(f"Total chunks: {len(results)}")

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(
            f"Cosine similarity: "
            f"{doc.metadata.get('cosine_similarity')}"
        )
        print(
            f"Rerank score: "
            f"{doc.metadata.get('rerank_score')}"
        )
        print(f"Metadata: {doc.metadata}")
        print("\nChunk content:\n")
        print(doc.page_content)
        print("\n" + "=" * 70)


    if not results:
        print("No relevant chunks found.")
        continue

    # =========================
    # PREPARE CONTEXT
    # =========================

    context = "\n\n".join(
        f"Source {i}:\n{doc.page_content}"
        for i, doc in enumerate(results, start=1)
    )
    
    # =========================
    # PREPARE CONTEXT
    # =========================

    context = "\n\n".join(
        f"Source {i}:\n{doc.page_content}"
        for i, doc in enumerate(results, start=1)
    )

    # =========================
    # QWEN ANSWER GENERATION
    # =========================

    print("\n========== GENERATED ANSWER ==========")
    print("Sending context to Qwen...")
    print(f"Context characters: {len(context)}")

    try:
        chain = prompt | llm

        response = chain.invoke({
            "question": question,
            "context": context
        })

        answer = response.content

        if isinstance(answer, str) and answer.strip():
            print("\nFinal LLM Answer:\n")
            print(answer.strip())
        else:
            print("Qwen returned an empty answer.")
            print("Raw response:", repr(response))

    except Exception as e:
        import traceback
        print(f"Qwen error: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Ready for your next question.")
'''
    