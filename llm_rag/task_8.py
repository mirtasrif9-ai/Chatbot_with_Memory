from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
K = 3

PDF_PATH = r"C:\Users\bjit\Downloads\attention.pdf"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"


# ============================================================
# 1. LOAD PDF
# ============================================================

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()


# ============================================================
# 2. SPLIT PDF INTO CHUNKS
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(docs)


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# 4. CREATE FAISS VECTOR STORE
# ============================================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# ============================================================
# 5. CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": K
    }
)


# ============================================================
# 6. LOAD QWEN LLM
# ============================================================

llm = ChatOllama(
    model=LLM_MODEL
)


# ============================================================
# 7. DEFINE RAG STATE
# ============================================================

class RAGState(TypedDict):
    question: str
    expected_keyword: str
    context: str
    answer: str
    correct_context_found: bool


# ============================================================
# 8. RETRIEVE CONTEXT
# ============================================================

def retrieve_context(state: RAGState):

    results = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    found = (
        state["expected_keyword"].lower()
        in context.lower()
    )

    return {
        "context": context,
        "correct_context_found": found
    }


# ============================================================
# 9. GENERATE ANSWER
# ============================================================

def generate_answer(state: RAGState):

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{state["context"]}

Question:
{state["question"]}

Rules:
- Use only information from the context.
- Do not invent information.
- If the answer cannot be found in the context,
  say "I don't know based on the provided context."
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content.strip()
    }


# ============================================================
# 10. BUILD LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(RAGState)

workflow.add_node(
    "retrieve_context",
    retrieve_context
)

workflow.add_node(
    "generate_answer",
    generate_answer
)

workflow.add_edge(
    START,
    "retrieve_context"
)

workflow.add_edge(
    "retrieve_context",
    "generate_answer"
)

workflow.add_edge(
    "generate_answer",
    END
)


# ============================================================
# 11. COMPILE WORKFLOW
# ============================================================

app = workflow.compile()


# ============================================================
# 12. TEST QUESTIONS
# ============================================================

test_questions = [
    (
        "What is scaled dot-product attention?",
        "scaled dot-product"
    ),
    (
        "How does multi-head attention work?",
        "multi-head"
    ),
    (
        "How does the Transformer encode position?",
        "positional encoding"
    ),
    (
        "What replaces recurrence in the Transformer?",
        "self-attention"
    ),
]


# ============================================================
# 13. RUN TESTS
# ============================================================

correct = 0

for question, keyword in test_questions:

    result = app.invoke(
        {
            "question": question,
            "expected_keyword": keyword
        }
    )

    print(
        question,
        "->",
        result["correct_context_found"]
    )

    print(
        "Answer:",
        result["answer"]
    )

    print("-" * 80)

    correct += result["correct_context_found"]


# ============================================================
# 14. RETRIEVAL ACCURACY
# ============================================================

print(
    f"Retrieval accuracy: "
    f"{correct}/{len(test_questions)} "
    f"(chunk_size={CHUNK_SIZE}, "
    f"overlap={CHUNK_OVERLAP}, "
    f"k={K})"
)