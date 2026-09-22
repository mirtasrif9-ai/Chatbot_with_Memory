import uuid

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore


# ============================================================
# 1. PostgreSQL configuration
# ============================================================

DB_URL = "postgresql://postgres:postgres@localhost:5432/memory_db"

NAMESPACE = ("user", "u1", "details")


# ============================================================
# 2. LLM
# ============================================================

llm = ChatOllama(
    model="qwen2.5:3b"
)


# ============================================================
# 3. Helper function: read memories
# ============================================================

def get_memories(store: BaseStore):
    items = store.search(
        NAMESPACE,
        limit=100
    )

    return [
        item.value["data"]
        for item in items
        if "data" in item.value
    ]


# ============================================================
# 4. Remember node
# ============================================================

def remember(state: MessagesState, store: BaseStore):
    # Get existing memories from PostgreSQL
    known = get_memories(store)

    # Get latest user message
    user_message = state["messages"][-1].content

    # Ask LLM to extract a new personal fact
    prompt = f"""
You are a memory extraction system.

Your job is to extract ONLY a stable personal fact
that the USER explicitly states about themselves.

Examples of facts worth remembering:

"My name is Maradona."
"My favorite sport is football."
"I like cricket."
"I am a Python developer."
"I am working on a chatbot project."

Do NOT extract:
- questions
- greetings
- requests
- temporary conversation
- information about other people
- information invented by you

Existing memories:

{known}

Latest user message:

{user_message}

Rules:

1. If the user explicitly gives a new personal fact,
   return exactly ONE short sentence containing that fact.

2. If there is no new personal fact,
   return exactly:

NONE

3. Do not explain your answer.
4. Do not use quotation marks.
"""

    response = llm.invoke(prompt)

    fact = response.content.strip()

    # Do not save empty responses
    if not fact:
        return {}

    # No new memory
    if fact.upper() == "NONE":
        return {}

    # Avoid exact duplicates
    if fact in known:
        return {}

    # Save new memory to PostgreSQL
    memory_id = str(uuid.uuid4())

    store.put(
        NAMESPACE,
        memory_id,
        {
            "data": fact
        }
    )

    return {}


# ============================================================
# 5. Chat node
# ============================================================

def chat(state: MessagesState, store: BaseStore):
    # Retrieve long-term memories from PostgreSQL
    known = get_memories(store)

    # Convert memories to text
    if known:
        memory_text = "\n".join(
            f"- {memory}"
            for memory in known
        )
    else:
        memory_text = "(No long-term information is known.)"

    # System message containing memory
    system_message = SystemMessage(
        content=f"""
You are a helpful chatbot.

You have access to long-term information about the user
stored in PostgreSQL.

Use this information when relevant.

LONG-TERM USER MEMORY:
{memory_text}

Important:
- Treat the memory above as information about the user.
- If the user's question is directly answered by the memory,
  answer using that memory.
- Do not ask the user to repeat information that is already
  present in the memory.
- Do not invent memories.
- Do not mention that you are using a memory system or database
  unless the user specifically asks about it.
"""
    )

    # Generate response
    response = llm.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 6. Build LangGraph
# ============================================================

builder = StateGraph(MessagesState)

builder.add_node("remember", remember)
builder.add_node("chat", chat)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)


# ============================================================
# 7. Connect to PostgreSQL
# ============================================================

with PostgresStore.from_conn_string(DB_URL) as store:

    # Create required LangGraph tables if they don't exist
    store.setup()

    # Compile graph
    app = builder.compile(
        store=store,
        checkpointer=InMemorySaver()
    )

    # Short-term conversation thread
    config = {
        "configurable": {
            "thread_id": "t1"
        }
    }

    print("Chatbot started.")
    print("Type 'quit' or 'exit' to stop.")
    print()

    # ========================================================
    # 8. Chat loop
    # ========================================================

    while True:

        text = input("You: ").strip()

        # Exit
        if text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Ignore empty input
        if not text:
            continue

        # Run graph
        result = app.invoke(
            {
                "messages": [
                    ("user", text)
                ]
            },
            config
        )

        # Display only the chatbot response
        print(
            "Bot:",
            result["messages"][-1].content
        )
        print()