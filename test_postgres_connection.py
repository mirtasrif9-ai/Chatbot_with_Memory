
import uuid

from langgraph.store.postgres import PostgresStore


DB_URL = "postgresql://postgres:postgres@localhost:5432/memory_db"

NAMESPACE = ("test", "user1", "memory")


print("Connecting to PostgreSQL...")

try:
    with PostgresStore.from_conn_string(DB_URL) as store:

        print("SUCCESS: PostgreSQL connection established.")

        # Create LangGraph PostgreSQL tables if needed
        store.setup()

        print("SUCCESS: LangGraph PostgreSQL store setup completed.")

        # --------------------------------------------------
        # Write a test memory
        # --------------------------------------------------

        memory_id = str(uuid.uuid4())

        test_memory = "My name is Maradona."

        store.put(
            NAMESPACE,
            memory_id,
            {
                "data": test_memory
            }
        )

        print("SUCCESS: Test memory written to PostgreSQL.")
        print("Saved:", test_memory)

        # --------------------------------------------------
        # Read the memory back
        # --------------------------------------------------

        memories = store.search(
            NAMESPACE
        )

        print("\nMemories found in PostgreSQL:")

        for item in memories:
            print("ID   :", item.key)
            print("Data :", item.value)

        # --------------------------------------------------
        # Verify our test memory exists
        # --------------------------------------------------

        found = any(
            item.value.get("data") == test_memory
            for item in memories
        )

        if found:
            print("\nSUCCESS: PostgreSQL read/write test PASSED.")
        else:
            print("\nFAILED: Memory was written but could not be read back.")


except Exception as e:
    print("\nFAILED: LangGraph PostgreSQL test.")
    print("Error:")
    print(e)
