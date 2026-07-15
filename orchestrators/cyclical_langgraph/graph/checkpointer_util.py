from langgraph.checkpoint.memory import MemorySaver

def get_checkpointer(backend: str = "memory"):
    if backend == "postgres":
        # from langgraph.checkpoint.postgres import PostgresSaver
        # return PostgresSaver.from_conn_string(os.getenv("CHECKPOINT_DATABASE_URL"))
        print("PostgresSaver not fully implemented in v1. Falling back to MemorySaver.")
        return MemorySaver()
    
    return MemorySaver()
