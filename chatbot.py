from models.chains import chain
from models.memory import chat_history

from rag.vector_store import create_vector_store

create_vector_store()
print("Starting Chat","\n")

while True:
    query=input("YOU: ")
    if query in ("EXIT","Exit","exit"):
        break
    response=chain.invoke(query)
    print(f"AI: {response}")
    print()
    print()

print("END CHAT")