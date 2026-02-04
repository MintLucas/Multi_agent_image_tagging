from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()

user_id = "1"
namespace_for_memory1 = (user_id, "memories")
namespace_for_memory2 = ("2", "memories")

import uuid  # 用于生成唯一ID
memory_id = str(uuid.uuid4())  # 生成唯一的记忆ID（比如 "07e0caf4-..."）
memory = {"food_preference" : "I like pizza"}  # 实际要存的记忆数据（字典格式）
in_memory_store.put(namespace_for_memory1, memory_id, memory)


memory_id = str(uuid.uuid4())  # 生成唯一的记忆ID（比如 "07e0caf4-..."）
memory = {"location" : "San Francisco"}  # 实际要存的记忆数据（字典格式）
in_memory_store.put(namespace_for_memory2, memory_id, memory)


memories = in_memory_store.search(namespace_for_memory1)
print(f"namespace_for_memory1: {memories}")
memories = in_memory_store.search(namespace_for_memory2)
print(f"namespace_for_memory2: {memories}")
# {'value': {'food_preference': 'I like pizza'},
#  'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
#  'namespace': ['1', 'memories'],
#  'created_at': '2024-10-02T17:22:31.590602+00:00',
#  'updated_at': '2024-10-02T17:22:31.590605+00:00'}



# from langchain.embeddings import init_embeddings

# store = InMemoryStore(
#     index={
#         "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
#         "dims": 1536,                              # Embedding dimensions
#         "fields": ["food_preference", "$"]              # Fields to embed
#     }
# )


