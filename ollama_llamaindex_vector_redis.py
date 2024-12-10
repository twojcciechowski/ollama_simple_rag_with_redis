from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.schema import BaseNode, TextNode
from llama_index.legacy.vector_stores import RedisVectorStore, VectorStoreQuery
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse, ResultType


llm = Ollama(
    model="llama3.2:3b-instruct-q8_0",
    base_url="http://localhost:11434",
    request_timeout=30.0,
    keep_alive=True,
    temperature=0.1
)

def create_new_vectorDb():
    vector_store = RedisVectorStore(redis_url="redis://localhost:6379", overwrite=True, index_name="test_idx")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = resolve_embed_model("local:sentence-transformers/all-mpnet-base-v2")
    parser = LlamaParse(result_type=ResultType.TXT, api_key="")
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_dir="./data", file_extractor=file_extractor).load_data()
    VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, embed_model=embed_model)


def run():
    vector_store = RedisVectorStore(redis_url="redis://localhost:6379", index_name="test_idx")
    embed_model = resolve_embed_model("local:sentence-transformers/all-mpnet-base-v2")

    query = VectorStoreQuery(
        query_embedding=embed_model.get_query_embedding(query="who is author of DORA report?"),
    )
    result = vector_store.query(query=query)
    print(result)
    nodes = []
    for n in result.nodes:
        nodes.append(TextNode.from_dict(n.to_dict()))

    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)
    result = query_engine.query("who are authors of DORA report?")
    print(result)


# create_new_vectorDb()
run()