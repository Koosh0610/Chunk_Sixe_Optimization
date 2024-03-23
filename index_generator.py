from llama_index.legacy import SimpleDirectoryReader
from llama_index.legacy import ServiceContext, VectorStoreIndex
from llama_index.legacy.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
chunk_size = [500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
for chunk_size in chunk_size:
        chunk_overlap = 0.10*chunk_size
        documents = SimpleDirectoryReader(input_dir="India_States_UT").load_data()
        service_context = ServiceContext.from_defaults(llm=None,embed_model=embed_model, chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        index = VectorStoreIndex.from_documents(documents=documents, service_context=service_context,show_progress=True)
        index.storage_context.persist(f"index_{chunk_size}_{chunk_overlap}")