from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_dir = settings.CHROMA_DB_DIR
        self.collection_name = "rag_documents"

    def initialize_db(self, documents):
        if not documents:
            raise ValueError("No documents provided to initialize vector store")

        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        db.persist()
        logger.info("Chroma DB initialized with %d documents", len(documents))
        return db

    def get_retriever(self):
        db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        return db.as_retriever(search_kwargs={"k": 4})
