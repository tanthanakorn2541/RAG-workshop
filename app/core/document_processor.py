import json
import logging
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SeniorSemanticProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
        )

    def process_metadata(self, json_path: str) -> list[Document]:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            all_docs: list[Document] = []

            for segment in data:
                content = (
                    f"หัวข้อวิเคราะห์: {segment.get('segmentName')}. "
                    f"สรุปภาพรวม: {segment.get('summary')}. "
                    f"คำแนะนำ: {segment.get('recommendation')}. "
                )

                for tag in segment.get("topHeadTags", []):
                    content += (
                        f"ประเด็นเรื่อง {tag.get('headTagNameThai')}: "
                        f"{tag.get('summary')} "
                    )

                docs = self.text_splitter.create_documents(
                    [content],
                    metadatas=[{
                        "source": json_path,
                        "segment": segment.get("segmentName"),
                        "type": "senior_analysis",
                    }]
                )

                all_docs.extend(docs)

            logger.info(
                "Semantic chunking completed: Created %d chunks",
                len(all_docs)
            )

            return all_docs

        except Exception as e:
            logger.exception("Semantic Processing Error")
            raise
