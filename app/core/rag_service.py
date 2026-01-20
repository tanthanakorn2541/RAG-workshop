from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from app.core.config import settings


class RAGService:
    def __init__(self, retriever):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,
        )

        # ✅ IMPORTANT: prompt MUST include {context}
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "คุณเป็นผู้ช่วย AI ที่ตอบคำถามเป็นภาษาไทยเท่านั้น\n"
                "ใช้ข้อมูลต่อไปนี้ในการตอบคำถาม:\n\n"
                "{context}\n\n"
                "ห้ามใช้ความรู้ภายนอกข้อมูลนี้\n"
                "ถ้าไม่พบข้อมูล ให้ตอบว่าไม่พบข้อมูลที่เกี่ยวข้อง"
            ),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )

        self.qa_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

    def answer_query(self, query: str) -> str:
        result = self.qa_chain.invoke({"input": query})
        return result["answer"]
