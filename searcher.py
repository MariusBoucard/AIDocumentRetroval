import requests
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from databaseService import databaseService

class Searcher:
    def __init__(self):
        self.dataBase = None
        self.embedding_api_url = "http://127.0.0.1:1234/v1/embeddings"
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer YOUR_API_KEY"  # Replace with your actual API key if needed
        }
        self.dataBaseService = databaseService()

    def loadDatabase(self, path):
        self.dataBase = self.dataBaseService.loadDatabase(path)
    def createDatabase(self, path, baseName):
        self.dataBase = self.dataBaseService.createBaseFromDocument(path,baseName)

    def embedAndSearch(self, question :str):
        Documents = self.dataBaseService.embedAndSearch(question)
        return Documents
    
    def rietriveFromQA(self, question,llm):
            # Build prompt
        
        QA_TEMPLATE = """
        [INST] <<SYS>> Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for being my wonderful bro, I love you" at the end of the answer
        {self.documentRetrieved} <</SYS>>

        Question: {question}
        Helpful Answer: [/INST]
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)
        # QA chain
        print("Not made yet")

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.dataBase.as_retriever(search_type="mmr"),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},

        )
        return self.embedAndSearch(question)