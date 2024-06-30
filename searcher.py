from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from databaseService import databaseService

class Searcher:
    def __init__(self):
        self.dataBase = None
        self.MODEL = 'orca-mini'
        self.embedder = OllamaEmbeddings(model=self.MODEL)
        self.dataBaseService = databaseService()

    def loadDatabase(self):
        self.dataBase = self.dataBaseService.loadDatabase()
    def createDatabase(self,fileName, path, baseName):
        self.dataBase = self.dataBaseService.createBaseFromDocument(path,fileName,baseName)

    def embedAndSearch(self, question):
        return self.dataBaseService.search(question)
        sentence_embedding=self.embedder.embed_query(question)
        docs = self.dataBase.similarity_search_with_score(sentence_embedding)
        print("Number of documents returned : ",len(docs))
        print(docs[0])
        # retriever = self.dataBase.as_retriever(search_type="mmr")
        
        # print("Number of documents returned : ",len(retriever))
        # print(retriever[0])
        self.documentsRetrieved = docs
        return docs
    
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

        

        # langchain.debug = True

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.dataBase.as_retriever(search_type="mmr"),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},

        )
        return self.embedAndSearch(question)