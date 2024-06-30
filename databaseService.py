
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
import pickle
import faiss
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from openai import OpenAI
import json


class databaseService:
 
    def __init__(self):     
        self.MODEL = 'orca-mini'
        self.embedder = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.dimension = 768  # Dimension of the vector
        self.index = faiss.IndexFlatL2(self.dimension)  # Build the index
        self.documents = []


    def get_embedding(self,text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
        text = text.replace("\n", " ")
        return self.embedder.embeddings.create(input = [text], model=model,dimensions=self.dimension).data[0].embedding
    
    def create_embedding_object(self,embeddings_list):
        embeddings_array = np.array(embeddings_list)
        print(embeddings_array.shape)
        reshaped_embeddings = embeddings_array.reshape((-1, embeddings_array.shape[-1]))


        return reshaped_embeddings
    
    #Modifier cette fonction pour add a la db et au JSON
    def createBaseFromDocument(self,filePath): # add doc to base
        

        if  os.path.exists('documents/vectorialDB.db'):
            self.index = faiss.read_index('documents/vectorialDB.db')

        # Commencer par load la db si existe

        loader = PyMuPDFLoader(filePath)
        data = loader.load()
        print("\n\ndone")
        pages = []
        for docu in data:
            newtext = docu.page_content
            if newtext != '':
                pages.append(newtext)

        print(pages)
        # Join all the pages into a single string
        huge_text = ' '.join(pages)

        # Now huge_text contains all the page_content joined together
        #print(huge_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n22"] # splits first with '\n\n' separator then split with '\n', ... until the chunk has the right size
        )

        all_splits = text_splitter.split_documents(data)
        print("\n\n")
        splits = []
        for docu in all_splits:
            newtext = docu.page_content
            if newtext != '':
                splits.append(newtext)
        allEmbeddings = []
        for split in splits:
            embed= self.get_embedding(split)
            allEmbeddings.append(embed)
        
        JSONList = []
        dataBasePath = 'documents/database.json'
        if os.path.exists(dataBasePath):
            with open(dataBasePath) as f:
             JSONList = json.loads(f.read())
        for a in range(len(JSONList),len(JSONList)+len(splits)):
            JSONList.append({"id": a,"text":splits[a]})
        self.documents = JSONList
        with open(dataBasePath, 'w') as f:
            json.dump(JSONList, f)
        
        data = self.create_embedding_object(allEmbeddings)
        self.index.train(data)
        self.index.add(data)

        faiss.write_index(self.index, 'documents/vectorialDB.db')
       
        return self.index

    def addFromTextDocument(self,docPath):
        #Split le text du doc
        with open('documents/'+docPath) as f:
            text = f.read()
        
        separator="kkk"
        documents = text.split(separator)
        allEmbeddings = []
        for split in documents:
            embed= self.get_embedding(split)
            #enregistrer les splits dans la db
            allEmbeddings.append(embed)

        JSONList = []
        dataBasePath = 'documents/database.json'
        if os.path.exists(dataBasePath):
            with open(dataBasePath) as f:
             JSONList = json.loads(f.read())
        for a in range(len(JSONList),len(JSONList)+len(documents)):
            JSONList.append({"id": a,"text":documents[a-len(JSONList)]})
        self.documents = JSONList
        with open(dataBasePath, 'w') as f:
            json.dump(JSONList, f)

        data = self.create_embedding_object(allEmbeddings)
        self.index.train(data)
        self.index.add(data)

        faiss.write_index(self.index, 'documents/vectorialDB.db')
       

    def createDatabaseFromJSON(self):
        JSONList = []
        if os.path.exists('documents/database.json'):
            JSONList = json.loads(f.read())
        else :
            print("No JSON file found")
            return
        allEmbeddings = []
        for doc in JSONList:
            embed= self.get_embedding(doc["text"])
            allEmbeddings.append(embed)

        self.index = faiss.IndexFlatL2(self.dimension)
        data = self.create_embedding_object(allEmbeddings)
        self.index.train(data)
        self.index.add(data)
        faiss.write_index(self.index, 'documents/vectorialDB.db')
        #Reload The vectorial database
        pass

    def loadDatabase(self):
        self.index = faiss.read_index('documents/vectorialDB.db')
        with open('documents/database.json') as f:
            self.documents = json.loads(f.read())
        return self.index

    def search(self,query):
        query_embedding = [self.get_embedding(query)]
        query_embedding = np.array(query_embedding)
        distances,indices = self.index.search(query_embedding,2 )
        response = ""
        print(indices)
        for doc in self.documents:
            if doc["id"] == indices[0][0]:
                response = doc["text"]
                break

        print(response)
        return response
        # if self is not None:
            
        # else:
        #     texts = self.session.query(Text).all()
        #     for text in texts:
        #         print(text.id, text.text)
        #     print("No text found with id"+str(indices[0][0]))
        #     return "I haven't found any information after the research"
        
      

        k = 5
if __name__ == "__main__":
    db = databaseService()
  #  db.createBaseFromDocument("./documents/inf-basse.pdf")
    db.addFromTextDocument("manualDocuments.txt")
    # db.loadDatabase("./flicflac2.db.faiss")
    # search = db.search(" Inf Bass plugin ?")
    # # print("done")


    # print("#######################\n DB CREATION")
    # import json

    # data_string = '[{"id":1,"text":"toto"},{"id":"2","text":"bite"}]'
    # with open('documents/inf-basse.pdf.json') as f:
    #     string = f.read()
    #     print(string)
    #     data = json.loads(string)

    # print(data[0]) 