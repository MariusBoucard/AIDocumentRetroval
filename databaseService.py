
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
import pickle
import faiss
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from openai import OpenAI
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
        # Convert the list of embeddings to a numpy array
        embeddings_array = np.array(embeddings_list)
        print(embeddings_array.shape)
        # Reshape the array into the desired shape
        # In this case, we'll reshape it into a 2D array with shape (n, d)
        # where n is the number of embeddings and d is the dimension of each embedding
        reshaped_embeddings = embeddings_array.reshape((-1, embeddings_array.shape[-1]))


        return reshaped_embeddings
    def createBaseFromDocument(self,path,baseName):
        loader = PyMuPDFLoader(path)
        data = loader.load()
        print("\n\ndone")
        pages = []
        for docu in data:
            newtext = docu.page_content
            if newtext != '':
                pages.append(newtext)

        print(pages)
        self.documents = pages
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
            #embed = np.array(embed, dtype='float32').reshape(1, -1)
        print(allEmbeddings)
        data = self.create_embedding_object(allEmbeddings)
        self.index.train(data)
        self.index.add(data)
        print(split)
        #Create the vectorial database

        #Display what will be stored in the database
        #splitted document, could be better currated
        #pprint(all_splits)

        #Create the vectorial database

        file_path = "./"+baseName
        if not os.path.exists(file_path):
            faiss.write_index(self.index, file_path)
        # # load from disk
        else :
            self.index = faiss.read_index( file_path)
        #     vectorstoreChroma = Chroma(persist_directory=file_path, embedding_function=oembed)
        # return vectorstoreChroma
        return self.index

    def loadDatabase(self,path):
        self.index = faiss.read_index(path)
        return self.index

    def search(self,query):
        query_embedding = [self.get_embedding(query)]
        query_embedding = np.array(query_embedding)

       # query_embedding = self.create_embedding_object(query_embedding)
        distances,indices = self.index.search(query_embedding, 5)
        print(distances)
        print(indices)
        # recup le number one
        print(self.index.d)
        embedding = np.empty(self.index.d, dtype='float32')
        #Returns only the best document
        return self.documents[indices[0][0]]
      

        k = 5
if __name__ == "__main__":
    db = databaseService()
    db.createBaseFromDocument("./documents/inf-basse.pdf","./flicflac2.db.faiss")
    db.loadDatabase("./flicflac2.db.faiss")
    search = db.search(" Inf Bass plugin ?")
    # print("done")
