
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
import pickle
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_community.vectorstores import Chroma

class databaseService:
    def __init__(self):     
        self.MODEL = 'orca-mini'
        self.embedder = OllamaEmbeddings(model=self.MODEL)

    def createBaseFromDocument(self,path,baseName):
        loader = PyMuPDFLoader(path)
        data = loader.load()
        
        oembed = OllamaEmbeddings(model=self.MODEL)
        pages = []
        data= data[3:len(data)]
        for docu in data:
            newtext = docu.page_content
            if newtext != '':
                pages.append(newtext)

        # Join all the pages into a single string
        huge_text = ' '.join(pages)

        # Now huge_text contains all the page_content joined together
        #print(huge_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=70,
            separators=["\n22","\n\n", "\n", " ", ""] # splits first with '\n\n' separator then split with '\n', ... until the chunk has the right size
        )

        all_splits = text_splitter.split_documents(data)

        #Display what will be stored in the database
        #splitted document, could be better currated
        #pprint(all_splits)

        #Create the vectorial database

        file_path = "./"+baseName
        if not os.path.exists(file_path):
            vectorstoreChroma = Chroma.from_documents(documents=all_splits, embedding=oembed,persist_directory=file_path)
        # load from disk
        else :
            vectorstoreChroma = Chroma(persist_directory=file_path, embedding_function=oembed)
        return vectorstoreChroma

    def loadDatabase(self,path):
        return Chroma(persist_directory=path, embedding_function=self.embedder)

