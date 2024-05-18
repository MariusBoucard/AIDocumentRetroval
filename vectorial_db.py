
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from pprint import pprint
def create_db(embedding, file_path):
    loader = PyMuPDFLoader("./documents/metatune.pdf")
    data = loader.load()


    data= data[3:len(data)]

    # Join all the pages into a single string

    # Now huge_text contains all the page_content joined together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=10,
        separators=["\n22","\n\n", "."] # splits first with '\n\n' separator then split with '\n', ... until the chunk has the right size
    )

    all_splits = text_splitter.split_documents(data)
    for a in all_splits:
        print(a.page_content)
        print("##################################################################################\n\n")
#Display what will be stored in the database
#splitted document, could be better currated
#pprint(all_splits)

#Create the vectorial database or load it 

    import os
    if not os.path.exists(file_path):
        print()
        vectorstoreChroma = Chroma.from_documents(documents=all_splits, embedding=embedding,persist_directory=file_path)
    
    
def load_db(embedding, file_path):
    return Chroma(persist_directory=file_path, embedding_function=embedding)