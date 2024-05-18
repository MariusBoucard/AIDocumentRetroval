
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from pprint import pprint

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
def create_db(embedding, file_path):
    loader = PyMuPDFLoader("./documents/metatune.pdf")
    data = loader.load()


    data= data[3:len(data)]
    print(data)
    # Join all the pages into a single string
    huge_text = ""
    for page in data:
        huge_text += page.page_content

    print(huge_text)
        
    # Now huge_text contains all the page_content joined together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
        separators=["\n22","\n\n", "."] # splits first with '\n\n' separator then split with '\n', ... until the chunk has the right size
    )

    # Tokenize the text
    words = word_tokenize(huge_text)

    # Part of Speech (POS) tagging
    pos_tags = pos_tag(words)

    # Named Entity Recognition
    tree = ne_chunk(pos_tags)

    # Print named entities
    for subtree in tree.subtrees():
        # if subtree.label() == 'NE':
            print(subtree)
    print('SHOULD HAVE BEEN SUBTREE \n\n\n\n')
    all_splits = text_splitter.split_documents(data)
    # for a in all_splits:
    #     print(a.page_content)
    #     print("##################################################################################\n\n")
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