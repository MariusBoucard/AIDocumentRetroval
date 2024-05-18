from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain.llms import Ollama
from langchain_community.chat_models import ChatOllama
# from langchain_community import invoke
from pprint import pprint
# langchain.debug = True # uncomment if you want to see what's going on under the hood
MODEL = 'orca-mini'
llm = ChatOllama(
    model=MODEL,
       # verbose=True,
      callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0 # creativity of the model from 0 to 1 (more creative)
)

#message = llm.invoke( "What carrer Obama would have really wanted to do?")
#can already talk to the model like this

#print(message)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

template="""
[INST] <<SYS>> The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. \n\nCurrent conversation:\n {history}
 <</SYS>>

\nHuman: {input}\nAI: [/INST]
"""

memory = ConversationBufferMemory() # ConversationBufferWindowMemory(k=1) # you can choose how long the history is kept
conversation = ConversationChain(
    llm=llm,
    memory = memory,
#    verbose=True,
    prompt=PromptTemplate(input_variables=['history', 'input'], template=template)
)
m = conversation.predict(input="Hi, my name is Camille")


## Possibilité de faire un summary, voir apres
from pprint import pprint # pretty print for printing lists
from langchain_community.embeddings import OllamaEmbeddings
# On peut utiliser des modeles de Ollama ou ceux de hugging face


##
# On embed la phrase de question
#
oembed = OllamaEmbeddings(model=MODEL)
sentence_embedding=oembed.embed_query("Llamas are social animals and live with others as a herd.")
sentence_embedding[0:10]

from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./sd.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./sd.pdf")
data = loader.load()
#pprint(data)
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
import pickle
import os
file_path = "./chroma.db"
if not os.path.exists(file_path):
    vectorstoreChroma = Chroma.from_documents(documents=all_splits, embedding=oembed,persist_directory=file_path)
# load from disk
else :
    vectorstoreChroma = Chroma(persist_directory=file_path, embedding_function=oembed)


question = "how could I do a really modern auto tune effect ?"


# ask the databae for document
docs = vectorstoreChroma.similarity_search_with_score(question)
print("Number of documents returned : ",len(docs))
pprint(docs)

retriever = vectorstoreChroma.as_retriever(search_type="mmr")
docs = retriever.invoke(question)
pprint(docs)



# Rag asking with the database

from langchain.prompts import PromptTemplate


# Build prompt
QA_TEMPLATE = """
[INST] <<SYS>> Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for being my wonderful bro, I love you" at the end of the answer
{context} <</SYS>>

Question: {question}
Helpful Answer: [/INST]
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)
QA_CHAIN_PROMPT
# QA chain
from langchain.chains import RetrievalQA
import langchain
# langchain.debug = True

qa_chain = RetrievalQA.from_chain_type(
    llm,
     retriever=vectorstoreChroma.as_retriever(search_type="mmr"),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},

)

# #result = qa_chain.invoke(question)
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain import hub

# combine_docs_chain = create_stuff_documents_chain(
#     llm, QA_CHAIN_PROMPT
# )
# retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# retrieval_chain.invoke({"input":question,"question": question})

while True:
    question = input("Enter your question: ")
    result = qa_chain.invoke({  "verbose": True,"query": question})
    #print(result)