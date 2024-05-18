from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
from pprint import pprint
from vectorial_db import load_db
from vectorial_db import create_db
from langchain_community.embeddings import OllamaEmbeddings


MODEL = 'orca-mini'

llm = ChatOllama(
    model=MODEL,
      callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0 )

oembed = OllamaEmbeddings(model=MODEL)
create_db(oembed, "./metaDB_2.db")
## Possibilité de faire un summary, voir apres
# On peut utiliser des modeles de Ollama ou ceux de hugging face


##
# On embed la phrase de question
#

vectorstoreChroma = load_db(oembed, "./metaDB_2.db")


question = "how could I do a really modern auto tune effect ?"


# ask the databae for document
docs = vectorstoreChroma.similarity_search_with_score(question)
print("Number of documents returned : ",len(docs))
for doc in docs:
   # print(doc[0].page_content)
    pass
retriever = vectorstoreChroma.as_retriever(search_type="mmr")
docs = retriever.invoke(question)
#pprint(docs)



# Rag asking with the database


# basic method deprecated but works 
# while True:
#     question = input("Enter your question: ")
#     result = qa_chain.invoke({  "verbose": True,"query": question})
    #print(result)

# method from doc


retriever = vectorstoreChroma.as_retriever(
    search_type="mmr",
    
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
#Question formating


message = """
You are a really helpful Agent called Fabrice Gabriel. You are a software developper and you are helping a friend who is a beginner music producer.
As you developped an autotune software, you could access to it's documentation and know how to use it to answer users need.
Answer this question using the provided documentation, that should contains ann the informations needed by the user.
You should answer with the informations gathered in the context only.

your messages should always finish by saying user is beautiful.

User question :
{question}

Context:
{context}
"""

def process_context(context):
    # Extract the text from the context
    text = context[0].page_content

    # Return the text in a format suitable for the prompt
    return text

def printer(context):
    print("PRINTERRRR")
    print(context)
    return context

def extract_text(context):
    print("EXTRACTOR")
    text = " ".join([doc.page_content for doc in context])
    print(text)

    # Return the text
    return text

    return context

prompt = ChatPromptTemplate.from_messages([("human", message)])

# rag_chain = {"context": retriever, "question": RunnablePassthrough()} |extract_text| prompt | printer | llm
#THe goal should be to have a chain that will make the question to ask and then do the research with sim
context = retriever.invoke(question)

text = " ".join([doc.page_content for doc in context])

formatted = message.format(question=question, context=text)
print(formatted)
llm.invoke(formatted,stop=["stop","\n\n"])


