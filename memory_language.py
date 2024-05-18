

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

m = conversation.predict(input="Hi I m marius, tell my how I could fuck off")

