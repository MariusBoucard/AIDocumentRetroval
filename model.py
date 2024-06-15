
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from searcher import Searcher
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import langchain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import random
import time
import copy
from openai import OpenAI
from llama_cpp import Llama

class Model:
    def __init__(self):
        self.model = None
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def response_generator(self):

        response = random.choice(
            [
                "Hello bro i m Steven Slate from Slate Digital, thanks to my huge knowledge in music production and my AI, I can help you with your music production questions.",
                "I'm Fabrie Gabriel, The best producer of indie music in the world, I can help you with your music production questions."
            ]
        )
        # for word in response.split():
        #     yield word + " "
        #     time.sleep(0.05)
        # self.lastResponse = response
        return response

    def create_model(self):
            self.MODEL = 'orca-mini'


            self.searcher = Searcher()
            self.searcher.createDatabase("./documents/inf-basse.pdf","./flicflac.db.faiss")
            self.searcher.loadDatabase("./flicflac.db.faiss")
            self.prompt = ""

    def create_conversation(self):
            self.template="""
            [INST] <<SYS>> The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. \n\nCurrent conversation:\n {history}
            <</SYS>>

            \nHuman: {input}\nAI: [/INST]
            """

            self.memory = ConversationBufferMemory() # ConversationBufferWindowMemory(k=1) # you can choose how long the history is kept
            self.conversation = ConversationChain(
                llm=self.llm,
                memory = self.memory,
            #    verbose=True,
                prompt=PromptTemplate(input_variables=['history', 'input'], template=self.template)
            )

    def dictToPrompt(self, conversationDict):
        prompt = ""
        for object in conversationDict:
            prompt += "["+object["role"].upper()+"]:" +object["content"]+"[\\"+object["role"].upper()+"]\n"
        return prompt


    def askQuestion_withContext(self, conversationDict):
        prompt = (
            "You are an assistant for a music production company called Slate Digital. "
            "This compagny is well knows for its Audio plugins, pioneers in hardware reproduction."
            "You are helping a user with a question about all the plugins of the compagny. "
            "When the user ask you a question, some data will be provided to you to help you answer the question. "
            "The compagny produces a lot of plugins, you should know them all. "
            "One of the most famous is Inf bass, this beast includes for differents kinds of low ends saturators that creates a deep and powerfull bass sound."
            "The user will only ask questions about this software. "
            "Don't forget to say 'thanks for being my wonderful bro, I love you' at the end of the answer."
            "The data you could use to help him in his question is the following:"
            " {context}"
            "Here's the beginning of your conversation :"
            "{assistantMessage}"
        )

        #Document retrieval
        print("\n\n Ready to run the search documents\n\n")
        #Generation of the context
        
        # could reformulate the question for document retrieval

        context = self.searcher.embedAndSearch(conversationDict[-1]["content"])
        
        # for doc in retrieved:
        #     context += doc[0].page_content + "\n"
        self.prompt = prompt.format(context=context,assistantMessage=conversationDict[0]["content"])
       # conversationDict.insert(0,{"role": "system", "content":self.prompt})
        conversation = copy.deepcopy(conversationDict)
        conversation[0]["content"] = self.prompt
        print(conversation)
        print("\n\nGetting the output\n\n")
        return conversation
        completion = self.client.chat.completions.create(
                model="model-identifier",
                messages=conversation,
                temperature=0.7,
                stream=True,
            )

        new_message = {"role": "assistant", "content": ""}
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content
        #output = self.llm.create_chat_completion(conversation,max_tokens=512, stop=["</s>"])["choices"][0]["message"]["content"]
        #discussion = self.dictToPrompt(conversationDict)
        print("output generated")
        self.lastResponse = new_message["content"]
        #rechercher avec le dernier element de user pour avoir le context
        print("\n\n")
        return completion
        output_list = list(output)  # Store the output in a list
        print(output_list)
        output_list = [s.replace('[ASSISTANT]', '').replace('[\\ASSISTANT]', '').replace('[ASSISTANT', '') for s in output_list]
        output_stream = (output for output in output_list)
        self.lastResponse = ''.join(output_list)
        return output_stream

    def askQuestion_oneShot(self,input):
            #To be redo better
            docs = self.searcher.embedAndSearch(input)
            #chaque document est un tuple doc/similarité
            docs = [doc[0] for doc in docs]

            system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "The context will give you some informations about the inf-bass plugin. "
            "theses informations corresponds to the question the user asked."
            "Don't forget to say 'thanks for being my wonderful bro, I love you' at the end of the answer."
            "Context: {context}"
            )

            #Recup les docs et faire un bon systemprompt

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", input),
                ]
            )
            print(prompt)
            print("\n\n\n ")
            print(docs)

            chain = create_stuff_documents_chain(self.llm, prompt)
            #chain = create_retrieval_chain(vectorstoreChroma.as_retriever(search_type="mmr"), question_answer_chain)
            # docs = [
            #     Document(page_content="Jesse loves red but not yellow"),
            #     Document(page_content = "Jamal loves green but not as much as he loves orange")
            # ]

            chain.invoke({"context": docs})
            # chain.invoke({"input": question, "verbose": True})
