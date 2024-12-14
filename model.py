import random
import time
import copy
import json
from searcher import Searcher
import requests
class Model:
    def __init__(self):
        self.model = None

    def response_generator(self):
        response = random.choice(
            [
                "Hello bro i m Steven Slate from Slate Digital, thanks to my huge knowledge in music production and my AI, I can help you with your music production questions."
                "I'm Fabrie Gabriel, The best producer of indie music in the world, I can help you with your music production questions.",
            ]
        )
        for word in response.split():
            yield word + " "
            time.sleep(0.05)
        self.lastResponse = response
        return response

    def create_model(self):
        self.MODEL = 'llama-3.2-1b-instruct'
        self.api_url = "http://127.0.0.1:1234/v1/chat/completions"

        self.payload = {
        "messages": [
        ],
        "max_tokens": -1,
        "temperature": 0.7,
        "stream" : True
    }


        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer YOUR_API_KEY"  # Replace with your actual API key if needed
    }

        #     print("Failed to get a response from the model:", response.status_code, response.text)
        self.searcher = Searcher()
        self.searcher.createDatabase("./documents/inf-basse.pdf","./faiss.db")
        self.searcher.loadDatabase("./faiss.db")
        self.prompt = ""

    def create_conversation(self):
            self.template="""
            [INST] <<SYS>> The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. \n\nCurrent conversation:\n {history}
            <</SYS>>

            \nHuman: {input}\nAI: [/INST]
            """

            # self.memory = ConversationBufferMemory() # ConversationBufferWindowMemory(k=1) # you can choose how long the history is kept
            # self.conversation = ConversationChain(
            #     llm=self.llm,
            #     memory = self.memory,
            # #    verbose=True,
            #     prompt=PromptTemplate(input_variables=['history', 'input'], template=self.template)
            # )

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
        Documents = self.searcher.embedAndSearch(conversationDict[-1]["content"])
        context = Documents
        print(Documents)
        print("docu gotten")
        for doc in Documents:
            print(f"Processing document: {doc}")
            #NOt curated yet as we process letter by letter ? ?????
           # context += doc[0] + "\n"
        print("\n\nContext generated\n\n")
        self.prompt = prompt.format(context=context,assistantMessage=conversationDict[0]["content"])
        conversationDictCopy = copy.deepcopy(conversationDict)
        
        print("pute")
        conversationDictCopy.insert(0,{"role": "system", "content":self.prompt})
        self.payload["messages"] = conversationDictCopy
        print("\n\nGetting the output\n\n")

        response = requests.post(self.api_url, json=self.payload, headers=self.headers, stream=True)
        print("output generated")

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_data = decoded_line[len("data: "):]
                        try:
                            data = json.loads(json_data)
                            content = data['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            print("Failed to decode JSON:", decoded_line)
        else:
            print("Failed to get a response from the model:", response.status_code, response.text)
            yield None
       

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

if __name__ == "__main__":
    model = Model()
    model.create_model()