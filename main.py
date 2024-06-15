from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain.llms import Ollama
from langchain_community.chat_models import ChatOllama
# from langchain_community import invoke
from pprint import pprint
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from model import Model

if __name__ == "__main__":

        

        print("welcome to the agent")
        #Create the model
        model = Model()
        model.create_model()
        model.askQuestion_oneShot("Could you tell me what is the Inf Bass plugin ?")

