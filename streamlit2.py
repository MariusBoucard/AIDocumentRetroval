import streamlit as st
from openai import OpenAI
from model import Model
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = Model()
model.create_model()

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

def store_and_yield(responses, storage):
    for response in responses:
        msg = response.choices[0].delta.content
        if msg is not None:
            storage.append(msg)
            yield msg

st.title("💬 Slate Digital Produceur AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content" : model.response_generator()}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
   # query = model.reformulate_query(prompt)
    responses = client.chat.completions.create(
                model="model-identifier",
                messages=model.askQuestion_withContext(list(st.session_state.messages),prompt),
                temperature=0.7,
                stream=True)
    
    # full_response = ""
    # for response in responses:
    #     msg = response.choices[0].delta.content
    #     if msg == None:
    #         break
    #     full_response += msg
    stored_response = []
    responseStream = store_and_yield(responses,stored_response)
    st.chat_message("assistant").write(responseStream)
    full_response = "".join(stored_response)
    print(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    