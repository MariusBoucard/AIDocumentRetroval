"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import random
import time
from model import Model
# Streamed response emulator

#init du model
model = Model()
model.create_model()



first = ""


st.title("Slate Digital Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    
    with st.chat_message("user"):
        st.markdown(prompt)
    # st.write(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

response = f"Echo: {prompt}"
# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = ""
    if len(st.session_state.messages) == 0:
        response = model.response_generator()  # Call the function and store its result
        st.write(response)  # Display the response
       # response= st.write(response_generator())
        response = model.lastResponse
        print(first)
    else:
       response = st.write(model.askQuestion_withContext(st.session_state.messages))
       response = model.lastResponse

    #Brancher le model ici
    #Creation de la bonne input à lui envoyer a partir du s
# Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    #model.askQuestion_withContext(st.session_state.messages)

st.session_state.messages


