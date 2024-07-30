import streamlit as st
from main import ChatBot


st.set_page_config(page_title="Business Strategy FGV EBAPE")
with st.sidebar:
    st.title('Business Strategy FGV EBAPE')


## Function for generating LLM response
def generate_response(input):
    result = ChatBot(input = input)
    return result

## Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{
        "role"   : "assistant",
        "content": "Welcome, this is a chatbot developed to discuss strategy for your academic research"
    }]

## Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

## User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input, "sources": []})
    with st.chat_message("user"):
        st.write(input)

## Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from our AI..."):
            response, sources = generate_response(input)
            st.write(response)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"- {source}")
    message = {"role": "assistant", "content": response, "sources": sources}
    st.session_state.messages.append(message)


