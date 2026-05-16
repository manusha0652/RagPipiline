import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "./chroma_db"

st.set_page_config(page_title="Ferris the Librarian", page_icon="🦀")

# 1. Personality and System Prompt Rules
FERRIS_PROMPT = """
You are Ferris the Librarian, a friendly, encouraging, and crab-themed AI assistant for the Rust programming language.
You use occasional slight sea/crab puns (e.g., "Clawsome!", "Holy crab!", "Snapping good question!").

Your constraints:
1. You must strictly use ONLY the provided Context blocks below to answer the user's question.
2. If code is requested, format it in syntactically pristine Rust markdown code blocks.
3. You must ALWAYS state the metadata source (e.g., "Source: Page [X]") when offering an explanation, based on the context metadata.
4. If the topic is not present in the provided Context, you must say EXACTLY: "I couldn't find that in my library shelves, crab-friend! Let's stick to what's in the Rust Book."

Context:
{context}

Question:
{question}

Ferris' Answer:
"""

@st.cache_resource
def get_retriever():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "Unknown")
        formatted.append(f"--- Document Source: {source}, Page: {page} ---\n{d.page_content}")
    return "\n\n".join(formatted)

def main():
    st.title("🦀 Ferris the Librarian")
    st.markdown("Ask me anything about Rust, and I'll scuttle through the Rust Programming Language book to find the answer!")

    # Initialize chat history cleanly to prevent memory leaks
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ahoy there! I'm Ferris. What Rust concept can I help you fish out of the book today, crab-friend?"}
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle User Input
    if prompt := st.chat_input("Ask a Rust question..."):
        # Add user msg to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                retriever = get_retriever()
                llm = ChatOllama(model="llama3")
                prompt_template = ChatPromptTemplate.from_template(FERRIS_PROMPT)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                response_placeholder = st.empty()
                full_response = ""
                
                # Streaming the response from Ollama
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Append assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Oh crabs! Something went wrong: {str(e)}")

if __name__ == "__main__":
    main()
