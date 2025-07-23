import streamlit as st
import requests

# URL of your deployed FastAPI app
API_URL = "https://lmsqlquerygenerator-production.up.railway.app/generate"


st.title("💬 Natural Language to SQL (Gemini)")

question = st.text_input("Ask a question in plain English:")

if st.button("Generate SQL"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL..."):
            try:
                response = requests.post(API_URL, json={"question": question})
                data = response.json()
                if response.status_code == 200:
                    st.subheader("📄 Generated SQL")
                    st.code(data["sql"], language="sql")

                    st.subheader("🧾 Query Results")
                    st.write(data["result"])
                else:
                    st.error(f"Error: {data.get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Request failed: {e}")
