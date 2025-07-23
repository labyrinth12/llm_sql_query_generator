import streamlit as st
import requests
API_URL = "https://lmsqlquerygenerator-production.up.railway.app/generate"
  # Your FastAPI endpoint

st.set_page_config(page_title="Gemini SQL Generator", layout="wide")

st.title("🧠 Gemini-powered SQL Generator")

st.markdown("Type your question about the database:")

question = st.text_input("Question", placeholder="e.g., Show top 5 cement brands by sales")

if st.button("Generate SQL & Run Query"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL & querying database..."):
            try:
                response = requests.post(API_URL, json={"question": question})
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("📜 Generated SQL")
                    st.code(data['sql'], language="sql")

                    st.subheader("📊 Query Result")
                    if data["result"]:
                        st.dataframe(data["result"])
                    else:
                        st.info("No records found.")
                else:
                    st.error(f"❌ Error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"💥 Failed to connect to backend: {e}")
