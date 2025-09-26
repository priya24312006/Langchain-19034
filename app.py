import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from pypdf import PdfReader

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="Personal Diary Chatbot")
st.title("Personal Diary Chatbot")
st.write("Upload your diary (txt/pdf) and ask questions like: 'What did I write about exams last week?'")

uploaded_files = st.file_uploader(
    "Upload your diary files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        if file.name.endswith(".txt"):
            all_text += file.read().decode("utf-8") + "\n"
        elif file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

    if all_text.strip() == "":
        st.warning("No text could be extracted. Make sure your PDF is text-based, not scanned images.")
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(all_text)

        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever()
        st.success("Diary uploaded and ready for questions")

        llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
        llm = HuggingFacePipeline(pipeline=llm_pipeline)

        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        query = st.text_input("Ask something about your diary:")
        if query:
            answer = qa.run(query)
            st.subheader("Answer")
            st.write(answer)
