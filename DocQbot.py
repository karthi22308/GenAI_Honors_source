import streamlit as st
from PyPDF2 import PdfReader
from openai import AzureOpenAI

# Function to extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to generate answer using Azure OpenAI
def generate_answer(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    client = AzureOpenAI(
        api_version='2024-06-01',
        azure_endpoint='https://hexavarsity-secureapi.azurewebsites.net/api/azureai',
        api_key='922892c42af122a9'
    )
    
    res = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with your actual model name
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7,
        max_tokens=256,
        top_p=0.6,
        frequency_penalty=0.7
    )
    
    return res.choices[0].message.content

# Streamlit UI
st.title("DocQBot - Automated Document Retrieval")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    document_text = extract_text_from_pdf(uploaded_file)
    st.success("Document processed successfully!")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if uploaded_file is None:
        st.error("Please upload a document first.")
    elif question.strip() == "":
        st.error("Please enter a question.")
    else:
        st.write("Generating answer...")
        answer = generate_answer(question, document_text)
        st.subheader("Answer:")
        st.write(answer)
