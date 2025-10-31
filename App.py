import streamlit as st
from PyPDF2 import PdfReader
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve sensitive credentials securely
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")


# ---------------- PDF Processing ---------------- #
def extract_text_from_pdf(uploaded_file):
    """
    Extracts all text content from a given PDF file.
    :param uploaded_file: Streamlit uploaded file object
    :return: Extracted text as a string
    """
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text.strip()
    except Exception as e:
        st.error(f"Error while reading PDF: {e}")
        return ""


# ---------------- Azure OpenAI Interaction ---------------- #
def generate_answer(question, context):
    """
    Sends a question and context to Azure OpenAI to generate an answer.
    :param question: User's question as a string
    :param context: Extracted document text as context
    :return: Generated answer string
    """
    if not AZURE_API_KEY or not AZURE_ENDPOINT:
        st.error("Azure OpenAI credentials are missing in the .env file.")
        return None

    config = 'you are a bot that answers questions from the document'
    prompt = f"\n{config} document:\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION
        )

        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
            top_p=0.6,
            frequency_penalty=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error generating answer from Azure OpenAI: {e}")
        return None


# ---------------- Streamlit UI ---------------- #
def main():
    st.title("📘 DocQBot - Automated Document Retrieval")

    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    document_text = ""
    if uploaded_file is not None:
        document_text = extract_text_from_pdf(uploaded_file)
        if document_text:
            st.success("✅ Document processed successfully!")
        else:
            st.warning("⚠️ Could not extract text from the uploaded PDF.")

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if uploaded_file is None:
            st.error("Please upload a document first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            st.info("Generating answer... Please wait.")
            answer = generate_answer(question, document_text)
            if answer:
                st.subheader("💬 Answer:")
                st.write(answer)
            else:
                st.error("Failed to generate an answer.")


if __name__ == "__main__":
    main()
