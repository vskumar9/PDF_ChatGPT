import dotenv
from django.shortcuts import render
from PyPDF2 import PdfReader
import os
import pickle
from django.conf import settings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

load_dotenv()


def pdf_chat(request):
    if request.method == 'POST' and request.FILES.get('upload_pdf'):
        uploaded_pdf = request.FILES['upload_pdf']

        pdf_reader = PdfReader(uploaded_pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Process text and embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = uploaded_pdf.name[:-4]

        try:
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
        except Exception as e:
            return render(request, 'upload_pdf.html', {'error': e})

        if request.POST.get('query'):
            query = request.POST.get('query')
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)

            context = {
                'response': response
            }

            return render(request, 'pdf_chat.html', context)

    return render(request, 'upload_pdf.html')
    # return render(request, 'pdf_chat.html')
