from utils.mail_reader import load_mails_from_folder
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import os

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

mails = load_mails_from_folder()
if not mails:
    print("Aucun mail trouvé.")
    exit()

selected_mail = mails[0]  # ou utiliser un input() pour choisir
mail_input = selected_mail["content"]
print(f"Analyse du mail : {selected_mail['subject']} ({selected_mail['from']})")


docs = retriever.get_relevant_documents(mail_input)

oai = Ollama(model="mistral:latest", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
qa_chain = load_qa_with_sources_chain(oai, chain_type="stuff")
result = qa_chain({"input_documents": docs, "question": mail_input}, return_only_outputs=True)

print("\n===== Réponse générée =====\n" + result["output_text"])
