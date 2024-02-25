from langchain_community.embeddings import SentenceTransformerEmbeddings ## get text embeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader #cargar los documentos
from langchain.text_splitter import RecursiveCharacterTextSplitter # dividir los documentos en textos

from langchain_community.vectorstores import Qdrant ## guardar los vectores

from config import URL_VECTOR_DATABASE

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

embedding=SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")


## load document
loader=DirectoryLoader("data_folder",
                       glob="**/*.pdf", 
                       show_progress=True,
                       loader_cls=UnstructuredFileLoader ## loader class
                       )

documents=loader.load()

## chunk document in texts

text_splitter=RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=40)

texts=text_splitter.split_documents(documents=documents)

## save into database (w)

qdrant=Qdrant.from_documents(
    texts, # cuales textos quiero convertir en embeddings
    embedding, # qué motor de embeddings utilizaré
    url=URL_VECTOR_DATABASE, ## direccion donde se está ejecutando el vector database
    prefer_grpc=False, # protocolo de conexión creada por google,
    collection_name="vector_db_rag"

)
local_llm = "model/BioMistral-7B-DARE-Q4_K_M.gguf"
# Make sure the model path is correct for your system!
## instalar %pip install --upgrade --quiet  llama-cpp-python
# instalar xcode en mac m1: xcode-select --install


#https://huggingface.co/LoneStriker/BioMistral-7B-DARE-GGUF/blob/main/BioMistral-7B-DARE-Q4_K_M.gguf
llm = LlamaCpp(
    model_path= local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1
)
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
retriever = qdrant.as_retriever(search_kwargs={"k":1})
chain_type_kwargs = {"prompt": prompt}
query=input("ingresa tu pregunta:")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
response = qa(query)

print(response)


