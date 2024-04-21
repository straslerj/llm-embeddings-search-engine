from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline as hf_pipeline
from wandb.integration.langchain import WandbTracer
import os

DOCUMENT_PATH = "documents/2023-rulebook_final.pdf"
FAISS_INDEX_PATH = "FAISS_index"

TEMPLATE = """
<|SYSTEM|>
You are an expert on professional sports rules.
<|USER|>

Please answer the following question using the context provided. If you don't know the answer, just say that you don't know. Base your answer on the context below. Say "I don't know" if the answer does not appear to be in the context below. 

QUESTION: {question} 
CONTEXT: 
{context}

ANSWER: <|ASSISTANT|>
"""
PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])


class LoadDocs:
    """
    Represents a class for loading documents and preprocessing them.

    Methods:
        __init__: Initializes the LoadDocs object.
        preprocess: Splits a document into smaller chunks to fit into a model's context window.
        embed_sentences: Embeds sentences using HuggingFace embedding model.
    """

    def __init__(self):
        """
        Initializes the LoadDocs object.
        """
        self.documents = PyPDFLoader(DOCUMENT_PATH).load()
        self.chunks = self.preprocess(self.documents)

    def preprocess(self, documents):
        """
        Splits a document into smaller chunks to fit into a model's context window.

        Args:
            documents (list): Contains langchain documents that are not yet chunked.

        Returns:
            list: Langchain documents that are chunked to be smaller than the documents passed in to the method.
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=750,
            chunk_overlap=200,  # Experiment with values, see https://langchain-text-splitter.streamlit.app
            length_function=len,  # I increased this to provide more context to LLM, but hit a limit for gpt2 so this is a sweet spot for the gpt2 model
        )
        chunks = splitter.create_documents(
            [document.page_content for document in documents],
            metadatas=[document.metadata for document in documents],
        )
        return chunks

    ## so far, i do not think we need to include this instance method. Doing the embeddings a little differently
    def embed_sentences(self, chunks):
        """
        Embeds sentences using HuggingFace embedding model.

        Args:
            chunks (list): Contains langchain documents that have been chunked.

        Returns:
            list: A list of lists of floats.
        """
        embedder = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )  # Model name from https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers/
        texts = [doc.page_content for doc in chunks]
        embeddings = embedder.embed_documents(texts)
        return embeddings


class DocChat:
    """
    Represents a class for handling document chat.

    Methods:
        __init__: Initializes the DocChat object. When initialized, will store the FAISS embeddings in file FAISS_index if D.N.E.
        qNa: Executes a question and answer process based on the provided query.
    """

    def __init__(self, model_id):
        """
        Initializes the DocChat object.
        """
        self.chunks = LoadDocs().chunks
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 100},
        )

        cwd = os.getcwd()
        file_path = os.path.join(cwd, FAISS_INDEX_PATH)

        if os.path.exists(FAISS_INDEX_PATH):
            self.vector_db = FAISS.load_local(
                FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            self.vector_db = FAISS.from_documents(self.chunks, self.embeddings)
            self.vector_db.save_local("FAISS_index")

        self.chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=PROMPT)

    def qNa(self, query, just_answer=False):
        """
        Executes a question and answer process based on the provided query.

        Args:
            query (str): The query for the question and answer process.

        Returns:
            str: The output text of the question and answer process.
        """
        sim_search_res = self.vector_db.similarity_search(query)
        res = self.chain({"input_documents": sim_search_res, "question": query})
        output_text = res["output_text"]
        if just_answer:
            return output_text.split("ANSWER:")[1]
        else:
            return output_text


# from transformers import GPT2Tokenizer -- me playing around with gpt2 tokenizer to find sweet spot. Could implement error
# capturing with code like this but I think that is out of our scope for now with the time we have

# # Load the GPT-2 tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# input_text = 'PATRICK - If a non-'
# # Tokenize input text
# tokens = tokenizer.encode(input_text, return_tensors="pt")
# print(len(tokens[0]))
