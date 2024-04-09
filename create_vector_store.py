from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> list:
    """Loads in a PDF using the langchain PyPDFLoader

    Args:
        file_path (str): path to PDF file

    Returns:
        list: of langchain documents where one document == one page of the pdf
    """
    loader = PyPDFLoader(file_path)
    return loader.load()


def preprocess(documents: list) -> list:
    """Splits a documet into smaller chunks to fit into a model's context window

    Args:
        documents (list): contains langchain documents that are not yet chunked

    Returns:
        list: langchain documents that are chunked to be smaller than the documents passed in to the method
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=300,
        chunk_overlap=60,  # not sure what the exact values should be here, or even if it will matter. this site is nice for experimenting: https://langchain-text-splitter.streamlit.app
        length_function=len,
    )
    chunks = splitter.create_documents(
        [document.page_content for document in documents],
        metadatas=[document.metadata for document in documents],
    )
    return chunks


def embed_sentences(chunks: list) -> list:
    """Embeds sentences using HuggingFace embedding model

    Args:
        chunks (list): contains langchain documents that have been chunked

    Returns:
        list: a list of lists of floats
    """
    embedder = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )  # model_name from https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers/
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.embed_documents(texts)
    return embeddings


if __name__ == "__main__":
    print("Loading PDF...")
    documents = load_pdf("documents/2023-rulebook_final.pdf")
    print("Loaded PDF.\n")
    print("Preprocessing...")
    chunks = preprocess(documents)
    print("Preprocessed.\n")
    print("Embedding sentences...")
    embedded_sentences = embed_sentences(chunks)
    print("Embedded sentences.\n")
