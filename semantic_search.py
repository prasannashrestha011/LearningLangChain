"""
Here PDF content will be extracted in contextually meaniningful chunks without cutting off important points, and making context clear.

chunk_size:1000->Each chunk will have approximately 1000 characters.
chunk_overlap:200-> Next chunk will overlap previous chunk with 200 characters, so important contenxt won't be cutoff.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
file_path="./assets/UNIT-2.pdf"

loader=PyPDFLoader(file_path)

docs=loader.load()

text_splitters=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
        )
all_splits=text_splitters.split_documents(docs)
for split in all_splits:
    print(split)
