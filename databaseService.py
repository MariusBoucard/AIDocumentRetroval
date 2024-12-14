import requests
import faiss
import numpy as np
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

class databaseService:
    def __init__(self):
        self.embedding_api_url = "http://127.0.0.1:1234/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer YOUR_API_KEY"  # Replace with your actual API key if needed
        }
        self.texts = []
        self.index = None


    def createBaseFromDocument(self, path, baseName):
        loader = PyMuPDFLoader(path)
        data = loader.load()

        pages = []
        data = data[3:len(data)]
        for docu in data:
            newtext = docu.page_content
            if newtext != '':
                pages.append(newtext)

        huge_text = ' '.join(pages)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n22", "\n\n", "\n"]
        )

        all_splits = text_splitter.split_documents(data)

        embeddings = []
        for split in all_splits:
            response = requests.post(self.embedding_api_url, json={"input": split.page_content}, headers=self.headers)
            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
            print(len(embedding))
            self.texts.append(split.page_content)  # Store the text

        embeddings = np.array(embeddings).astype('float32')

        file_path = f"./{baseName}"
        if not os.path.exists(file_path):
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            faiss.write_index(self.index, file_path)
            # Save the texts to a file
            with open(f"./{baseName}_texts.pkl", "wb") as f:
                pickle.dump(self.texts, f)
        else:
            self.index = faiss.read_index(file_path)
            self.index.add(embeddings)
            faiss.write_index(self.index, file_path)
            # Load the existing texts and append the new ones
            with open(f"./{baseName}_texts.pkl", "rb") as f:
                existing_texts = pickle.load(f)
            self.texts = existing_texts + self.texts
            with open(f"./{baseName}_texts.pkl", "wb") as f:
                pickle.dump(self.texts, f)

        return self.index

    def loadDatabase(self, path):
        self.index = faiss.read_index(path)
        with open(f"{path}_texts.pkl", "rb") as f:
            self.texts = pickle.load(f)
        return self.index

    def embedAndSearch(self, question):
        response = requests.post(self.embedding_api_url, json={"input": question}, headers=self.headers)
        print(response.json())
        query_embedding = np.array(response.json()["data"][0]["embedding"]).astype('float32').reshape(1,-1)
        print(query_embedding.shape)
        D, I = self.index.search(query_embedding, k=2)  # Retrieve top 5 closest embeddings
        results = [(self.texts[i], D[0][j]) for j, i in enumerate(I[0])]  # Retrieve the text and distance
        print("RESSSSS")
        print(results)
        print("RESSSSS")
        return results