from abc import ABC
from operator import itemgetter
import os

from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings
from src.retrievers.json_reader import PCJSONReader

############## Save Chunk Embeddings Logic ##############
import json
SAVE_EMB_DIR = "./doc_embeddings"
def save_vector_retriever_embeddings(name, retriever):
    os.makedirs(SAVE_EMB_DIR, exist_ok=True)

    data = []

    vector_store = retriever._index._vector_store
    docstore = retriever._docstore

    # embeddings live HERE:
    for node_id, embedding in vector_store._data.embedding_dict.items():
        node = docstore.get_document(node_id)

        data.append({
            "node_id": node_id,
            "text": node.get_content(),
            "metadata": node.metadata,
            "embedding": embedding
        })

    out_path = os.path.join(SAVE_EMB_DIR, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[EMBEDDINGS SAVED] {out_path}")
########################################################

############## Load Embeddings for BGE ##############
def load_vector_retriever_from_json(json_path, embed_model, similarity_top_k=2):
    """
    Load precomputed embeddings from a JSON file into a VectorIndexRetriever.
    """
    import json
    from llama_index import StorageContext, ServiceContext, VectorStoreIndex
    from llama_index.embeddings import LangchainEmbedding
    from llama_index.storage.docstore import SimpleDocumentStore
    from llama_index.schema import TextNode

    print(f"[LOADING EMBEDDINGS] {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    nodes = []
    for entry in data:
        node = TextNode(
            text=entry["text"],
            id_=entry["node_id"],
            metadata=entry["metadata"]
        )
        node.embedding = entry["embedding"]
        nodes.append(node)

    # Build index from precomputed embeddings
    embed_model_instance = LangchainEmbedding(embed_model)
    service_context = ServiceContext.from_defaults(embed_model=embed_model_instance, llm=None)

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    index = VectorStoreIndex(
        nodes,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=False
    )

    return VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
########################################################

def construct_retriever(docs_directory, embed_model=None, chunk_size=128, chunk_overlap=0, similarity_top_k=2):
    documents = SimpleDirectoryReader(docs_directory, file_extractor={".json": PCJSONReader()}, recursive=True).load_data(num_workers=8)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True, num_workers=8)
    
    if embed_model:
        embed_model_instance = LangchainEmbedding(embed_model)
        service_context = ServiceContext.from_defaults(embed_model=embed_model_instance, llm=None)
        doc_store = SimpleDocumentStore()
        doc_store.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=doc_store)
        vector_index = VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context, show_progress=True)
        
        retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=similarity_top_k)
        save_vector_retriever_embeddings(os.path.basename(docs_directory), retriever)        
        return retriever
    else:
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)

class CustomBGEM3Retriever(ABC):
    def __init__(self, docs_directory, embed_model, embed_dim=768, chunk_size=128, chunk_overlap=0, similarity_top_k=2, emb_path=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.emb_path = emb_path
        self.retrievers = self.construct_retrievers(docs_directory, embed_model)

    def construct_retrievers(self, docs_directory, embed_model):
        # ===== CASE 1: Precomputed embeddings directory =====
        if self.emb_path is not None:
            if os.path.isdir(self.emb_path):
                print(f"[INFO] Loading embedding directory: {self.emb_path}")

                retrievers = {}
                for file in os.listdir(self.emb_path):
                    if file.endswith(".json"):
                        subname = file.replace(".json", "")
                        retrievers[subname] = load_vector_retriever_from_json(
                            os.path.join(self.emb_path, file),
                            embed_model,
                            similarity_top_k=self.similarity_top_k
                        )

                print("[INFO] Loaded all precomputed embedding files.")
                return retrievers

            else:
                raise ValueError(
                    f"[ERROR] --emb_path was provided but is not a directory: {self.emb_path}"
                )

        # ===== CASE 2: Normal full embedding pipeline =====
        retrievers = {}
        is_subdir_present = False

        for subdir in os.listdir(docs_directory):
            subdir_path = os.path.join(docs_directory, subdir)
            if os.path.isdir(subdir_path):
                is_subdir_present = True
                retrievers[subdir] = construct_retriever(
                    subdir_path,
                    embed_model,
                    self.chunk_size,
                    self.chunk_overlap,
                    self.similarity_top_k
                )

        if not is_subdir_present:
            retrievers["default"] = construct_retriever(
                docs_directory,
                embed_model,
                self.chunk_size,
                self.chunk_overlap,
                self.similarity_top_k
            )

        print("Indexing finished for all directories!")
        return retrievers


    def search_docs(self, query: dict):
        doc_name = query.get("doc_name")
        query_text = query.get("questions")
        subdir = doc_name.split('/')[0]
        retriever = self.retrievers.get(subdir, self.retrievers.get('default'))
        if not retriever:
            raise ValueError(f"No retriever found for directory: {subdir}")
        response_nodes = retriever.retrieve(query_text)
        return [{
            "text": node.get_content(),
            "page_idx": node.metadata.get("page_idx", None),
            "file_name": node.metadata.get("file_name", "").replace(".json", "")
        } for node in response_nodes]

class CustomBM25Retriever(ABC):
    def __init__(self, docs_directory, chunk_size=128, chunk_overlap=0, similarity_top_k=2):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.retrievers = self.construct_retrievers(docs_directory)

    def construct_retrievers(self, docs_directory):
        retrievers = {}
        is_subdir_present = False
        for subdir in os.listdir(docs_directory):
            subdir_path = os.path.join(docs_directory, subdir)
            if os.path.isdir(subdir_path):
                is_subdir_present = True
                retrievers[subdir] = construct_retriever(subdir_path, None, self.chunk_size, self.chunk_overlap, self.similarity_top_k)
        if not is_subdir_present:
            retrievers['default'] = construct_retriever(docs_directory, None, self.chunk_size, self.chunk_overlap, self.similarity_top_k)
        
        print("Indexing finished for all directories!")
        return retrievers

    def search_docs(self, query: dict):
        doc_name = query.get("doc_name")
        query_text = query.get("questions")
        subdir = doc_name.split('/')[0]
        retriever = self.retrievers.get(subdir, self.retrievers.get('default'))
        if not retriever:
            raise ValueError(f"No retriever found for directory: {subdir}")
        response_nodes = retriever.retrieve(query_text)
        return [{
            "text": node.get_content(),
            "page_idx": node.metadata.get("page_idx", None),
            "file_name": node.metadata.get("file_name", "").replace(".json", "")
        } for node in response_nodes]

class CustomHybridRetriever(ABC):
    def __init__(self, docs_directory, embed_model, embed_dim=768,
                 chunk_size=128, chunk_overlap=0, similarity_top_k=2,
                 emb_path=None):

        self.weights = [0.5, 0.5]          # RRF weights: [BM25, BGE]
        self.c: int = 60                  # RRF constant
        self.top_k = similarity_top_k
        self.similarity_top_k = similarity_top_k 
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.emb_path = emb_path
        self.embed_model = embed_model
        self.collection_name = "hybrid"
        self.retrievers = self.construct_retrievers(docs_directory, embed_model)

    def construct_retrievers(self, docs_directory, embed_model):
        # ===== CASE 1: Precomputed embedding directory =====
        if self.emb_path is not None and os.path.isdir(self.emb_path):
            print(f"[HYBRID] Loading precomputed embedding directory: {self.emb_path}")

            retrievers = {}

            for file in os.listdir(self.emb_path):
                if not file.endswith(".json"):
                    continue

                subname = file.replace(".json", "")
                json_path = os.path.join(self.emb_path, file)

                # Load embedding retriever (BGE)
                embedding_retriever = load_vector_retriever_from_json(
                    json_path,
                    self.embed_model,
                    similarity_top_k=self.top_k
                )

                # Load BM25 retriever for the SAME SUBDIRECTORY
                subdir_path = os.path.join(docs_directory, subname)
                if not os.path.isdir(subdir_path):
                    raise RuntimeError(f"[HYBRID ERROR] Expected matching directory: {subdir_path}")

                bm25_retriever = self._construct_bm25(subdir_path)

                retrievers[subname] = (embedding_retriever, bm25_retriever)

            print("[HYBRID] Loaded all precomputed embedding files.")
            return retrievers

        # ===== CASE 2: No precomputed embeddings → normal pipeline =====
        retrievers = {}
        is_subdir_present = False

        for subdir in os.listdir(docs_directory):
            subdir_path = os.path.join(docs_directory, subdir)

            if os.path.isdir(subdir_path):
                is_subdir_present = True
                retrievers[subdir] = self._construct_full_pair(subdir_path)

        if not is_subdir_present:
            retrievers["default"] = self._construct_full_pair(docs_directory)

        print("[HYBRID] Finished constructing new hybrid indexes.")
        return retrievers

    def _construct_bm25(self, path: str):
        """Build only the BM25 retriever for a given directory."""
        documents = SimpleDirectoryReader(
            path,
            file_extractor={".json": PCJSONReader()},
            recursive=True
        ).load_data()
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(
            documents,
            show_progress=False
        )
        return BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self.top_k
        )

    def _construct_full_pair(self, path: str):
        """Build both embedding retriever + BM25 retriever from raw docs."""
        documents = SimpleDirectoryReader(
            path,
            file_extractor={".json": PCJSONReader()},
            recursive=True
        ).load_data()
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(
            documents,
            show_progress=True
        )

        embed_model_instance = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=embed_model_instance,
            llm=None
        )
        doc_store = SimpleDocumentStore()
        doc_store.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=doc_store)

        vector_index = VectorStoreIndex(
            nodes,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=True
        )
        embedding_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self.top_k
        )
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self.top_k
        )

        return (embedding_retriever, bm25_retriever)

    def search_docs(self, query: dict):
        doc_name = query.get("doc_name")
        query_text = query.get("questions")
        subdir = doc_name.split('/')[0]

        retriever_pair = self.retrievers.get(subdir, self.retrievers.get('default'))
        if not retriever_pair:
            raise ValueError(f"No retrievers found for directory: {subdir}")

        embedding_retriever, bm25_retriever = retriever_pair

        bm25_search_docs = bm25_retriever.retrieve(query_text)
        embedding_search_docs = embedding_retriever.retrieve(query_text)

        doc_lists = [bm25_search_docs, embedding_search_docs]

        # RRF fusion
        all_documents = {}

        # Collect unique documents by node id
        for doc_list in doc_lists:
            for d in doc_list:
                nid = d.node.node_id
                all_documents[nid] = d

        # Compute scores
        rrf_score_dic = {nid: 0.0 for nid in all_documents}

        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, d in enumerate(doc_list, start=1):
                nid = d.node.node_id
                rrf_score_dic[nid] += weight * (1 / (rank + self.c))

        # Sort
        sorted_docs = sorted(rrf_score_dic.items(), key=itemgetter(1), reverse=True)

        # Retrieve Node objects
        top_docs = [all_documents[nid] for nid, _ in sorted_docs[:self.top_k]]

        # Format output
        return [{
            "text": d.node.get_content(),
            "page_idx": d.node.metadata.get("page_idx"),
            "file_name": d.node.metadata.get("file_name", "").replace(".json", "")
        } for d in top_docs]




class CustomPageRetriever(ABC):
    def __init__(self, docs_directory: str):
        self.documents = self.construct_index(docs_directory)

    def construct_index(self, docs_directory):
        documents_by_subdir = {}
        is_subdir_present = False
        for subdir in os.listdir(docs_directory):
            subdir_path = os.path.join(docs_directory, subdir)
            if os.path.isdir(subdir_path):
                is_subdir_present = True
                documents = SimpleDirectoryReader(subdir_path, file_extractor={".json": PCJSONReader()}, recursive=True).load_data(num_workers=8)
                documents_by_subdir[subdir] = documents
        if not is_subdir_present:
            documents = SimpleDirectoryReader(docs_directory, file_extractor={".json": PCJSONReader()}, recursive=True).load_data(num_workers=8)
            documents_by_subdir['default'] = documents
        return documents_by_subdir

    def search_docs(self, query: dict):
        doc_name = query.get("doc_name")
        evidence_page_no = query.get("evidence_page_no")
        if not isinstance(evidence_page_no, list):
            evidence_page_no = [evidence_page_no]
        subdir = doc_name.split('/')[0]
        documents = self.documents.get(subdir, self.documents.get('default'))

        def get_doc_name(metadata: dict):
            return os.path.basename(os.path.dirname(metadata.get("file_path", ""))) + "/" + metadata.get("file_name", "").replace(".json", "")

        if not documents:
            raise ValueError(f"No documents found for directory: {subdir}")

        results = [
            {
                "text": doc.text,
                "page_idx": doc.metadata.get("page_idx", None),
                "file_name": doc.metadata.get("file_name", "").replace(".json", "")
            }
            for doc in documents
            if get_doc_name(doc.metadata) == doc_name and doc.metadata.get("page_idx") in evidence_page_no
        ]

        return results
