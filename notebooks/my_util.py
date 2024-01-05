import os

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer, util

import torch
from torch import Tensor, device

from typing import List

from tqdm.auto import tqdm
from tqdm.autonotebook import trange


def get_chunks(company_name: str):
    file_folder_path = f"pdf/{company_name}"
    doc = [os.path.join(file_folder_path, file) for file in os.listdir(file_folder_path)][0]  # Only the first doc

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True
    )

    loader = UnstructuredFileLoader(doc, mode="single", strategy="fast")
    chunks = loader.load_and_split(text_splitter)

    content = [f"Company: {company_name}. " + chunk.page_content for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]

    return content, metadata


def show_tokens(sentence: str, tokenizer):
    
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

    print("Number of tokens:", len(inputs.input_ids[0]))
        
    for input_id in inputs.input_ids[0]:
        print(input_id, "->", tokenizer.decode(input_id))


def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0]


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def get_batched_embeddings(sentences: List[str], batch_size: int, tokenizer, model, padding=True, device="cpu"):
    
    all_embeddings = []
    once = False
    
    for start_index in trange(0, len(sentences), batch_size):
        # 1. Tokenize sentences
        batch = sentences[start_index:start_index+batch_size]
        encoded_input = tokenizer(batch, padding=padding, truncation=True, return_tensors='pt')
        
        encoded_input = batch_to_device(encoded_input, device)
        
        with torch.no_grad():
            # 2. Compute token embeddings -> Same toke might have different embeddings due to context.
            model_output = model(**encoded_input)
            if once is False:
                print("Shape of embedded tokens:", model_output.last_hidden_state.shape)
                once = True

            # 3. Perform pooling
            # Option 1: Mean pooling
            # embeddings = mean_pooling(model_output, encoded_input.attention_mask)
            
            # Option 2: CLS pooling
            embeddings = cls_pooling(model_output)
            
            embeddings = embeddings.detach()
            all_embeddings.extend(embeddings)
    
    all_embeddings = torch.stack(all_embeddings)
    
    return all_embeddings


def get_topk_similarity(k, encoded_query, encoded_docs, is_cos_sim, debug):

    if is_cos_sim:
        # Compute cosine similarity between query and all document embeddings
        cos_sim = util.cos_sim(encoded_query, encoded_docs)
        
        # Combine docs & scores
        doc_idx_score_pairs = list(zip(range(len(encoded_docs)), cos_sim[0].tolist()))  # The first query

    else:
        # Compute dot score between query and all document embeddings
        scores = util.dot_score(encoded_query, encoded_docs)[0].tolist()

        # Combine docs & scores
        doc_idx_score_pairs = list(zip(range(len(encoded_docs)), scores))

    
    # Sort by decreasing score
    doc_idx_score_pairs = sorted(doc_idx_score_pairs, key=lambda x: x[1], reverse=True)

    if debug:
        print(f"Most similar pairs:\ndoc_idx\t score")
        for doc_idx, score in doc_idx_score_pairs[:k]:
            print(f"{doc_idx} \t {score:.4f}")
    else:
        return doc_idx_score_pairs[:k]