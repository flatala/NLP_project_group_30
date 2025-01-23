"""Code inspired from HfRetriever.py in dexter/retriever/dense/HfRetriever.py.

Dropped support for retrieving in chunks.
"""

import logging
import os
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import torch
from dexter.data.datastructures.evidence import Evidence
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.datastructures.question import Question
from dexter.retriever.BaseRetriever import BaseRetriver
from dexter.utils.metrics.SimilarityMatch import SimilarityMetric
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class AdoreRetriever(BaseRetriver):
    """Adore retriever class.

    Based on the HfRetriever. We use the same document encoder, we only change the query encoder.
    This encoder is trained using the ADORE methodology.
    """

    def __init__(
        self,
        config: DenseHyperParams,
        corpus_folder: str = None,
        corpus_file: str = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.addHandler(logging.FileHandler("AdoreRetriever.log"))
        self.logger.info("Initializing AdoreRetriever...")
        super().__init__()
        self.config: DenseHyperParams = config

        ## SET UP CONTEXT ENCODER ##
        self.context_tokenizer = AutoTokenizer.from_pretrained(
            self.config.document_encoder_path
        )
        self.context_encoder = AutoModel.from_pretrained(
            self.config.document_encoder_path
        )

        ## SET UP QUESTION ENCODER ##
        self.question_tokenizer = AutoTokenizer.from_pretrained(
            self.config.query_encoder_path
        )
        self.question_encoder = AutoModel.from_pretrained(
            self.config.query_encoder_path
        )
        self.question_encoder.cuda()

        self.context_encoder.cuda()
        self.batch_size = self.config.batch_size
        self.sep = "[SEP]"

        self.corpus_folder = corpus_folder
        self.corpus_file = corpus_file
        self.logger.info("AdoreRetriever succesfully initialized...")

    def _load_index_if_available(self) -> Tuple[Any, bool]:
        """COPIED FROM HfRetriever.py"""
        if (
            self.corpus_folder is not None
            and self.corpus_file is not None
            and os.path.exists(os.path.join(self.corpus_folder, self.corpus_file))
        ):
            corpus_embeddings = joblib.load(
                os.path.join(self.corpus_folder, self.corpus_file)
            )
            index_present = True
        else:
            index_present = False
            corpus_embeddings = None
        return corpus_embeddings, index_present

    def _mean_pooling(self, token_embeddings, mask):
        """COPIED FROM HfRetriever.py"""
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def _encode_corpus(
        self, corpus: List[Evidence], **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        """Function to encode the corpus, using the document encoder.

        COPIED FROM HfRetriever.py

        Args:
            corpus (List[Evidence]): List of evidence objects.

        Returns:
            Union[List[Tensor], np.ndarray, Tensor]: Encoded corpus.
        """

        contexts = []
        for evidence in corpus:
            context = ""
            if evidence.title():
                context = (evidence.title() + self.sep + evidence.text()).strip()
            else:
                context = evidence.text().strip()
            contexts.append(context)
        context_embeddings = []
        index = 0
        pbar = tqdm(total=len(contexts))
        self.logger.info("Starting encoding of contexts....")
        with torch.no_grad():
            while index < len(contexts):
                samples = contexts[index : index + self.batch_size]
                tokenized_contexts = self.context_tokenizer(
                    samples, padding=True, truncation=True, return_tensors="pt"
                ).to("cuda")
                token_emb = self.context_encoder(**tokenized_contexts)
                sentence_emb = self._mean_pooling(
                    token_emb[0], tokenized_contexts["attention_mask"]
                )
                context_embeddings.append(sentence_emb)
                index += self.batch_size
                pbar.update(self.batch_size)
        pbar.close()
        context_embeddings = torch.cat(context_embeddings, dim=0)
        return context_embeddings

    def set_query_encoder(self, query_encoder_path: str) -> None:
        """Patch to replace query encoder after initialization."""
        try:
            self.question_encoder = AutoModel.from_pretrained(query_encoder_path)
            self.question_encoder.cuda()
            self.config.query_encoder_path = query_encoder_path
            self.logger.info(
                f"Adore query encoder successfully overwritten to: {query_encoder_path}"
            )
        except Exception as e:
            self.logger.error("Failed to load query encoder...!")
            self.logger.error(e)
            raise e

    def set_query_tokenizer(self, query_tokenizer_path: str) -> None:
        """Patch to replace query tokenizer after initialization."""
        try:
            self.question_tokenizer = AutoTokenizer.from_pretrained(
                query_tokenizer_path
            )
            self.logger.info(
                f"Adore query tokenizer successfully overwritten to: {query_tokenizer_path}"
            )
        except Exception as e:
            self.logger.error("Failed to load query tokenizer...!")
            self.logger.error(e)
            raise e

    def encode_queries(
        self, queries: List[Question], batch_size: int = 16, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        with torch.no_grad():
            tokenized_questions = self.question_tokenizer(
                [query.text() for query in queries],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            token_emb = self.question_encoder(**tokenized_questions)
        sentence_emb = self._mean_pooling(
            token_emb[0], tokenized_questions["attention_mask"]
        )
        return sentence_emb

    # Make it return hard negatives and positives in the same format
    def retrieve(
        self,
        corpus: List[Evidence],
        queries: List[Question],
        top_k: int,
        score_function: SimilarityMetric,
        return_sorted: bool = True,
        return_hard_negatives: bool = False,  # New flag to return negatives
        qrels: Dict[str, List[str]] = None,  # Ground truth relevance
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve top-k results and optionally sample hard negatives.
        
        Slightly altered, but mostly copied from dexter/retriever/dense/HfRetriever.py
        """

        self.logger.debug("Encoding Queries...")
        query_embeddings = self.encode_queries(queries, batch_size=self.batch_size)

        self.logger.debug("Encoding Corpus...")
        corpus_embeddings, index_present = self._load_index_if_available()
        if not index_present:
            self.logger.info("Preparing Embeddings...")
            corpus_embeddings = self._encode_corpus(corpus)
            os.makedirs(self.corpus_folder, exist_ok=True)
            path = os.path.join(self.corpus_folder, self.corpus_file)
            joblib.dump(corpus_embeddings, path)

        # Compute similarites using either cosine-similarity or dot product
        cos_scores = score_function.evaluate(query_embeddings,corpus_embeddings)
        
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
        response = {}
        for idx, q in enumerate(queries):
            response[q.id()] = {}
            for index, id in enumerate(cos_scores_top_k_idx[idx]):
                response[q.id()][corpus[id].id()] = float(cos_scores_top_k_values[idx][index])
        return response
                
