"""
Open-Vocabulary Category Embedding Mapper (Phase 3)

Decouples the 43-category inference taxonomy from whatever categories
were used during FusionEncoder training.

Each target category is described in natural language.  The descriptions are
embedded via the FusionEncoder's text branch to produce 43 prototype vectors.
At inference, cosine similarity between the query embedding and each category
prototype gives a soft probability distribution over all categories — without
any retraining when categories are added or renamed.

Usage
-----
mapper = TaxonomyMapper(
    encoder=pipeline.encoder,
    tokenizer=pipeline.tokenizer,
    category_descriptions=CATEGORY_DESCRIPTIONS,
    device=pipeline.device,
)

# Pure taxonomy-based prediction (no k-NN)
scores = mapper.score(query_embedding)   # {category_name: float}

# Or blend with k-NN scores from the inference pipeline
final_scores = mapper.blend(knn_scores, query_embedding, alpha=0.7)
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Default category descriptions (update to match your actual 43 categories)
# ---------------------------------------------------------------------------
DEFAULT_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "Food_restaurants":         "restaurants, cafes, food delivery, dining, takeaway, canteen, swiggy, zomato",
    "Home_commodity_services":  "household bills, home appliances, furniture, interior, utility services, plumber, electrician",
    "Education":                "school fees, college tuition, coaching, exam board, online course, stationery, textbooks",
    "P2P":                      "personal money transfer to individual, send money to friend, family payment, person to person",
    "Mobility":                 "fuel, petrol, diesel, cab, auto, ride hailing, ola, uber, vehicle maintenance, toll, parking",
    "Healthcare":               "hospital, clinic, pharmacy, diagnostic lab, health insurance premium, doctor consultation",
    "Shopping":                 "retail shopping, clothing, electronics, online marketplace, amazon, flipkart, e-commerce",
    "Salary":                   "monthly salary credit, payroll, employer payment, wage transfer",
    "Utilities":                "electricity bill, water bill, gas, broadband, mobile recharge, DTH, internet",
    "Entertainment":            "movies, OTT subscription, netflix, hotstar, gaming, events, concert tickets",
    "Investment":               "mutual funds, stocks, SIP, trading, demat, broker, fixed deposit",
    "Rent":                     "house rent, apartment rent, property rental payment, lease",
    "Groceries":                "supermarket, grocery store, vegetables, fruits, daily essentials, big bazaar, dmart",
    "Transportation":           "train ticket, bus, flight, metro, travel booking, irctc, makemytrip",
    "Transfer":                 "bank transfer, fund transfer, NEFT, RTGS, IMPS transfer to account",
    # Add all remaining categories with natural language descriptions
    "other":                    "unclassified transaction, miscellaneous, purpose unclear, unknown merchant",
}


class TaxonomyMapper:
    """
    Embeds category descriptions via FusionEncoder's text branch and provides
    cosine-similarity-based classification scores.
    """

    def __init__(
        self,
        encoder,
        tokenizer,
        category_descriptions: Optional[Dict[str, str]] = None,
        device: Optional[torch.device] = None,
        descriptions_path: Optional[str] = None,
    ):
        """
        Args:
            encoder:               Trained FusionEncoder (used in eval mode).
            tokenizer:             BertTokenizer matching the encoder.
            category_descriptions: {category_name: description_string}.
                                   Defaults to DEFAULT_CATEGORY_DESCRIPTIONS.
            device:                Torch device.
            descriptions_path:     Optional JSON file to load/override descriptions.
        """
        self.encoder   = encoder
        self.tokenizer = tokenizer
        self.device    = device or next(encoder.parameters()).device

        # Load descriptions
        descriptions = dict(DEFAULT_CATEGORY_DESCRIPTIONS)
        if category_descriptions:
            descriptions.update(category_descriptions)
        if descriptions_path and os.path.exists(descriptions_path):
            with open(descriptions_path) as f:
                descriptions.update(json.load(f))

        self.descriptions = descriptions
        self._embed_all_categories()

    # ------------------------------------------------------------------
    def _embed_all_categories(self):
        """Pre-compute L2-normalised embeddings for all category descriptions."""
        self.encoder.eval()
        category_embeddings: dict = {}

        with torch.no_grad():
            for cat, desc in self.descriptions.items():
                text = f"The transaction description is: {desc}"
                enc  = self.tokenizer(
                    text, padding='max_length', truncation=True,
                    max_length=128, return_tensors='pt',
                )
                ids  = enc['input_ids'].to(self.device)
                mask = enc['attention_mask'].to(self.device)

                # Use only the text branch of the encoder (no categorical/numeric)
                bert_out    = self.encoder.bert(input_ids=ids, attention_mask=mask)
                token_embs  = bert_out.last_hidden_state
                attn_mask   = mask.unsqueeze(-1).float()
                text_emb    = (token_embs * attn_mask).sum(1) / attn_mask.sum(1).clamp(min=1e-9)
                text_proj   = self.encoder.text_proj(text_emb)   # [1, text_proj_dim]
                # L2-normalise
                text_proj   = F.normalize(text_proj, p=2, dim=1)
                category_embeddings[cat] = text_proj[0].cpu().numpy().astype('float32')

        self.categories         = sorted(category_embeddings.keys())
        self.category_embeddings = {c: category_embeddings[c] for c in self.categories}
        # Matrix [C, D] for fast batch scoring
        self._matrix = np.stack(
            [category_embeddings[c] for c in self.categories], axis=0
        ).astype('float32')

    # ------------------------------------------------------------------
    def score(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """
        Compute cosine similarity between query_embedding and all category prototypes.

        Args:
            query_embedding: [1, D] or [D] float32 numpy array (L2-normalised).

        Returns:
            {category_name: similarity_score} — values in [-1, 1].
        """
        q = query_embedding.flatten()
        sims = (self._matrix @ q)   # [C]
        return {cat: float(sims[i]) for i, cat in enumerate(self.categories)}

    # ------------------------------------------------------------------
    def predict(self, query_embedding: np.ndarray) -> str:
        """Return the category name with highest cosine similarity."""
        scores = self.score(query_embedding)
        return max(scores, key=scores.__getitem__)

    # ------------------------------------------------------------------
    def blend(
        self,
        knn_scores:      Dict[str, float],
        query_embedding: np.ndarray,
        alpha:           float = 0.7,
    ) -> Dict[str, float]:
        """
        Blend k-NN vote distribution with taxonomy similarity scores.

        Args:
            knn_scores:      {category_name: vote_weight_fraction} from pipeline.
            query_embedding: [1, D] or [D] float32 array.
            alpha:           weight given to k-NN scores (1-alpha → taxonomy).

        Returns:
            Combined score dict {category_name: score}, normalised to sum=1.
        """
        taxonomy_raw  = self.score(query_embedding)
        all_cats      = set(list(knn_scores.keys()) + list(taxonomy_raw.keys()))

        # Normalise taxonomy scores to [0, 1]
        t_vals  = list(taxonomy_raw.values())
        t_min, t_range = min(t_vals), max(t_vals) - min(t_vals) + 1e-9
        t_norm  = {c: (taxonomy_raw.get(c, t_min) - t_min) / t_range for c in all_cats}

        combined = {
            c: alpha * knn_scores.get(c, 0.0) + (1 - alpha) * t_norm.get(c, 0.0)
            for c in all_cats
        }
        total    = sum(combined.values()) + 1e-9
        return {c: v / total for c, v in sorted(combined.items(), key=lambda x: -x[1])}

    # ------------------------------------------------------------------
    def add_category(self, name: str, description: str):
        """
        Add a new category at runtime — no retraining needed.
        Just provide a natural language description.
        """
        self.descriptions[name] = description
        self._embed_all_categories()   # re-embed all (fast: text-only, no FAISS rebuild)
        print(f"TaxonomyMapper: added category '{name}'")
