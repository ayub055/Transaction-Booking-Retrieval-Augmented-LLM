"""
LLM-Augmented Few-Shot Classifier (Phase 3 — the missing Amazon paper component)

Uses a locally running Ollama model (default: llama3.2) as the LLM backend.
No API keys or external services required — fully local inference.

Start Ollama before use:
    ollama serve
    ollama pull llama3.2

Prompt engineering techniques used:
  A. Canonical examples   — curated best representatives per category (data/canonical_examples.json)
  B. "other" examples     — real OOD examples to teach the OOD boundary
  C. Contrastive hints    — disambiguation notes for confusable category pairs
  D. Chain-of-thought     — step-by-step reasoning for very low confidence cases
  E. Self-consistency     — majority vote over N LLM runs for high-value transactions
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# Disambiguation notes for confusable category pairs
# ---------------------------------------------------------------------------
_DISAMBIGUATION_NOTES = """
DISAMBIGUATION NOTES (read carefully before classifying):
- P2P vs other: P2P = deliberate money transfer to a known individual (family, friend,
  colleague). "other" = purpose is unclear, or recipient is a business/vendor/unknown.
- Education vs other: Education = payment to school, college, coaching centre, exam board,
  online course, educational books. "other" = generic transfer with no educational context.
- Healthcare vs other: Healthcare = payment to hospital, clinic, pharmacy, diagnostic lab,
  or a health-insurance premium. "other" = person's name with no medical context.
- Food_restaurants vs Home_commodity_services: Food_restaurants = dining, food delivery,
  cafes, canteens.  Home_commodity_services = household bills, appliances, utilities,
  repairs, furniture.
- Mobility vs other: Mobility = fuel, vehicle maintenance, cab/auto booking, toll, parking.
  "other" = travel purpose unclear.
"""


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------
class _OllamaBackend:
    """
    Calls a locally running Ollama instance.

    Start Ollama:
        ollama serve
        ollama pull llama3.2
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        self.model    = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.0) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model":  self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# LLMClassifier
# ---------------------------------------------------------------------------
class LLMClassifier:
    """
    Hybrid LLM classifier using a local Ollama model (llama3.2 by default).
    Steps in when the k-NN vote has low confidence or the transaction is OOD.

    Requires Ollama running locally:
        ollama serve
        ollama pull llama3.2

    Typical usage
    -------------
    pipeline = TransactionInferencePipeline(...)
    llm_clf  = LLMClassifier(
        pipeline,
        categories=list_of_43_category_strings,
    )

    result = pipeline.predict(txn)
    if result['needs_llm']:
        result = llm_clf.classify(txn, retrieval_result=result)
    """

    def __init__(
        self,
        pipeline,                                        # TransactionInferencePipeline
        categories: List[str],                           # all 43 target category strings
        model: str = "llama3.2",                         # Ollama model tag
        base_url: str = "http://localhost:11434",        # Ollama server URL
        canonical_examples_path: Optional[str] = "data/canonical_examples.json",
        ood_examples_path: Optional[str] = "data/ood_examples.json",
        confidence_threshold: float = 0.75,
        high_value_threshold: float = 50_000,            # ₹ — self-consistency fires above this
        self_consistency_n: int = 3,                     # LLM runs for self-consistency vote
        cot_confidence_threshold: float = 0.4,           # chain-of-thought below this confidence
    ):
        self.pipeline             = pipeline
        self.categories           = sorted(categories)
        self.confidence_threshold = confidence_threshold
        self.high_value_threshold = high_value_threshold
        self.self_consistency_n   = self_consistency_n
        self.cot_threshold        = cot_confidence_threshold
        self.llm                  = _OllamaBackend(model=model, base_url=base_url)

        # Load canonical examples {category: [list of example dicts]}
        self.canonical: dict = {}
        if canonical_examples_path and os.path.exists(canonical_examples_path):
            with open(canonical_examples_path) as f:
                self.canonical = json.load(f)

        # Load real "other" examples for OOD boundary teaching
        self.ood_examples: list = []
        if ood_examples_path and os.path.exists(ood_examples_path):
            with open(ood_examples_path) as f:
                self.ood_examples = json.load(f)

        print(f"LLMClassifier ready | model={self.llm.model} | url={self.llm.base_url}")

    # ------------------------------------------------------------------
    def classify(
        self,
        txn: Dict,
        retrieval_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Classify a single transaction using LLM few-shot reasoning.

        Args:
            txn:              Transaction dict with tran_partclr, tran_amt_in_ac,
                              dr_cr_indctor, tran_mode.
            retrieval_result: Output of TransactionInferencePipeline.predict().
                              If None, calls the pipeline internally.

        Returns:
            Updated result dict with 'predicted_category' from the LLM.
        """
        if retrieval_result is None:
            retrieval_result = self.pipeline.predict(txn, return_embeddings=False)

        confidence = retrieval_result.get('confidence', 0.0)
        top_k_txns = retrieval_result.get('similar_transactions', [])
        top_cat    = retrieval_result.get('predicted_category', '')
        amount     = float(txn.get('tran_amt_in_ac', 0))

        use_cot = confidence < self.cot_threshold
        n_runs  = self.self_consistency_n if amount > self.high_value_threshold else 1

        prompt    = self._build_prompt(txn, top_k_txns, top_cat, use_cot)
        raw_preds = [
            self.llm.generate(prompt, temperature=0.0 if n_runs == 1 else 0.3)
            for _ in range(n_runs)
        ]

        if n_runs > 1:
            parsed    = [self._parse_response(r) for r in raw_preds]
            llm_label = Counter(parsed).most_common(1)[0][0]
        else:
            llm_label = self._parse_response(raw_preds[0])

        result = dict(retrieval_result)
        result['predicted_category'] = llm_label
        result['llm_raw_response']   = raw_preds[0] if n_runs == 1 else raw_preds
        result['llm_used']           = True
        result['confidence']         = 1.0 if llm_label != 'other' else 0.0
        return result

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        txn: Dict,
        retrieved: List[Dict],
        top_candidate: str,
        use_cot: bool,
    ) -> str:
        cats_str = ", ".join(self.categories)

        # Section A: canonical examples for the top candidate category
        canon_shots = ""
        if top_candidate and top_candidate in self.canonical:
            for ex in self.canonical[top_candidate][:2]:
                canon_shots += self._fmt_example(ex, ex.get('category', top_candidate))

        # Section B: top k-NN retrieved examples
        retrieval_shots = ""
        for sim in retrieved[:4]:
            t = sim['transaction']
            ex = {
                'tran_partclr':   t.get('description', ''),
                'tran_amt_in_ac': t.get('amount', 0),
                'dr_cr_indctor':  t.get('dr_cr', ''),
                'tran_mode':      t.get('mode', ''),
            }
            retrieval_shots += self._fmt_example(ex, sim.get('label', ''))

        # Section C: real "other" examples (teaches OOD boundary)
        ood_shots = ""
        for ex in self.ood_examples[:2]:
            ood_shots += self._fmt_example(ex, 'other')

        # Section D / E: CoT or direct answer instruction
        if use_cot:
            answer_instruction = (
                "Think step by step:\n"
                "1. Does the description mention a recognisable merchant or business type?\n"
                "2. Is the amount consistent with this category?\n"
                "3. Does Debit/Credit direction support this category?\n"
                "4. Could this be 'other' (genuinely unclassifiable)?\n"
                "State your reasoning briefly, then on the LAST line write ONLY the category name."
            )
        else:
            answer_instruction = "Output ONLY the category name on a single line. No explanation."

        return f"""You are a banking transaction classifier for Indian retail banking.

Classify the transaction into EXACTLY ONE category from the list, or output "other" if none fits.

VALID CATEGORIES:
{cats_str}
(Use "other" only when the transaction genuinely does not belong to any category above.)
{_DISAMBIGUATION_NOTES}
LABELED EXAMPLES — study these carefully:
{canon_shots}{retrieval_shots}{ood_shots}
NEW TRANSACTION TO CLASSIFY:
Description : {txn.get('tran_partclr', '')}
Amount      : ₹{float(txn.get('tran_amt_in_ac', 0)):.2f}
Direction   : {'Debit' if txn.get('dr_cr_indctor') == 'D' else 'Credit'}
Mode        : {txn.get('tran_mode', 'N/A')}

{answer_instruction}"""

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_example(txn: Dict, label: str) -> str:
        direction = 'Debit' if txn.get('dr_cr_indctor') == 'D' else 'Credit'
        return (
            f"  Description : {txn.get('tran_partclr', '')}\n"
            f"  Amount      : ₹{float(txn.get('tran_amt_in_ac', 0)):.2f} | "
            f"Direction: {direction} | "
            f"Mode: {txn.get('tran_mode', 'N/A')}\n"
            f"  → Category  : {label}\n\n"
        )

    # ------------------------------------------------------------------
    def _parse_response(self, response: str) -> str:
        """
        Extract a valid category name from the raw LLM response.
        Handles CoT output (take last non-empty line) and fuzzy matching.
        """
        lines     = [l.strip() for l in response.strip().splitlines() if l.strip()]
        last_line = lines[-1] if lines else response.strip()

        # Remove common artefacts: quotes, punctuation, "Category:" prefixes
        cleaned = re.sub(r'^(category\s*[:\-]?\s*)', '', last_line, flags=re.IGNORECASE)
        cleaned = cleaned.strip('"\' .,')

        # Exact match (case-insensitive)
        lower_map = {c.lower(): c for c in self.categories}
        lower_map['other'] = 'other'
        if cleaned.lower() in lower_map:
            return lower_map[cleaned.lower()]

        # Partial match — first category that is a substring of the response
        for cat in self.categories:
            if cat.lower() in cleaned.lower() or cleaned.lower() in cat.lower():
                return cat

        return 'other'
