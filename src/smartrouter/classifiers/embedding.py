"""Default difficulty classifier: embedding + logistic-regression head.

Two embedding backends:
  * "hashing"             -> scikit-learn HashingVectorizer (no torch, ships light)
  * "sentence-transformer" -> a local MiniLM-style model (requires the
                              ``smartrouter[local-embed]`` extra, higher quality)

The head is a scikit-learn ``LogisticRegression`` whose P(hard) is the score.
A pre-fit head is bundled (``data/default_classifier.joblib``) so the classifier
works on day 0; ``fit()`` retrains it on your own logged traffic.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np

from ..features import RequestFeatures, extract
from .base import Classifier

_DEFAULT_HASH_FEATURES = 1024

# A handful of engineered difficulty signals appended to the embedding so the
# linear head sees real "this needs reasoning" cues, not just lexical overlap.
_HARD_KEYWORDS = re.compile(
    r"\b(prove|derive|optimi[sz]e|debug|refactor|algorithm|complexity|"
    r"step[- ]by[- ]step|reason|explain why|trade[- ]?off|architect|"
    r"theorem|integral|differential|recursion|concurren|distributed|"
    r"analy[sz]e|design|diagnose)\b",
    re.IGNORECASE,
)
_N_ENGINEERED = 5


def engineered_features(text: str) -> np.ndarray:
    f = extract([{"role": "user", "content": text}])
    hits = len(set(m.lower() for m in _HARD_KEYWORDS.findall(text)))
    return np.array(
        [
            min(f.estimated_tokens / 600.0, 1.0),
            min(f.code_ratio * 2.0, 1.0),
            min(hits / 3.0, 1.0),
            min(f.raw_chars / 2000.0, 1.0),
            1.0 if f.has_cjk else 0.0,
        ],
        dtype=np.float32,
    )


class _HashingEmbedder:
    def __init__(self, n_features: int = _DEFAULT_HASH_FEATURES):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.n_features = n_features
        self._vec = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm="l2"
        )
        self.model_id = f"hashing:{n_features}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        m = self._vec.transform(list(texts))
        return m.toarray().astype(np.float32)


class _SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "sentence-transformer backend requires `pip install "
                "smartrouter[local-embed]`"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.model_id = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True),
            dtype=np.float32,
        )


def _make_embedder(backend: str, model_name: str, n_features: int):
    if backend == "hashing":
        return _HashingEmbedder(n_features)
    if backend == "sentence-transformer":
        return _SentenceTransformerEmbedder(model_name)
    raise ValueError(f"unknown embedding backend: {backend!r}")


class EmbeddingClassifier(Classifier):
    def __init__(
        self,
        backend: str = "hashing",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_features: int = _DEFAULT_HASH_FEATURES,
        head=None,
        hard_label: int = 1,
        use_features: bool = True,
    ):
        self.backend = backend
        self.embedding_model = embedding_model
        self.n_features = n_features
        self.use_features = use_features
        self._embedder = _make_embedder(backend, embedding_model, n_features)
        suffix = "+feat" if use_features else ""
        self.embedding_model_id = self._embedder.model_id + suffix
        self.head = head  # fitted sklearn classifier or None
        self.hard_label = hard_label

    # ---- inference --------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        base = self._embedder.encode(texts)
        if not self.use_features:
            return base
        eng = np.vstack([engineered_features(t) for t in texts])
        return np.hstack([base, eng]).astype(np.float32)

    def score(self, features: RequestFeatures) -> float:
        if self.head is None:
            raise RuntimeError(
                "EmbeddingClassifier has no fitted head; call fit() or load a "
                "bundled/pre-fit head."
            )
        vec = self.embed(features.text).reshape(1, -1)
        return self.clamp(float(self._proba_hard(vec)[0]))

    def score_vector(self, vec: np.ndarray) -> float:
        """Score from an already-computed embedding (used by eval/retrain)."""
        return self.clamp(float(self._proba_hard(vec.reshape(1, -1))[0]))

    def score_and_embed(self, features: RequestFeatures):
        vec = self.embed(features.text)
        return self.score_vector(vec), vec

    def _proba_hard(self, vecs: np.ndarray) -> np.ndarray:
        proba = self.head.predict_proba(vecs)
        classes = list(self.head.classes_)
        idx = classes.index(self.hard_label) if self.hard_label in classes else -1
        return proba[:, idx]

    # ---- training ---------------------------------------------------------

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "EmbeddingClassifier":
        from sklearn.linear_model import LogisticRegression

        X = self.embed_batch(texts)
        y = np.asarray(labels)
        head = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        )
        head.fit(X, y)
        self.head = head
        return self

    def fit_vectors(
        self, vectors: np.ndarray, labels: Sequence[int]
    ) -> "EmbeddingClassifier":
        """Fit directly from stored embeddings (no re-embedding)."""
        from sklearn.linear_model import LogisticRegression

        head = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        )
        head.fit(vectors, np.asarray(labels))
        self.head = head
        return self

    # ---- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(
            {
                "backend": self.backend,
                "embedding_model": self.embedding_model,
                "n_features": self.n_features,
                "hard_label": self.hard_label,
                "use_features": self.use_features,
                "head": self.head,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "EmbeddingClassifier":
        import joblib

        d = joblib.load(path)
        return cls(
            backend=d["backend"],
            embedding_model=d["embedding_model"],
            n_features=d.get("n_features", _DEFAULT_HASH_FEATURES),
            head=d["head"],
            hard_label=d.get("hard_label", 1),
            use_features=d.get("use_features", True),
        )

    @classmethod
    def bundled(cls) -> "EmbeddingClassifier":
        """Load the pre-fit classifier shipped with the package."""
        from importlib import resources

        with resources.as_file(
            resources.files("smartrouter.data") / "default_classifier.joblib"
        ) as p:
            return cls.load(str(p))
