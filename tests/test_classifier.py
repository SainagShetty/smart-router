from smartrouter.classifiers.embedding import EmbeddingClassifier
from smartrouter.classifiers.heuristic import HeuristicClassifier
from smartrouter.features import extract


def _score(clf, text):
    return clf.score(extract([{"role": "user", "content": text}]))


def test_bundled_classifier_separates_easy_hard():
    clf = EmbeddingClassifier.bundled()
    easy = _score(clf, "what is the capital of france?")
    hard = _score(clf, "prove the halting problem is undecidable via diagonalization "
                       "and analyze the complexity")
    assert easy < 0.45
    assert hard > 0.55
    assert hard > easy


def test_bundled_score_in_range():
    clf = EmbeddingClassifier.bundled()
    for t in ["hi", "x" * 5000, "design a distributed consensus protocol"]:
        s = _score(clf, t)
        assert 0.0 <= s <= 1.0


def test_score_and_embed_returns_vector():
    clf = EmbeddingClassifier.bundled()
    score, emb = clf.score_and_embed(extract([{"role": "user", "content": "hello"}]))
    assert 0.0 <= score <= 1.0
    assert emb is not None and emb.ndim == 1


def test_heuristic_monotonic_on_keywords():
    clf = HeuristicClassifier()
    plain = _score(clf, "say hello")
    hard = _score(clf, "prove and derive the optimal algorithm step by step")
    assert hard > plain
    assert clf.embed("x") is None  # no embedding for heuristic


def test_fit_and_roundtrip(tmp_path):
    clf = EmbeddingClassifier(backend="hashing", use_features=True)
    texts = ["hi", "what is 2+2", "prove the theorem and derive the bound",
             "design a distributed system and analyze tradeoffs"]
    labels = [0, 0, 1, 1]
    clf.fit(texts, labels)
    path = str(tmp_path / "head.joblib")
    clf.save(path)
    loaded = EmbeddingClassifier.load(path)
    assert loaded.use_features is True
    assert loaded.embedding_model_id == clf.embedding_model_id
    s = loaded.score(extract([{"role": "user", "content": "prove the theorem"}]))
    assert 0.0 <= s <= 1.0
