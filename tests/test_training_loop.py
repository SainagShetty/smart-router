"""End-to-end proof that logged decisions are genuinely trainable: route a few
requests, attach labels (explicit + implicit), then refit a head purely from the
stored embeddings."""
import numpy as np

from smartrouter import RouterCore
from smartrouter.classifiers.embedding import EmbeddingClassifier
from smartrouter.logging_ import TrainingStore

from conftest import make_config, stub_providers


def test_decisions_are_logged(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    stub_providers(core)
    core.complete([{"role": "user", "content": "what is 2+2"}])
    core.complete([{"role": "user", "content": "prove the theorem and derive it"}])
    assert core.store.count() == 2
    core.close()


def test_explicit_and_implicit_labels(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    stub_providers(core, content="not json")
    # explicit feedback
    d = core.route([{"role": "user", "content": "hello"}])
    assert core.feedback(d.decision_id, 1, "manual") is True
    # implicit: JSON requested but content isn't valid JSON -> weak hard label
    _, d2 = core.complete(
        [{"role": "user", "content": "give me json"}],
        response_format={"type": "json_object"},
    )
    core.close()

    store = TrainingStore(tmp_db)
    assert store.count(labeled_only=True) == 2
    store.close()


def test_retrain_head_from_store(tmp_db):
    cfg = make_config(db_path=tmp_db)
    core = RouterCore(cfg)
    stub_providers(core)
    emb_id = core.classifier.embedding_model_id

    samples = [
        ("what is the capital of italy", 0),
        ("convert 3 miles to km", 0),
        ("summarize this sentence", 0),
        ("say hello", 0),
        ("prove the halting problem is undecidable and analyze complexity", 1),
        ("design a distributed consensus protocol and prove safety", 1),
        ("derive the gradient and optimize the algorithm step by step", 1),
        ("debug this concurrent deadlock and refactor the architecture", 1),
    ]
    for text, label in samples:
        d = core.route([{"role": "user", "content": text}])
        core.feedback(d.decision_id, label, "manual")
    core.close()

    store = TrainingStore(tmp_db)
    X, y = store.training_matrix(emb_id)
    store.close()
    assert X.shape[0] == len(samples)
    assert set(y) == {0, 1}

    # Refit a head purely from stored embeddings — no re-embedding.
    clf = EmbeddingClassifier(backend="hashing", use_features=True)
    clf.fit_vectors(np.asarray(X), y)
    assert clf.head is not None
    easy = clf.score_vector(np.asarray(X[0]))
    hard = clf.score_vector(np.asarray(X[-1]))
    assert 0.0 <= easy <= 1.0 and 0.0 <= hard <= 1.0
