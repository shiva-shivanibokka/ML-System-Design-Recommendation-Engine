import pytest
from serving.post_ranking import PostRanker


@pytest.fixture
def ranker():
    r = PostRanker()
    r.top_n = 5
    r.mmr_lambda = 0.7
    r.freshness_weight = 0.1
    r.max_same_genre = 2
    return r


GENRES = {1: "Action", 2: "Action", 3: "Drama", 4: "Comedy", 5: "Drama", 6: "Action"}


def test_rerank_returns_top_n(ranker):
    candidates = [(i, float(10 - i)) for i in range(1, 11)]
    result = ranker.rerank(candidates, GENRES, top_n=3)
    assert len(result) == 3


def test_rerank_returns_tuple_of_three(ranker):
    candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
    result = ranker.rerank(candidates, GENRES)
    item_id, score, meta = result[0]
    assert isinstance(item_id, int)
    assert isinstance(score, float)
    assert "genre" in meta
    assert "is_fresh" in meta


def test_rerank_empty_candidates(ranker):
    assert ranker.rerank([], GENRES) == []


def test_freshness_boost_raises_score(ranker):
    # Item 3 has lower raw score (0.5) but freshness boost (+10%) should surface it over item 1 (0.51)
    candidates = [(1, 0.51), (3, 0.50)]
    recent = {3}
    result = ranker.rerank(candidates, GENRES, recent_item_ids=recent, top_n=2)
    result_ids = [r[0] for r in result]
    assert result_ids[0] == 3


def test_freshness_boost_marks_is_fresh(ranker):
    candidates = [(1, 1.0), (3, 0.5)]
    result = ranker.rerank(candidates, GENRES, recent_item_ids={3}, top_n=2)
    meta_by_id = {r[0]: r[2] for r in result}
    assert meta_by_id[3]["is_fresh"] is True
    assert meta_by_id[1]["is_fresh"] is False


def test_genre_cap_enforced(ranker):
    # top_n=4: items 1,2 (Action, capped at 2), item 3 (Drama), item 4 (Comedy) fills 4
    # Item 6 (Action) is excluded because the cap is met and top_n=4 is already filled
    extra_genres = {1: "Action", 2: "Action", 3: "Drama", 4: "Comedy", 5: "Drama", 6: "Action"}
    candidates = [(1, 1.0), (2, 0.9), (6, 0.8), (3, 0.7), (4, 0.6)]
    result = ranker.rerank(candidates, extra_genres, top_n=4)
    genre_counts = {}
    for item_id, _, meta in result:
        g = meta["genre"]
        genre_counts[g] = genre_counts.get(g, 0) + 1
    assert genre_counts.get("Action", 0) <= 2


def test_mmr_returns_requested_count(ranker):
    genres = {i: "Action" for i in range(1, 11)}
    candidates = [(i, float(10 - i)) for i in range(1, 11)]
    result = ranker.rerank(candidates, genres, top_n=3)
    assert len(result) == 3


def test_genre_cap_fallback_fills_results(ranker):
    genres = {1: "Action", 2: "Action", 3: "Action", 4: "Drama"}
    ranker.max_same_genre = 1
    candidates = [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7)]
    result = ranker.rerank(candidates, genres, top_n=3)
    ids = [r[0] for r in result]
    assert 1 in ids   # best Action
    assert 4 in ids   # only Drama


def test_metadata_genre_populated(ranker):
    candidates = [(3, 0.9), (4, 0.8)]
    result = ranker.rerank(candidates, GENRES, top_n=2)
    genres_in_meta = [r[2]["genre"] for r in result]
    assert "Drama" in genres_in_meta
    assert "Comedy" in genres_in_meta


def test_single_candidate_returned(ranker):
    result = ranker.rerank([(1, 0.99)], GENRES, top_n=5)
    assert len(result) == 1
    assert result[0][0] == 1
