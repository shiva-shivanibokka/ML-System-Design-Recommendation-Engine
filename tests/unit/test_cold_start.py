import numpy as np
import pytest
from serving.cold_start import ColdStartHandler


@pytest.fixture
def handler():
    h = ColdStartHandler()
    h._popularity_pool = [(1, 1.0), (2, 0.8), (3, 0.6), (4, 0.4), (5, 0.2)]
    h._item_genre_vectors = {
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
        3: np.array([1.0, 0.0, 0.0]),  # same direction as item 1
    }
    h._item_meta = {
        1: {"title": "Pulp Fiction", "primary_genre": "Thriller"},
        2: {"title": "Toy Story", "primary_genre": "Animation"},
        3: {"title": "Die Hard", "primary_genre": "Action"},
    }
    h._loaded = True
    return h


def test_is_cold_user_below_threshold(handler):
    assert handler.is_cold_user(user_id=99, interaction_count=3) is True


def test_is_cold_user_above_threshold(handler):
    assert handler.is_cold_user(user_id=1, interaction_count=10) is False


def test_is_cold_item_below_threshold(handler):
    assert handler.is_cold_item(item_id=99, interaction_count=5) is True


def test_is_cold_item_above_threshold(handler):
    assert handler.is_cold_item(item_id=1, interaction_count=15) is False


def test_popularity_fallback_returns_top_k(handler):
    result = handler.get_popularity_fallback(top_k=3)
    assert len(result) == 3
    assert result[0][0] == 1  # highest score first


def test_popularity_fallback_excludes_seen(handler):
    result = handler.get_popularity_fallback(top_k=3, exclude_seen={1, 2})
    ids = [r[0] for r in result]
    assert 1 not in ids
    assert 2 not in ids


def test_popularity_fallback_respects_top_k(handler):
    result = handler.get_popularity_fallback(top_k=2)
    assert len(result) == 2


def test_content_based_fallback_finds_similar(handler):
    # Item 1 and 3 share vector [1,0,0]; liked_item_ids=[1] should surface item 3
    result = handler.get_content_based_fallback(liked_item_ids=[1], top_k=2, exclude_seen={1})
    ids = [r[0] for r in result]
    assert 3 in ids


def test_content_based_fallback_no_liked_items_falls_back(handler):
    result = handler.get_content_based_fallback(liked_item_ids=[], top_k=3)
    assert len(result) == 3  # falls back to popularity


def test_get_item_title_known(handler):
    assert handler.get_item_title(1) == "Pulp Fiction"


def test_get_item_title_unknown(handler):
    assert handler.get_item_title(9999) == "Item 9999"


def test_get_item_genre_known(handler):
    assert handler.get_item_genre(2) == "Animation"


def test_get_item_genre_unknown(handler):
    assert handler.get_item_genre(9999) == "Unknown"
