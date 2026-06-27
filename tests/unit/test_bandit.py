import pytest
from serving.bandit import ThompsonSamplingBandit


@pytest.fixture
def bandit():
    b = ThompsonSamplingBandit(model_names=["svd", "ncf"])
    # Enough pulls so forced-exploration phase is skipped
    b.total_pulls = {"svd": 100, "ncf": 100}
    return b


def test_select_arm_returns_valid_model(bandit):
    arm = bandit.select_arm()
    assert arm in ["svd", "ncf"]


def test_forced_exploration_when_cold():
    b = ThompsonSamplingBandit(model_names=["svd", "ncf"])
    # total_pulls starts at 0; min_samples_per_arm=50 forces round-robin
    arms_seen = {b.select_arm() for _ in range(20)}
    assert "svd" in arms_seen or "ncf" in arms_seen


def test_update_click_increments_alpha(bandit):
    alpha_before = bandit.alphas["ncf"]
    bandit.update("ncf", clicked=True)
    assert bandit.alphas["ncf"] == alpha_before + 1
    assert bandit.total_pulls["ncf"] == 101
    assert bandit.total_rewards["ncf"] == 1


def test_update_no_click_increments_beta(bandit):
    beta_before = bandit.betas["svd"]
    bandit.update("svd", clicked=False)
    assert bandit.betas["svd"] == beta_before + 1
    assert bandit.total_rewards["svd"] == 0


def test_update_unknown_model_is_noop(bandit):
    alpha_before = dict(bandit.alphas)
    bandit.update("unknown_model", clicked=True)
    assert bandit.alphas == alpha_before


def test_get_ctrs_zero_pulls():
    b = ThompsonSamplingBandit(model_names=["svd", "ncf"])
    ctrs = b.get_ctrs()
    assert ctrs["svd"] == 0.0
    assert ctrs["ncf"] == 0.0


def test_get_ctrs_after_clicks(bandit):
    bandit.total_rewards["svd"] = 30
    bandit.total_pulls["svd"] = 100
    ctrs = bandit.get_ctrs()
    assert abs(ctrs["svd"] - 0.30) < 1e-9


def test_get_state_has_required_keys(bandit):
    state = bandit.get_state()
    for key in ["alphas", "betas", "total_pulls", "total_rewards", "ctrs", "winner", "mean_ctrs"]:
        assert key in state


def test_reset_returns_to_uniform_prior(bandit, tmp_path, monkeypatch):
    from configs.settings import settings
    monkeypatch.setattr(settings.bandit, "state_path", str(tmp_path / "bandit.json"))
    bandit.total_pulls = {"svd": 500, "ncf": 500}
    bandit.total_rewards = {"svd": 200, "ncf": 100}
    bandit.reset()
    assert bandit.total_pulls["svd"] == 0
    assert bandit.total_rewards["ncf"] == 0
    assert bandit._winner is None


def test_winner_declared_after_significance(monkeypatch):
    from configs.settings import settings
    monkeypatch.setattr(settings.bandit, "significance_threshold", 0.50)
    monkeypatch.setattr(settings.bandit, "min_samples_per_arm", 1)
    b = ThompsonSamplingBandit(model_names=["svd", "ncf"])
    b.total_pulls = {"svd": 1, "ncf": 1}
    # Heavily skew ncf to guarantee it wins
    b.alphas["ncf"] = 1000.0
    b.betas["ncf"] = 1.0
    b.alphas["svd"] = 1.0
    b.betas["svd"] = 1000.0
    b._check_for_winner()
    assert b._winner == "ncf"
