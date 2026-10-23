import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from trading.executor import Executor
from trading.market_manager import MarketManager
from trading.paper_engine import PaperEngine
from trading.risk_manager import RiskManager


def make_executor(**kwargs) -> Executor:
    defaults = dict(
        mode="PAPER", host="https://clob.polymarket.com", chain_id=137,
        private_key="0x" + "0" * 64, funder="0x" + "0" * 40,
        max_single_trade_usd=10.0, max_daily_spend_usd=50.0,
    )
    defaults.update(kwargs)
    return Executor(**defaults)


@pytest.mark.unit
class TestSpendGuard:
    def test_allows_trade_under_caps(self):
        ex = make_executor()
        assert ex.check_spend_limits(5.0) is True

    def test_blocks_single_trade_over_cap(self):
        ex = make_executor()
        assert ex.check_spend_limits(10.01) is False

    def test_blocks_when_daily_cap_exhausted(self):
        ex = make_executor()
        for _ in range(5):
            ex.record_spend(10.0)  # 50 total
        assert ex.check_spend_limits(1.0) is False

    def test_daily_window_expires(self):
        ex = make_executor()
        ex._spend_log.append((time.time() - 86401, 50.0))
        assert ex.check_spend_limits(10.0) is True

    def test_cumulative_boundary_exact_cap_allowed(self):
        ex = make_executor()
        ex.record_spend(40.0)
        assert ex.check_spend_limits(10.0) is True
        assert ex.check_spend_limits(10.01) is False

    def test_non_live_execute_is_noop(self):
        ex = make_executor()
        assert ex.execute("BUY_DOWN", "tok", 0.5, 5.0, 100.0) is False


@pytest.mark.unit
class TestWinnerParsing:
    def test_up_winner(self):
        data = {"tokens": [
            {"outcome": "Up", "winner": True},
            {"outcome": "Down", "winner": False},
        ]}
        assert MarketManager.parse_winner(data) == "UP"

    def test_down_winner_via_no_outcome(self):
        data = {"tokens": [
            {"outcome": "YES", "winner": False},
            {"outcome": "NO", "winner": True},
        ]}
        assert MarketManager.parse_winner(data) == "DOWN"

    def test_unresolved_returns_none(self):
        data = {"tokens": [
            {"outcome": "Up", "winner": False},
            {"outcome": "Down", "winner": False},
        ]}
        assert MarketManager.parse_winner(data) is None

    def test_empty_market_returns_none(self):
        assert MarketManager.parse_winner({}) is None

    def test_get_market_winner_uses_clob_data(self, monkeypatch):
        mm = MarketManager("https://clob.polymarket.com")
        monkeypatch.setattr(
            mm, "_fetch_from_clob",
            lambda cid: {"tokens": [{"outcome": "Down", "winner": True}]},
        )
        assert mm.get_market_winner("0xabc", attempts=1) == "DOWN"

    def test_get_market_winner_none_without_condition_id(self):
        mm = MarketManager("https://clob.polymarket.com")
        assert mm.get_market_winner(None) is None


@pytest.mark.unit
class TestPreviousMarketTracking:
    def test_adoption_preserves_previous_market(self):
        mm = MarketManager("https://clob.polymarket.com")
        market_a = {
            "conditionId": "0xaaa", "id": "market-a",
            "tokens": [{"outcome": "Up", "token_id": "t1"},
                       {"outcome": "Down", "token_id": "t2"}],
        }
        market_b = {
            "conditionId": "0xbbb", "id": "market-b",
            "tokens": [{"outcome": "Up", "token_id": "t3"},
                       {"outcome": "Down", "token_id": "t4"}],
        }
        assert mm._try_adopt_market(market_a, epoch_end=time.time() + 300)
        assert mm.previous_condition_id is None
        assert mm._try_adopt_market(market_b, epoch_end=time.time() + 600)
        assert mm.previous_condition_id == "0xaaa"
        assert mm.previous_market_id == "market-a"
        assert mm.current_condition_id == "0xbbb"


@pytest.mark.unit
class TestRiskManagerSeeding:
    def test_seed_bankroll_rebaselines_everything(self):
        rm = RiskManager(starting_bankroll=1000.0)
        rm.seed_bankroll(250.0)
        assert rm.starting_bankroll == 250.0
        assert rm.current_bankroll == 250.0
        assert rm.window_start_bankroll == 250.0

    def test_drawdown_uses_seeded_baseline(self):
        rm = RiskManager(starting_bankroll=1000.0)
        rm.seed_bankroll(100.0)
        rm.update_bankroll(79.0)  # -21% vs seeded 100 -> beyond 20% limit
        assert rm.validate_trade(5.0) is False


@pytest.mark.unit
class TestLiveLedgerFlow:
    def test_fill_then_settle_win(self):
        ledger = PaperEngine(start_balance=100.0)
        assert ledger.execute_order("BUY_DOWN", 0.5, 10.0, 100.0)
        assert ledger.balance == 90.0
        pnl = ledger.settle_market("DOWN")  # 20 tokens -> $20 payout
        assert pnl == pytest.approx(10.0)
        assert ledger.balance == pytest.approx(110.0)

    def test_fill_then_settle_loss(self):
        ledger = PaperEngine(start_balance=100.0)
        assert ledger.execute_order("BUY_DOWN", 0.5, 10.0, 100.0)
        pnl = ledger.settle_market("UP")
        assert pnl == pytest.approx(-10.0)
        assert ledger.balance == pytest.approx(90.0)


@pytest.mark.unit
class TestIsotonicCalibrator:
    def _fitted(self):
        import numpy as np
        from ml.calibration import IsotonicLogOddsCalibrator
        rng = np.random.default_rng(7)
        raw = rng.uniform(0.40, 0.60, 500)
        labels = (rng.uniform(0, 1, 500) < raw).astype(int)
        return IsotonicLogOddsCalibrator().fit(raw, labels)

    def test_predict_proba_shape_and_sum(self):
        import numpy as np
        cal = self._fitted()
        out = cal.predict_proba([[0.0], [0.1]])
        assert out.shape == (2, 2)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_monotone_in_log_odds(self):
        import numpy as np
        cal = self._fitted()
        los = np.linspace(-0.5, 0.5, 21).reshape(-1, 1)
        p_up = cal.predict_proba(los)[:, 1]
        assert np.all(np.diff(p_up) >= -1e-12)

    def test_output_clipped(self):
        cal = self._fitted()
        out = cal.predict_proba([[-10.0], [10.0]])[:, 1]
        assert out[0] >= 0.10 and out[1] <= 0.90

    def test_pickle_roundtrip(self):
        import pickle
        cal = self._fitted()
        clone = pickle.loads(pickle.dumps(cal))
        assert clone.predict_proba([[0.05]])[0][1] == cal.predict_proba([[0.05]])[0][1]

    def test_production_artifact_loads(self):
        import os
        import pickle
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "catboost_model_v2_calibrator.pkl")
        if not os.path.exists(path):
            pytest.skip("production calibrator not present")
        with open(path, "rb") as f:
            cal = pickle.load(f)
        p = cal.predict_proba([[0.0]])[0][1]
        assert 0.0 < p < 1.0


@pytest.mark.unit
class TestHybridFeatureContract:
    def test_model_feature_names_exclude_sample_counts(self):
        from ml.label_dataset_v3 import MODEL_FEATURE_NAMES
        assert len(MODEL_FEATURE_NAMES) == 46
        assert "prev_n_samples" not in MODEL_FEATURE_NAMES
        assert "curr_n_samples_90s" not in MODEL_FEATURE_NAMES
        assert "curr_return_90s" in MODEL_FEATURE_NAMES

    def test_production_model_matches_contract(self):
        import os
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "catboost_model_v2.cbm")
        if not os.path.exists(model_path):
            pytest.skip("production model not present")
        from catboost import CatBoostClassifier
        from ml.label_dataset_v3 import MODEL_FEATURE_NAMES
        m = CatBoostClassifier()
        m.load_model(model_path)
        assert list(m.feature_names_) == list(MODEL_FEATURE_NAMES)

    def test_build_curr_features_return_math(self):
        from ml.label_dataset_v3 import build_curr_features
        rows = [{"timestamp": 0, "last_price": 50100.0,
                 "obi": 0.1, "cvd": 1.0, "weighted_obi": 0.2, "toxicity": 0.5,
                 "trade_intensity": 3.0, "volume_accel": 0.0}]
        f = build_curr_features(rows, epoch_start_price=50000.0)
        assert f["curr_return_90s"] == pytest.approx(20.0)  # +100/50000 in bps
        assert f["curr_n_samples_90s"] == 1.0

    def test_build_curr_features_empty_window(self):
        from ml.label_dataset_v3 import build_curr_features
        f = build_curr_features([], epoch_start_price=50000.0)
        assert f["curr_return_90s"] == 0.0
        assert f["curr_n_samples_90s"] == 0.0
