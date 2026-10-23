from typing import Dict, Any


class EVCalculator:
    # Don't trade tokens priced below this — no real liquidity and unrealistic fills
    MIN_TOKEN_PRICE = 0.05
    MAX_TOKEN_PRICE = 0.95

    def __init__(self, kelly_fraction: float = 0.25, min_ev_threshold: float = 0.03,
                 min_model_confidence: float = 0.05, max_market_disagree: float = 0.40,
                 model_version: str = "v1"):
        # Quarter Kelly for safety
        self.kelly_fraction = max(0.01, min(kelly_fraction, 1.0))
        # Minimum EV to act on (filters noise)
        self.min_ev_threshold = min_ev_threshold
        # Don't trade if model is near 50/50 (within this band of 0.5)
        self.min_model_confidence = min_model_confidence
        # Don't bet against market when disagreement exceeds this
        # e.g. market says 90% UP but model says 50% UP = 40% disagree → skip
        self.max_market_disagree = max_market_disagree
        self.model_version = model_version

    def evaluate_ev(self, prob_up: float, poly_bids: Dict[float, float],
                    poly_asks: Dict[float, float], fee_rate: float,
                    bankroll: float) -> Dict[str, Any]:
        """
        Evaluates EV and Kelly bet size.
        poly_bids/poly_asks represent the UP token's LOB.
        Buying UP = taking the ASK of UP token.
        Buying DOWN = buying NO token, whose implied ask = 1 - best_up_bid.
        """
        if not poly_bids and not poly_asks:
            return self._hold()

        # Model confidence filter: skip if too close to 50/50
        if abs(prob_up - 0.5) < self.min_model_confidence:
            return self._hold()

        best_up_ask = min(poly_asks.keys()) if poly_asks else None
        best_up_bid = max(poly_bids.keys()) if poly_bids else None
        best_no_ask = 1.0 - best_up_bid if best_up_bid is not None else None

        # Market disagreement filter: don't bet against extreme market consensus
        market_mid = None
        if best_up_ask is not None and best_up_bid is not None:
            market_mid = (best_up_ask + best_up_bid) / 2.0
        elif best_up_ask is not None:
            market_mid = best_up_ask
        elif best_up_bid is not None:
            market_mid = best_up_bid

        if market_mid is not None:
            disagree = abs(prob_up - market_mid)
            if disagree > self.max_market_disagree:
                return self._hold()

        action = "HOLD"
        max_ev = 0.0
        target_price = 0.0
        chosen_cost = 0.0
        chosen_prob = 0.0

        # EV of buying UP token. (The old V2-Platt model never signalled UP so this
        # side was suppressed; the hybrid isotonic model predicts both sides honestly
        # — see EXPERIMENTS.md Exp 4.)
        if best_up_ask is not None and self.MIN_TOKEN_PRICE <= best_up_ask <= self.MAX_TOKEN_PRICE:
            cost_up = best_up_ask * (1 + fee_rate)
            if cost_up < 1.0:
                ev_up = (prob_up * (1 - cost_up)) - ((1 - prob_up) * cost_up)
                if ev_up > max_ev:
                    max_ev = ev_up
                    action = "BUY_UP"
                    target_price = best_up_ask
                    chosen_cost = cost_up
                    chosen_prob = prob_up

        # EV of buying NO/DOWN token
        prob_down = 1.0 - prob_up
        if best_no_ask is not None and self.MIN_TOKEN_PRICE <= best_no_ask <= self.MAX_TOKEN_PRICE:
            cost_no = best_no_ask * (1 + fee_rate)
            if cost_no < 1.0:
                ev_down = (prob_down * (1 - cost_no)) - ((1 - prob_down) * cost_no)
                if ev_down > max_ev:
                    max_ev = ev_down
                    action = "BUY_DOWN"
                    target_price = best_no_ask  # actual NO token price
                    chosen_cost = cost_no
                    chosen_prob = prob_down

        # Minimum EV filter to avoid marginal bets
        if max_ev < self.min_ev_threshold:
            return self._hold()

        # Kelly sizing
        bet_size = 0.0
        f_star = 0.0
        if action != "HOLD" and chosen_cost > 0:
            b = (1 - chosen_cost) / chosen_cost  # odds ratio
            q = 1 - chosen_prob
            if b > 0 and chosen_prob > 0:
                f_star = (b * chosen_prob - q) / b
                f_star = min(1.0, max(0.0, float(f_star))) * self.kelly_fraction
                bet_size = bankroll * f_star

        if bet_size < 0.01:
            return self._hold()

        return {
            "action": action,
            "ev": max_ev,
            "target_price": target_price,
            "bet_size_usd": bet_size,
            "kelly_f": f_star if bet_size > 0 else 0.0,
        }

    @staticmethod
    def _hold():
        return {"action": "HOLD", "ev": 0.0, "target_price": 0.0, "bet_size_usd": 0.0, "kelly_f": 0.0}
