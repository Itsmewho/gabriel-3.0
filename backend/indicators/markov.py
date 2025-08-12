import json
from collections import defaultdict
from typing import Any, Optional, Union
from operations.redis_operations import get_cache


def build_markov_prediction(historical_states: list) -> dict:
    """
    Takes a list of historical states and builds a transition matrix
    and predicts the next state.
    """
    states = [item["state"] for item in historical_states]

    states.reverse()

    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(states) - 1):
        transition_counts[states[i]][states[i + 1]] += 1

    transition_matrix = {}
    for state, transitions in transition_counts.items():
        total = sum(transitions.values())
        transition_matrix[state] = {
            next_state: round(count / total, 2)
            for next_state, count in sorted(transitions.items())
        }

    current_state = states[-1]
    next_transitions = transition_matrix.get(current_state, {})
    predicted_next_state = (
        max(next_transitions.items(), key=lambda item: item[1])[0]
        if next_transitions
        else "Unknown"
    )

    return {
        "current_state": current_state,
        "transition_matrix": transition_matrix,
        "predicted_next_state": predicted_next_state,
    }


def calculate_markov_states_historic(
    symbol: str, timeframe: str
) -> dict[str, Union[bool, str, list[dict[str, str]]]]:
    history_key = f"market:previous:{symbol.upper()}:{timeframe}"
    raw_data: Optional[str] = get_cache(history_key)

    if not raw_data:
        return {"success": False, "message": "No historical data found"}

    try:
        market_data: list[dict[str, Any]] = json.loads(raw_data)

        historical_states: list[dict[str, str]] = []
        for entry in market_data:
            change = entry["close"] - entry["open"]
            state = "Bullish" if change > 0 else "Bearish" if change < 0 else "Neutral"

            timestamp = entry["time"]
            if not timestamp.endswith("Z"):
                timestamp += "Z"

            historical_states.append({"time": timestamp, "state": state})

        return {"success": True, "data": historical_states}

    except Exception as e:
        return {
            "success": False,
            "message": f"Error calculating Markov states: {str(e)}",
        }
