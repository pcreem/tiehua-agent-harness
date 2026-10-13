"""Intent Classification Node for Tool Routing.

Classifies user intent to enable smart tool selection and routing.
"""

import re
from typing import Tuple, List, Dict
from src.state import AgentState
from src.harness.graph_types import HarnessGraphState


class IntentClassifier:
    """Classifies user intent for tool routing and response strategy."""

    INTENTS = ["weather", "attraction", "general", "mixed"]
    DEFAULT_INTENT = "general"

    # Intent keywords with weights
    WEATHER_KEYWORDS = {
        # Chinese (Traditional)
        "天氣": 1.0, "氣溫": 1.0, "溫度": 1.0, "下雨": 1.0, "會不會下雨": 1.0, "雨": 0.8,
        "降雨": 0.9, "濕度": 0.8, "風": 0.7, "颱風": 0.9, "晴": 0.8, "陰": 0.7, "多雲": 0.7,
        # English
        "weather": 1.0, "rain": 1.0, "temperature": 1.0, "forecast": 1.0,
        "humidity": 0.8, "wind": 0.7, "typhoon": 0.9, "sunny": 0.8, "cloudy": 0.7,
        # Japanese
        "天気": 1.0, "気温": 1.0, "温度": 1.0, "雨": 1.0, "雨が降る": 1.0,
        "降雨": 0.9, "湿度": 0.8, "風": 0.7, "台風": 0.9, "晴れ": 0.8, "曇り": 0.7,
        "天気予報": 1.0
    }

    ATTRACTION_KEYWORDS = {
        # Chinese (Traditional)
        "住宿": 1.0, "飯店": 1.0, "民宿": 1.0, "旅館": 1.0, "露營": 1.0,
        "景點": 1.0, "玩": 0.8, "去哪": 0.9, "推薦": 0.9, "好玩": 0.8,
        "活動": 0.9, "夜市": 0.9, "美食": 0.9, "餐廳": 0.9, "小吃": 0.8, "宵夜": 0.8,
        "好吃": 0.8, "吃": 0.6, 
        # English
        "accommodation": 1.0, "hotel": 1.0, "hostel": 0.9, "attraction": 1.0,
        "sightseeing": 0.9, "food": 0.9, "restaurant": 0.9, "activity": 0.9,
        "night market": 0.9, "recommend": 0.8, "where to": 0.9, "taitung": 0.8,
        # Japanese
        "宿泊": 1.0, "ホテル": 1.0, "民宿": 0.9, "旅館": 1.0, "キャンプ": 0.9,
        "観光": 1.0, "観光地": 1.0, "遊ぶ": 0.8, "どこ": 0.9, "おすすめ": 0.9, "面白い": 0.8,
        "アクティビティ": 0.9, "夜市": 0.9, "グルメ": 0.9, "レストラン": 0.9, "屋台": 0.8, "夜食": 0.8,
        "美味しい": 0.8, "食べる": 0.6,         "宿": 0.8, "宿泊施設": 1.0, "観光スポット": 1.0
    }

    GENERAL_KEYWORDS = {
        "你好": 0.5, "哈囉": 0.5, "謝謝": 0.3, "再見": 0.3,
        "hello": 0.5, "hi": 0.5, "thanks": 0.3, "bye": 0.3,
        "こんにちは": 0.5, "ありがとう": 0.3, "さようなら": 0.3,
        "你好嗎": 0.4, "お元気ですか": 0.4, "how are you": 0.4
    }

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile keyword patterns for efficiency."""
        self.weather_pattern = self._build_pattern(self.WEATHER_KEYWORDS)
        self.attraction_pattern = self._build_pattern(self.ATTRACTION_KEYWORDS)
        self.general_pattern = self._build_pattern(self.GENERAL_KEYWORDS)

    def _build_pattern(self, keywords: Dict[str, float]) -> re.Pattern:
        """Build regex pattern from weighted keywords."""
        # Sort by length descending to match longer phrases first
        sorted_kws = sorted(keywords.keys(), key=len, reverse=True)
        pattern = "|".join(re.escape(kw) for kw in sorted_kws)
        return re.compile(pattern, re.IGNORECASE)

    def _score_intent(self, text: str, pattern: re.Pattern, keywords: Dict[str, float]) -> float:
        """Calculate weighted intent score."""
        score = 0.0
        matches = pattern.findall(text)
        for match in matches:
            # Find the original keyword (case-insensitive)
            for kw, weight in keywords.items():
                if kw.lower() == match.lower():
                    score += weight
                    break
        return score

    def classify(self, query: str, language: str = "zh-TW") -> Tuple[str, float]:
        """Classify intent from query.

        Returns:
            Tuple of (intent, confidence)
        """
        if not query or not query.strip():
            return self.DEFAULT_INTENT, 0.3

        text = query.strip()

        # Score each intent
        weather_score = self._score_intent(text, self.weather_pattern, self.WEATHER_KEYWORDS)
        attraction_score = self._score_intent(text, self.attraction_pattern, self.ATTRACTION_KEYWORDS)
        general_score = self._score_intent(text, self.general_pattern, self.GENERAL_KEYWORDS)

        scores = {
            "weather": weather_score,
            "attraction": attraction_score,
            "general": general_score
        }

        # Determine primary intent
        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]

        # Check for mixed intent (both weather and attraction above threshold)
        threshold = 0.5
        active_intents = [k for k, v in scores.items() if v >= threshold]

        if len(active_intents) >= 2:
            intent = "mixed"
            confidence = min(1.0, sum(scores[k] for k in active_intents) / len(active_intents))
        elif max_score >= threshold:
            intent = max_intent
            # Normalize confidence
            total = sum(scores.values())
            confidence = min(1.0, max_score / max(total, 0.1)) if total > 0 else 0.5
        else:
            intent = "general"
            confidence = 0.4

        return intent, confidence

    def process(self, graph_state: HarnessGraphState) -> HarnessGraphState:
        """LangGraph node: classifies intent, updates agent_state."""
        agent_state = graph_state["agent_state"]
        intent, confidence = self.classify(agent_state.query, agent_state.language)
        agent_state.user_intent = intent
        agent_state.intent_confidence = confidence
        agent_state.metadata["intent_classification"] = {
            "intent": intent,
            "confidence": confidence,
            "method": "keyword_weighted"
        }
        agent_state.update_timestamp()

        new_state = graph_state.copy()
        new_state["agent_state"] = agent_state
        new_state["current_node"] = "intent_classification"
        new_state["execution_path"] = graph_state.get("execution_path", []) + ["intent_classification"]
        return new_state


def create_intent_classifier_node():
    """Factory for LangGraph node."""
    classifier = IntentClassifier()

    def node(state: HarnessGraphState) -> HarnessGraphState:
        return classifier.process(state)

    return node