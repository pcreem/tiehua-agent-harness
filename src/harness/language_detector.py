"""Language Detection Node for Multilingual Support.

Lightweight language detection for zh-TW, en, ja using heuristic character analysis.
For production, consider integrating fasttext or langdetect.
"""

import re
from typing import Tuple
from src.state import AgentState
from src.harness.graph_types import HarnessGraphState


class LanguageDetector:
    """Detects user language from query text.

    Supports: zh-TW (Traditional Chinese), en (English), ja (Japanese)
    """

    SUPPORTED_LANGUAGES = ["zh-TW", "en", "ja"]
    DEFAULT_LANGUAGE = "zh-TW"

    # Character range patterns
    HIRAGANA_PATTERN = re.compile(r"[\u3040-\u309F]")
    KATAKANA_PATTERN = re.compile(r"[\u30A0-\u30FF]")
    KANJI_PATTERN = re.compile(r"[\u4E00-\u9FFF]")
    LATIN_PATTERN = re.compile(r"[a-zA-Z]")

    # Language-specific keywords for confidence boosting
    ZH_TW_KEYWORDS = [
        "台東", "台北", "台中", "台南", "高雄", "天氣", "景點", "住宿", "美食",
        "推薦", "查詢", "怎麼", "什麼", "哪裡", "如何", "謝謝", "請問"
    ]
    EN_KEYWORDS = [
        "weather", "attraction", "hotel", "food", "restaurant", "recommend",
        "where", "what", "how", "taitung", "taipei", "thanks", "please"
    ]
    JA_KEYWORDS = [
        "天気", "観光", "ホテル", "食事", "レストラン", "おすすめ",
        "どこ", "何", "どう", "台東", "台北", "ありがとう", "お願いします"
    ]

    def detect(self, text: str) -> Tuple[str, float]:
        """Detect language from text.

        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or not text.strip():
            return self.DEFAULT_LANGUAGE, 0.5

        text = text.strip()

        # Count character types
        hiragana_count = len(self.HIRAGANA_PATTERN.findall(text))
        katakana_count = len(self.KATAKANA_PATTERN.findall(text))
        kanji_count = len(self.KANJI_PATTERN.findall(text))
        latin_count = len(self.LATIN_PATTERN.findall(text))
        total_chars = len(text)

        # Japanese detection (hiragana/katakana strong signal)
        if hiragana_count > 0 or katakana_count > 0:
            ja_score = (hiragana_count + katakana_count) / max(total_chars, 1)
            # Boost with keywords
            ja_keyword_boost = sum(1 for kw in self.JA_KEYWORDS if kw in text) * 0.1
            confidence = min(1.0, ja_score + ja_keyword_boost)
            return "ja", confidence

        # English detection (latin dominant, few kanji)
        if latin_count > kanji_count * 2 and latin_count > total_chars * 0.3:
            en_score = latin_count / max(total_chars, 1)
            en_keyword_boost = sum(1 for kw in self.EN_KEYWORDS if kw.lower() in text.lower()) * 0.1
            confidence = min(1.0, en_score + en_keyword_boost)
            return "en", confidence

        # Default to zh-TW (kanji present or mixed)
        zh_score = kanji_count / max(total_chars, 1) if total_chars > 0 else 0.5
        zh_keyword_boost = sum(1 for kw in self.ZH_TW_KEYWORDS if kw in text) * 0.1
        confidence = min(1.0, max(0.5, zh_score + zh_keyword_boost))

        return "zh-TW", confidence

    def process(self, graph_state: HarnessGraphState) -> HarnessGraphState:
        """LangGraph node: detects language, updates agent_state."""
        agent_state = graph_state["agent_state"]
        language, confidence = self.detect(agent_state.query)
        agent_state.language = language
        agent_state.metadata["language_detection"] = {
            "detected_language": language,
            "confidence": confidence,
            "method": "heuristic"
        }
        agent_state.update_timestamp()

        new_state = graph_state.copy()
        new_state["agent_state"] = agent_state
        new_state["current_node"] = "language_detection"
        new_state["execution_path"] = graph_state.get("execution_path", []) + ["language_detection"]
        return new_state


def create_language_detector_node():
    """Factory for LangGraph node."""
    detector = LanguageDetector()

    def node(state: HarnessGraphState) -> HarnessGraphState:
        return detector.process(state)

    return node