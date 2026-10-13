# Tiehua Tourism Agent Harness - Architecture Extensions

**Purpose:** Design document for three major feature extensions to align with "Multilingual Tourism AI Agent Architect" requirements.

---

## 1. Multilingual Support & Intent Routing

### 1.1 AgentState Extensions (`src/state.py`)

```python
@dataclass
class AgentState:
    # ... existing 20 fields ...
    
    # NEW: Multilingual fields
    language: str = "zh-TW"          # Detected/selected language: zh-TW, en, ja
    user_intent: str = "unknown"     # Classified intent: weather, attraction, general, mixed
    intent_confidence: float = 0.0   # Confidence score for intent classification
```

### 1.2 Language Detector Node (New: `src/harness/language_detector.py`)

```python
class LanguageDetector:
    """Lightweight language detection for zh-TW, en, ja."""
    
    SUPPORTED_LANGUAGES = ["zh-TW", "en", "ja"]
    
    def detect(self, text: str) -> tuple[str, float]:
        """Returns (language_code, confidence)"""
        # Heuristic: character set analysis + keyword matching
        # For production: integrate fasttext or langdetect
        
    def route_to_node(self, state: AgentState) -> AgentState:
        """Graph node: detects language, updates state, returns next node."""
```

### 1.3 Intent Classifier Node (New: `src/harness/intent_classifier.py`)

```python
class IntentClassifier:
    """Classifies user intent for tool routing."""
    
    INTENTS = ["weather", "attraction", "general", "mixed"]
    
    def classify(self, query: str, language: str) -> tuple[str, float]:
        """Returns (intent, confidence) using keyword + LLM fallback."""
        
    def route_to_node(self, state: AgentState) -> AgentState:
        """Graph node: classifies intent, updates state."""
```

### 1.4 Tool Integration with Language

**Weather Tool (`src/tools/weather.py`):**
- Add `language` parameter to `WeatherInput` schema
- Open-Meteo API supports `language` param (zh, en, ja)
- Return localized weather descriptions

**Attractions Tool (`src/tools/attractions.py`):**
- Add `language` parameter to `AttractionSearchInput`
- Filter/sort results by language preference
- Return localized field values where available

**RAG Retrieval (`src/rag/retrieval.py`):**
- Add `language` filter to `RetrievalConfig.filters`
- Embeddings: use multilingual model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`)
- Filter metadata: `{"language": state.language}`

### 1.5 Graph Integration

```
LanguageDetector → IntentClassifier → AgentExecution → Tools → Evaluation
     (node)            (node)            (existing)      (updated)   (existing)
```

---

## 2. Modular Prompt Template System

### 2.1 Directory Structure

```
prompts/
├── roles/
│   ├── local_guide.md      # 在地嚮導 persona
│   ├── professional_cs.md  # 專業客服 persona
│   └── cultural_expert.md  # 文化專家 persona
├── styles/
│   ├── zh-TW.md            # 繁體中文語氣風格
│   ├── en.md               # English tone/style
│   └── ja.md               # 日本語トーン
├── formats/
│   ├── json_output.md      # JSON輸出護欄
│   ├── markdown_output.md  # Markdown輸出護欄
│   └── structured_text.md  # 結構化文字護欄
├── system/
│   ├── base.md             # 基礎系統提示
│   └── safety.md           # 安全規則注入
├── judge/
│   ├── groundedness.md     # Groundedness 審核
│   └── quality.md          # 品質評分標準
└── feedback/
    ├── retry_hint.md       # 重試提示模板
    └── error_explanation.md # 錯誤解釋模板
```

### 2.2 Prompt Manager (`src/harness/prompt_manager.py`)

```python
class PromptManager:
    """Composes prompt modules based on AgentState."""
    
    def compose_system_prompt(self, state: AgentState) -> str:
        """Build system prompt from role + style + format + safety."""
        role = self._load_role(state.metadata.get("role", "local_guide"))
        style = self._load_style(state.language)
        format_guard = self._load_format("json_output")
        safety = self._load_safety()
        return self._combine([role, style, format_guard, safety])
    
    def compose_retry_prompt(self, state: AgentState, errors: list) -> str:
        """Build retry hint from feedback + error_explanation."""
        
    def compose_judge_prompt(self, state: AgentState, eval_type: str) -> str:
        """Build judge prompt for groundedness/quality."""
        
    def _load_role(self, role_name: str) -> str: ...
    def _load_style(self, language: str) -> str: ...
    def _load_format(self, format_name: str) -> str: ...
    def _load_safety(self) -> str: ...
    def _combine(self, parts: list[str]) -> str: ...
```

### 2.3 Agent Integration

- `AgentCore` uses `PromptManager` instead of hardcoded `system_prompt`
- Prompt selection driven by `state.language`, `state.metadata["role"]`, `state.attempt`

---

## 3. Automated QA & Content Audit Scripts

### 3.1 QA Report Generator (`scripts/generate_qa_report.py`)

```python
class QAReportGenerator:
    """Reads evaluation logs, runs LLM-as-a-Judge, produces quality report."""
    
    def __init__(self, logs_dir: Path, judge_model: str):
        self.logs_dir = logs_dir
        self.judge = LLMJudge(model=judge_model)
    
    def generate_report(self, output_path: Path) -> QAReport:
        # 1. Load all evaluation logs (JSONL)
        # 2. Group by conversation/session
        # 3. For each conversation, run LLM Judge on:
        #    - Groundedness
        #    - Helpfulness
        #    - Tone appropriateness
        #    - Error classification
        # 4. Aggregate statistics
        # 5. Output structured report (JSON + Markdown)
```

**Output Schema (`QAReport`):**
```json
{
  "summary": {
    "total_conversations": 100,
    "pass_rate": 0.85,
    "avg_score": 0.82
  },
  "error_classification": {
    "missing_evidence": 12,
    "tool_error": 8,
    "hallucination": 5,
    "tone_mismatch": 3,
    "safety_violation": 1
  },
  "by_language": {
    "zh-TW": {"pass_rate": 0.88, "count": 60},
    "en": {"pass_rate": 0.80, "count": 25},
    "ja": {"pass_rate": 0.78, "count": 15}
  },
  "by_intent": {
    "weather": {"pass_rate": 0.90, "count": 30},
    "attraction": {"pass_rate": 0.82, "count": 45},
    "general": {"pass_rate": 0.75, "count": 25}
  },
  "retry_analysis": {
    "avg_retries": 0.8,
    "retry_success_rate": 0.65
  },
  "detailed_findings": [...]
}
```

### 3.2 POI Content Auditor (`scripts/audit_poi_content.py`)

```python
class POIContentAuditor:
    """Batch audits attraction data for hallucination, format, freshness."""
    
    def __init__(self, data_path: Path, judge_model: str):
        self.data_path = data_path
        self.judge = LLMJudge(model=judge_model)
    
    def audit(self, output_path: Path) -> AuditReport:
        # 1. Load data/processed/attractions.json
        # 2. For each POI, run LLM checks:
        #    - Hallucination: cross-reference with known facts / web search
        #    - Format consistency: required fields, field types, enum values
        #    - Freshness: last_updated vs current date, source URL accessibility
        # 3. Output structured audit results
```

**Output Schema (`AuditReport`):**
```json
{
  "summary": {
    "total_pois": 6176,
    "passed": 5900,
    "warnings": 250,
    "failures": 26
  },
  "hallucination_findings": [
    {"poi_id": "A001", "field": "description", "issue": "Claims UNESCO status but not listed", "severity": "high"}
  ],
  "format_issues": [
    {"poi_id": "A015", "field": "latitude", "issue": "Invalid coordinate format", "severity": "medium"}
  ],
  "freshness_issues": [
    {"poi_id": "A023", "field": "last_updated", "issue": "Data older than 2 years", "severity": "low"}
  ],
  "by_category": {
    "accommodation": {"pass": 12, "warn": 2, "fail": 1},
    "activity": {"pass": 18, "warn": 3, "fail": 2}
  }
}
```

---

## 4. Folder Structure Changes

```
tiehua-agent-harness/
├── prompts/                          # NEW: Modular prompt system
│   ├── roles/
│   ├── styles/
│   ├── formats/
│   ├── system/
│   ├── judge/
│   └── feedback/
├── scripts/
│   ├── download_tourism_data.py      # EXISTING
│   ├── init-pgvector.sql             # EXISTING
│   ├── generate_qa_report.py         # NEW: QA automation
│   └── audit_poi_content.py          # NEW: POI audit
├── src/
│   ├── state.py                      # MODIFIED: +language, +user_intent
│   ├── harness/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── loop.py
│   │   ├── graph.py
│   │   ├── feedback.py
│   │   ├── retry.py
│   │   ├── result.py
│   │   ├── language_detector.py      # NEW
│   │   ├── intent_classifier.py      # NEW
│   │   └── prompt_manager.py         # NEW
│   ├── tools/
│   │   ├── base.py
│   │   ├── weather.py                # MODIFIED: +language support
│   │   ├── attractions.py            # MODIFIED: +language support
│   │   └── __init__.py
│   ├── rag/
│   │   ├── retrieval.py              # MODIFIED: +language filter
│   │   └── ...
│   ├── agent_core/
│   │   ├── agent_core.py             # MODIFIED: uses PromptManager
│   │   └── ...
│   └── ...
└── DEVELOPMENT_GUIDE.md              # MODIFIED: +Prompt Engineering section
```

---

## 5. Implementation Sequence

### Phase A: State & Detection (Foundation)
1. Extend `AgentState` with `language`, `user_intent`, `intent_confidence`
2. Implement `LanguageDetector` node
3. Implement `IntentClassifier` node
4. Integrate into LangGraph (`graph.py`) as first two nodes

### Phase B: Tool Multilingual Support
1. Update `WeatherInput` + `WeatherTool.execute()` for language
2. Update `AttractionSearchInput` + `AttractionsTool.execute()` for language
3. Update `RetrievalConfig` + `VectorRetriever` for language filtering
4. Update embedding model to multilingual variant

### Phase C: Prompt Template System
1. Create `prompts/` directory structure with template files
2. Implement `PromptManager` in `src/harness/`
3. Integrate into `AgentCore` for dynamic prompt composition
5. Add retry-aware prompt variation (attempt-based)

### Phase D: QA & Audit Scripts
1. Implement `generate_qa_report.py` with LLM Judge integration
2. Implement `audit_poi_content.py` with batch LLM evaluation
3. Add to CI/CD pipeline for automated quality gates

---

## 6. Configuration Additions

**`config/default.yaml` additions:**
```yaml
multilingual:
  supported_languages: ["zh-TW", "en", "ja"]
  default_language: "zh-TW"
  detection_method: "heuristic"  # or "fasttext"
  
prompt:
  roles_dir: "prompts/roles"
  styles_dir: "prompts/styles"
  formats_dir: "prompts/formats"
  default_role: "local_guide"
  
qa_report:
  judge_model: "poolside/laguna-s-2.1:free"
  output_dir: "evaluation/reports"
  
poi_audit:
  judge_model: "poolside/laguna-s-2.1:free"
  output_dir: "evaluation/reports"
  freshness_threshold_days: 730
```

---

## 7. Compatibility Notes

- All changes are **backward compatible** - existing tests should pass
- `AgentState` extensions use default values
- Tool `execute()` signatures unchanged; language passed via input schema
- LangGraph integration: new nodes inserted before `agent_execution`
- PromptManager: optional, falls back to hardcoded prompt if templates missing
- Scripts: standalone, no runtime dependencies on Harness

---

## 8. Validation Criteria

Each feature must have:
- [ ] Unit tests for new classes
- [ ] Integration test in `test_harness.py` (or new test file)
- [ ] Demonstration script showing end-to-end flow
- [ ] Documentation updated in `DEVELOPMENT_GUIDE.md`
- [ ] Configuration examples in `config/default.yaml`