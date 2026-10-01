# Tiehua Tourism Agent Harness — 正式版開發指南

 > **Project:** Tiehua Tourism Agent Harness\
>  **Subtitle:** A Tool-Using Tourism Agent with Evaluation, Structured Feedback and Stateful Retry\
>  **Status:** Formal Development Specification\
>  **Primary Language:** Python 3.11+\
>  **Document:** `DEVELOPMENT_GUIDE.md`

---

 ## 1\. Project Overview

 ### 1.1 專案定位

 Tiehua Tourism Agent Harness 是一個以台東旅遊情境為核心的 **可控制、可驗證、可恢復的 Agent execution environment**。

 系統將 Agent execution 拆分為四個責任層：

 | 層級 | 責任 |
| --- | --- |
| **LLM** | Reasoning、intent interpretation、tool selection、response generation |
| **Tools** | 真實世界 action、API 呼叫、資料查詢 |
| **Evaluator** | 驗證 tool output、response、groundedness 與任務條件 |
| **Harness** | 控制 execution、保存 state、分類錯誤、產生 feedback、執行 retry |

核心原則：

```
LLM 負責 reasoning
Tools 負責 action
Evaluator 負責 verification
Harness 負責 control and recovery
```

 ### 1.2 解決的核心問題

 一般 LLM chatbot 主要追求「產生合理回答」，但 Tool-Using Agent 還需要處理：

 - Tool 是否真的被正確呼叫。
- Tool output 是否有效。
- 回答是否 grounded 在 tool output。
- 外部 API 失敗後如何恢復。
- 第一次 execution 失敗後如何帶著 feedback 重試。
- Retry 是否保留前一次 execution 的 state。
- 如何以 deterministic rules 驗證 agent behavior。
- 如何將 evaluator、LLM judge 與測試 dataset 整合。
- 如何觀察一次完整 execution 的 logs、metrics 與 traces。
- 如何在錯誤、安全性或資料品質問題發生時停止 execution。

 ### 1.3 與一般 Chatbot 的差異

 | 一般 Chatbot | Tiehua Tourism Agent Harness |
| --- | --- |
| 主要輸出文字 | 執行 tool + 驗證 \+ 輸出 |
| 失敗通常直接回答 | failure → classify → feedback → retry |
| State 多為 conversation history | Explicit `AgentState` |
| Tool execution 非核心控制層 | Tool 是受 Harness 控制的 action |
| Quality 多依賴人工判斷 | Deterministic Evaluator \+ LLM Judge |
| Retry 通常無明確 policy | Stateful Retry Policy |
| Debug 依賴 logs | Logs + Metrics + Tracing |
| prompt-centric | execution-contract-centric |
| 不一定可重現 | Dataset + expected behavior \+ seed/config 可重現 |

---

 # 2\. Architecture Design

 ## 2.1 系統架構

```
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Safety / Input    │
                         │      Guardrails      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────────┐
                  │              Harness                │
                  │                                     │
                  │  ┌──────────┐   ┌──────────────┐   │
                  │  │  Runner  │──▶│  Agent Loop  │   │
                  │  └──────────┘   └──────┬───────┘   │
                  │                        │           │
                  │                  ┌─────▼─────┐     │
                  │                  │ Agent/LLM │     │
                  │                  └─────┬─────┘     │
                  │                        │           │
                  │                 Function Calling  │
                  │                        │           │
                  │                  ┌─────▼─────┐     │
                  │                  │   Tools   │     │
                  │                  └─────┬─────┘     │
                  │                        │           │
                  │                  Tool Results      │
                  │                        │           │
                  │              ┌─────────▼────────┐  │
                  │              │    Evaluator     │  │
                  │              │                  │  │
                  │              │ deterministic    │  │
                  │              │ + LLM Judge      │  │
                  │              └─────────┬────────┘  │
                  │                        │           │
                  │               pass / fail        │
                  │                  feedback         │
                  │                        │           │
                  │            ┌───────────▼────────┐  │
                  │            │  Retry Policy      │  │
                  │            └───────────┬────────┘  │
                  │                        │           │
                  │                 retry / stop      │
                  └────────────────────────┼───────────┘
                                           │
                                           ▼
                              ┌──────────────────────┐
                              │ Final Agent Response │
                              └──────────────────────┘

      ┌────────────────────────────────────────────────────────┐
      │ Cross-cutting: Logging / Metrics / Tracing / Config    │
      └────────────────────────────────────────────────────────┘
```

 ## 2.2 Core Execution Loop

```
initialize_state()
       │
       ▼
input guardrails
       │
       ▼
agent execution
       │
       ├── tool call ──▶ tool dispatch ──▶ tool result
       │
       ▼
collect execution result
       │
       ▼
evaluate()
       │
       ├── PASS ───────────────▶ finalize
       │
       └── FAIL
              │
              ▼
       classify failure
              │
              ▼
       build structured feedback
              │
              ▼
       retry policy
              │
        ┌─────┴─────┐
        │           │
      RETRY        STOP
        │           │
        ▼           ▼
  update state    failed result
        │
        └──────────────▶ agent execution
```

 ## 2.3 核心模組職責

 | Module | Responsibility |
| --- | --- |
| `agent` | LLM interaction、function calling、tool dispatch |
| `tools` | External APIs、structured data retrieval |
| `harness` | execution lifecycle、state、retry、feedback |
| `evaluation` | deterministic / LLM evaluation |
| `safety` | input/output/tool safety controls |
| `observability` | logs、metrics、traces |
| `config` | runtime configuration |
| `prompts` | system、agent、judge prompts |
| `evaluation/datasets` | evaluation cases |
| `scripts` | data ingestion / maintenance |

## 2.4 關鍵資料流

 每次 execution 必須形成可追蹤的資料鏈：

```
Request
  ↓
AgentState
  ↓
LLM Request
  ↓
ToolCall
  ↓
ToolResult
  ↓
AgentResponse
  ↓
EvaluationResult
  ↓
Feedback
  ↓
RetryDecision
  ↓
Updated AgentState
```

 每個 execution 必須具有唯一 `run_id`，每次 retry 必須具有 `attempt`。

---

 # 3\. Technology Stack

 | Technology | Purpose | 版本/備註 |
| --- | --- | --- |
| Python | Agent / automation | 3.11+ |
| OpenRouter | LLM API | Agent / Judge model gateway |
| Function Calling | Tool execution | Tool invocation protocol |
| Open-Meteo | Weather API | Geocoding + Forecast |
| RAG | Tourism knowledge | P4 實作 |
| Chroma / PGVector | Vector Store | P4 引入 |
| LangGraph | Agent workflow | Phase 27 可選 |
| pytest | Engineering tests | Test framework |
| LLM-as-a-Judge | Quality evaluation | **免費模型** |
| structlog | Structured logging | JSON-compatible structured logs |
| prometheus-client | Metrics | Prometheus metrics |
| opentelemetry | Distributed tracing | OpenTelemetry |
| pydantic | Schema validation | Runtime/config/data validation |
| pyyaml | Config management | YAML configuration |

### 3.1 Model Configuration

 **Agent**

```
nvidia/nemotron-3-ultra-550b-a55b:free
```

 **Judge**

 Primary:

```
poolside/laguna-s-2.1:free
```

 Fallback:

```
dots-studio/dots-3-note-preview:free
```

 所有 model identifier 必須由 configuration 管理，不得硬編碼於 business logic。

---

 # 4\. Project Structure

 正式版建議目錄：

```
tiehua-agent-harness/
├── README.md
├── DEVELOPMENT_GUIDE.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/                          # 🟡 Planned
│   ├── default.yaml
│   ├── development.yaml
│   ├── test.yaml
│   └── production.yaml
│
├── prompts/                         # 🟡 Planned
│   ├── agent.md
│   ├── system.md
│   ├── judge.md
│   └── feedback.md
│
├── data/
│   ├── raw/                         # 🟡 Planned (CSV source files)
│   └── processed/
│       └── attractions.json         # ✅ Existing (58 records)
│
├── evaluation/                      # 🟡 Planned
│   ├── datasets/
│   │   ├── smoke.json
│   │   ├── regression.json
│   │   └── edge_cases.json
│   ├── expected_behaviors/
│   │   └── tourism.yaml
│   └── reports/
│
├── scripts/
│   ├── download_tourism_data.py     # ✅ Existing
│   ├── run_evaluation.py            # 🟡 Planned
│   └── build_knowledge_base.py      # 🟡 Planned
│
├── src/
│   ├── __init__.py                  # ✅ Existing
│   │
│   ├── agent/                       # 🟡 Planned (currently flat)
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── client.py
│   │   ├── dispatcher.py
│   │   └── schemas.py
│   │
│   ├── tools/                       # ✅ Existing (flat structure)
│   │   ├── __init__.py
│   │   ├── base.py                  # 🟡 Planned
│   │   ├── registry.py              # 🟡 Planned
│   │   ├── weather.py               # ✅ Existing
│   │   ├── attractions.py           # ✅ Existing
│   │   └── schemas.py               # 🟡 Planned
│   │
│   ├── harness/                     # 🟡 Planned (state.py currently in src/)
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── loop.py
│   │   ├── state.py                 # ✅ Existing (src/state.py)
│   │   ├── feedback.py
│   │   ├── retry.py
│   │   └── result.py
│   │
│   ├── evaluation/                  # 🟡 Planned (currently src/evaluator.py)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── evaluator.py
│   │   ├── deterministic.py
│   │   ├── judge.py
│   │   ├── schemas.py
│   │   └── dataset.py
│   │
│   ├── errors/                      # 🟡 Planned
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── classifier.py
│   │   └── models.py
│   │
│   ├── safety/                      # 🟡 Planned
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── input.py
│   │   ├── output.py
│   │   └── tool.py
│   │
│   ├── observability/               # 🟡 Planned
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── tracing.py
│   │
│   ├── rag/                         # 🟡 Planned (P4)
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── retrieval.py
│   │   └── rerank.py
│   │
│   ├── config/                      # 🟡 Planned
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   └── main.py                      # ✅ Existing (entry point)
│
└── tests/
    ├── __init__.py                  # ✅ Existing
    ├── test_agent.py                # ✅ Existing (22 tests)
    ├── test_weather.py              # ✅ Existing (18 tests)
    ├── test_attractions.py          # ✅ Existing (30 tests)
    ├── test_evaluator.py            # ✅ Existing (19 tests)
    ├── test_harness.py              # 🟡 Planned
    ├── test_retry.py                # 🟡 Planned
    ├── test_safety.py               # 🟡 Planned
    ├── test_observability.py        # 🟡 Planned
    └── test_rag.py                  # 🟡 Planned
```

**圖例**：
- ✅ Existing：已存在於程式庫
- 🟡 Planned：正式版規劃目錄，尚未建立

---

 # 5\. 開發階段規劃 P0–P6

 | 階段 | 重點 | 產出 | 預估 |
| --- | --- | --- | --- |
| **P0: 核心閉環** | Error Classification、Retry Policy、Harness Loop | 完整「執行→評估→重試」閉環 | 2–3 週 |
| **P1: 評估體系** | Evaluator Registry、Dataset、LLM Judge、CI | 可信評估系統 | 1–2 週 |
| **P2: 標準化** | BaseTool、LLM Client 抽象、Tools 重構 | 統一介面 | 1 週 |
| **P3: 觀測性** | Structured Logging、Metrics、Tracing | 可觀測系統 | 1 週 |
| **P4: RAG** | Ingestion、Chunking、Embedding、Retrieval、Rerank | 知識庫系統 | 2–3 週 |
| **P5: Safety** | Guardrails Pipeline | 安全防護 | 1 週 |
| **P6: 部署** | Docker、Config、Secrets、Health Check | 生產就緒 | 1 週 |

## 5.1 共通驗收標準

 每一階段完成後必須：

 1. 所有 automated tests 通過。
2. 新增功能具有對應 tests。
3. `DEVELOPMENT_GUIDE.md` 與實際 implementation 同步。
4. Configuration、API contract 與 schema 有文件。
5. 至少有一個可執行 demonstration。
6. 不破壞既有功能。
7. Failure path 必須具有明確行為。
8. Logging / evaluation / retry 等跨模組行為必須可觀察。

 ## 5.2 P0 — 核心閉環

 ### 目標

 建立：

```
Execute → Evaluate → Classify → Feedback → Retry → Execute
```

 ### 必須完成

 - `ErrorType`
- `ClassifiedError`
- `ErrorClassifier`
- `RetryPolicy`
- `HarnessRunner`
- `HarnessLoop`
- `FeedbackBuilder`
- `AgentState` retry metadata

 ### 驗收

 至少涵蓋：

 - tool timeout
- tool structured error
- invalid tool output
- missing required information
- evaluator failure
- groundedness failure
- retry exhaustion

---

 ## 5.3 P1 — 評估體系

 ### 目標

 建立可擴充 evaluator framework：

```
Evaluator Registry
       │
       ├── Deterministic Evaluator
       ├── Tool Evaluator
       ├── Groundedness Evaluator
       └── LLM Judge
```

 ### 必須完成

 - evaluator registry
- dataset schema
- expected behavior schema
- deterministic regression tests
- LLM judge adapter
- evaluation report
- CI evaluation command

---

 ## 5.4 P2 — 標準化

 建立：

 - `BaseTool`
- `ToolRegistry`
- `LLMClient`
- `LLMResponse`
- `ToolResult`
- Pydantic schemas

 所有 tool 必須遵循統一 contract。

---

 ## 5.5 P3 — 觀測性

 必須提供：

 - JSON structured logs
- `run_id`
- `attempt`
- `tool_name`
- `evaluation_id`
- execution duration
- retry count
- tool error count
- evaluator pass rate
- trace/span correlation

---

 ## 5.6 P4 — RAG

 Pipeline：

```
Source
  ↓
Ingestion
  ↓
Normalization
  ↓
Chunking
  ↓
Embedding
  ↓
Vector Store
  ↓
Retrieval
  ↓
Rerank
  ↓
Context
  ↓
Agent
```

 Vector Store：

 - development：Chroma
- production：PGVector

---

 ## 5.7 P5 — Safety

 建立：

```
Input Guardrail
      ↓
Agent
      ↓
Tool Guardrail
      ↓
Tool
      ↓
Output Guardrail
      ↓
Final Response
```

 Safety failure 不得直接進入普通 retry loop；必須區分：

 - recoverable safety issue
- non-recoverable safety issue

---

 ## 5.8 P6 — 部署

 完成：

 - Docker image
- Docker Compose
- production configuration
- secret management
- health check
- readiness check
- Kubernetes baseline manifests
- graceful shutdown

---

 # 6\. 核心模組設計規格

 ## 6.1 Agent — `src/agent/`

 ### 職責

 Agent 負責：

 1. 接收 `AgentState`。
2. 建構 LLM messages。
3. 呼叫 OpenRouter。
4. 處理 function calling。
5. dispatch tool。
6. 收集 tool results。
7. 產生 agent response。

 Agent 不負責：

 - retry policy
- evaluator decision
- error classification
- global execution control

 ### 核心介面

```
from typing import Protocol

class Agent(Protocol):
    def run(self, state: "AgentState") -> "AgentResult":
        ...
```

 ### LLM Client

```
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> "LLMResponse":
        ...
```

 OpenRouter timeout 必須維持明確設定，baseline 為 **30 秒**。

---

 # 6.2 Tools — `src/tools/`

 ## BaseTool Interface Contract

 每一個 tool 必須：

 - 有唯一 `name`
- 有 description
- 有 input schema
- 有 output schema
- 回傳 structured result
- 將 expected external failure 轉換成 domain error
- 不自行執行 retry loop
- 不修改 global AgentState

 ### Interface

```
from typing import Generic, Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class BaseTool(Protocol, Generic[InputT, OutputT]):
    name: str
    description: str

    def schema(self) -> dict:
        ...

    def execute(self, input: InputT) -> OutputT:
        ...
```

 ### Tool Result

```
class ToolResult(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None
    metadata: dict = {}
```

 ### Tool Contract

 Tool failure 不應使用未結構化 exception 作為正常控制流程。

```
HTTP/API failure
       ↓
Tool-specific exception
       ↓
ToolResult(success=False, ...)
       ↓
Harness / ErrorClassifier
```

 ### Weather Tool

 既有 implementation：

 - Open-Meteo Geocoding
- Open-Meteo Forecast
- 中英城市名稱 mapping
- WMO code mapping
- structured errors

 不得將 API response 直接暴露給 Agent；必須轉換成穩定 schema。

 ### Attractions Tool

 既有資料：

 - 台東縣政府開放資料
- 58 筆
- 台東市 district
- `accommodation`
- `activity`
- city/category 雙向匹配
- provenance metadata

 `provenance` 必須保留，供 evaluator 與 RAG groundedness 使用。

---

# 6.3 State — `src/harness/state.py`

`AgentState` 是 Harness 的 canonical execution state。

## 6.3.1 Baseline (現有實作)

`src/state.py` 已存在 `AgentState` dataclass，包含以下 **13 個欄位**：

```python
@dataclass
class AgentState:
    query: str                          # 原始使用者查詢
    messages: list[dict]                # 對話歷史 (system/user/assistant/tool)
    tool_calls: list[dict]              # 工具呼叫記錄
    tool_results: list[dict]            # 工具執行結果
    retrieved_docs: list[dict]          # 未來 RAG 檢索文件
    answer: str | None                  # LLM 最終回答
    evaluation: dict | None             # Evaluator 結果
    feedback: list[str]                 # 結構化回饋
    retry_count: int                    # 重試次數
    last_error: dict | None             # 最近錯誤資訊
    created_at: str                     # ISO 8601 建立時間
    updated_at: str                     # ISO 8601 更新時間
```

**現有方法**：
- `add_message(role, content, **kwargs)` — 記錄訊息
- `add_tool_call(name, arguments, call_id)` — 記錄 tool call
- `add_tool_result(name, arguments, result, call_id)` — 記錄 tool result
- `set_answer(answer)` — 設定最終回答
- `set_evaluation(evaluation)` — 設定評估結果
- `add_feedback(feedback)` — 新增回饋
- `increment_retry()` — 重試計數 +1
- `has_called_tool(name)` — 檢查是否呼叫過某 tool
- `get_latest_tool_result(name)` — 取得最新 tool result
- `to_dict()` / `from_dict()` — 序列化 / 反序列化
- `copy()` — 深拷貝 (供 retry 使用)

## 6.3.2 Formal Version 擴充規劃

正式版需追加以下欄位與機制：

```python
@dataclass
class AgentState:
    # ... 既有 13 個欄位保留 ...
    
    # 正式版追加
    run_id: str                         # 本次 execution 唯一 ID
    attempt: int                        # 當前嘗試次數 (0-indexed)
    max_retries: int                    # 最大重試次數
    status: str                         # execution status: running/passed/failed/fallback
    metadata: dict                      # 任意中繼資料 (model, latency, etc.)
```

**State Invariants**：

1. `run_id` 在整個 execution 不變。
2. 每次 retry `attempt += 1`。
3. `retry_count` 不得超過 `max_retries`。
4. `tool_results` 必須保留歷史 execution context。
5. evaluation result 必須與對應 attempt 可關聯。
6. serialization → deserialization 不得遺失 execution-critical data。
7. retry 必須使用 deep copy 或 immutable transition，避免 mutation leakage。

### State Transition

```
S0 → AgentExecution → S1
S1 → Evaluation → S2
S2 → Feedback → S3
S3 → RetryPolicy
       ├── STOP → Final
       └── RETRY → S4 → AgentExecution
```

---

 # 6.4 Evaluator — `src/evaluation/`

 ## Evaluator Protocol

```
class Evaluator(Protocol):
    name: str

    def evaluate(
        self,
        state: AgentState,
        expected: "ExpectedBehavior | None" = None,
    ) -> "EvaluationResult":
        ...
```

 ## EvaluationResult

```
class EvaluationResult(BaseModel):
    evaluator: str
    passed: bool
    score: float | None = None
    feedback: list[str] = []
    retry: bool = False
    metadata: dict = {}
```

 ## Registry

```
class EvaluatorRegistry:
    def register(self, evaluator: Evaluator) -> None:
        ...

    def get(self, name: str) -> Evaluator:
        ...

    def all(self) -> list[Evaluator]:
        ...
```

 ### Evaluation Pipeline

```
EvaluatorRegistry
      │
      ├── weather correctness
      ├── attraction correctness
      ├── rain condition
      ├── groundedness
      ├── tool behavior
      └── LLM Judge
```

 ### Baseline Deterministic Evaluator v0.2

 目前 baseline：

 - 5 項 checks
- pass/fail
- feedback
- retry decision
- `tests/test_evaluator.py` 19 tests

 既有測試涵蓋：

 - weather
- attraction
- rain
- groundedness

 後續 evaluator 必須以 plugin 形式擴充，而不是將所有規則集中在單一大型 evaluator。

---

 # 6.5 Harness — `src/harness/`

 ## Runner

```
class HarnessRunner:
    def run(self, query: str) -> "HarnessResult":
        ...
```

 Runner 負責一次完整 execution。

 ## Loop

```
class HarnessLoop:
    def execute(self, state: AgentState) -> AgentState:
        ...
```

 ### Feedback

 Feedback 必須是 structured data，而不是單純文字。

```
class Feedback(BaseModel):
    source: str
    code: str
    message: str
    severity: str
    actionable: bool
    metadata: dict = {}
```

 Feedback 必須能直接轉換為下一次 agent execution 的 context。

 ### Retry Policy

```
class RetryDecision(str, Enum):
    RETRY = "retry"
    STOP = "stop"

class RetryPolicy:
    def decide(
        self,
        state: AgentState,
        errors: list["ClassifiedError"],
    ) -> RetryDecision:
        ...
```

 Retry policy 必須考慮：

 - retry count
- error type
- evaluator retry flag
- error recoverability
- safety status
- timeout budget

---

 # 6.6 Error Classification

 ## ErrorType

 標準化：

```
class ErrorType(str, Enum):
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_INVALID_INPUT = "tool_invalid_input"
    TOOL_INVALID_OUTPUT = "tool_invalid_output"
    LLM_TIMEOUT = "llm_timeout"
    LLM_ERROR = "llm_error"
    EVALUATION_FAILURE = "evaluation_failure"
    GROUNDEDNESS_FAILURE = "groundedness_failure"
    POLICY_FAILURE = "policy_failure"
    SAFETY_FAILURE = "safety_failure"
    UNKNOWN = "unknown"
```

 ## ClassifiedError

```
class ClassifiedError(BaseModel):
    error_type: ErrorType
    message: str
    recoverable: bool
    retryable: bool
    source: str
    metadata: dict = {}
```

 ## Classifier

```
class ErrorClassifier:
    def classify(self, error: Exception | ToolResult) -> ClassifiedError:
        ...
```

 ### Classification 原則

 | Error | Recoverable | Default |
| --- | --- | --- |
| Tool timeout | Yes | Retry |
| Temporary API unavailable | Yes | Retry |
| Invalid user input | Usually No | Stop / clarify |
| Invalid tool output | Conditional | Retry |
| Evaluation failure | Conditional | Retry |
| Groundedness failure | Yes | Retry |
| Safety failure | Usually No | Stop |
| Unknown error | No | Stop |

---

 # 6.7 Safety — `src/safety/`

 ## Guardrails Pipeline

```
class Guardrail(Protocol):
    def check(self, context: dict) -> "GuardrailResult":
        ...
```

```
class GuardrailsPipeline:
    def run(self, context: dict) -> "GuardrailResult":
        ...
```

 ### 三個 boundary

 1. **Input Guardrail**
2. **Tool Guardrail**
3. **Output Guardrail**

 ### Safety 原則

 - Guardrail failure 必須可觀察。
- Safety failure 必須具有明確 error code。
- 不得透過一般 retry 無限重試 safety failure。
- Tool arguments 必須在 dispatch 前驗證。
- Final response 必須在輸出前經過必要 validation。

---

 # 6.8 Observability — `src/observability/`

 ## Structured Logging

 使用 `structlog`。

 每個 log event 至少包含：

```
{
  "event": "tool_execution",
  "run_id": "run-123",
  "attempt": 1,
  "tool_name": "weather",
  "duration_ms": 421,
  "success": true
}
```

 禁止只記錄：

```
something failed
```

 而沒有 context。

 ## Metrics

 使用 `prometheus-client`。

 至少提供：

```
agent_runs_total
agent_run_failures_total
agent_retry_total
tool_calls_total
tool_failures_total
tool_duration_seconds
evaluation_total
evaluation_failures_total
evaluation_duration_seconds
```

 ## Tracing

 使用 OpenTelemetry。

 推薦 span：

```
agent.run
├── llm.complete
├── tool.weather
├── tool.attractions
├── evaluator.deterministic
├── evaluator.llm_judge
└── retry.decision
```

---

 # 7\. 關鍵設計模式

 ## 7.1 Tool Evaluator Plugin Architecture

 Tool-specific evaluation 不應硬編碼於 Agent。

```
class ToolEvaluator:
    def supports(self, tool_name: str) -> bool:
        ...

    def evaluate(
        self,
        tool_result: ToolResult,
        expected: ExpectedBehavior,
    ) -> EvaluationResult:
        ...
```

 Registry：

```
registry.register(WeatherEvaluator())
registry.register(AttractionEvaluator())
```

 新增 tool 時新增 evaluator plugin，不修改核心 Harness loop。

---

 ## 7.2 ExpectedBehavior 規格驅動評估

 Expected behavior 是 evaluator 的 contract。

```
class ExpectedBehavior(BaseModel):
    required_tools: list[str] = []
    forbidden_tools: list[str] = []
    required_facts: list[str] = []
    required_conditions: list[str] = []
    must_be_grounded: bool = False
    expected_outcome: str | None = None
```

 範例：

```
id: weather-rain-001

query: "台東明天下雨嗎？"

expected_behavior:
  required_tools:
    - weather
  required_conditions:
    - forecast_available
    - rain_probability_reported
  must_be_grounded: true
```

 Evaluator 根據規格判斷，不直接猜測「理想回答」。

---

 ## 7.3 Stateful Retry

 錯誤發生後，不建立一個完全新的 conversation。

 應使用：

```
State(t)
  ↓
Evaluation
  ↓
Feedback
  ↓
State(t+1)
```

 範例：

```
state.feedback.append(
    Feedback(
        source="groundedness",
        code="MISSING_TOOL_EVIDENCE",
        message="Weather claim is not supported by the weather tool result.",
        severity="error",
        actionable=True,
    )
)

state.retry_count += 1
state.attempt += 1
```

 下一次 execution 必須能看到：

 - previous tool results
- previous evaluation
- feedback
- retry count
- original user query

 ### Retry 不得發生

```
while True:
    retry()
```

 必須有明確上限：

```
while state.retry_count < state.max_retries:
    ...
```

---

 # 8\. Configuration Management

 ## 8.1 Directory

```
config/
├── default.yaml
├── development.yaml
├── test.yaml
└── production.yaml
```

 ## 8.2 Configuration Priority

 由高至低：

```
Runtime arguments
    >
Environment variables
    >
Environment-specific YAML
    >
default.yaml
    >
Code defaults
```

 Secret 不得寫入 YAML。

 ## 8.3 `.env.example`

```
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

AGENT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

JUDGE_PRIMARY_MODEL=poolside/laguna-s-2.1:free
JUDGE_FALLBACK_MODEL=dots-studio/dots-3-note-preview:free

OPENROUTER_TIMEOUT_SECONDS=30

LOG_LEVEL=INFO
ENVIRONMENT=development

VECTOR_STORE=chroma
CHROMA_PATH=./data/chroma
```

 `.env` 必須加入 `.gitignore`。

---

 # 9\. Evaluation Dataset 格式

 ## 9.1 JSON Schema

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Tiehua Evaluation Case",
  "type": "object",
  "required": [
    "id",
    "query",
    "expected_behavior"
  ],
  "properties": {
    "id": {
      "type": "string"
    },
    "query": {
      "type": "string"
    },
    "expected_behavior": {
      "type": "object",
      "properties": {
        "required_tools": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "forbidden_tools": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "required_facts": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "required_conditions": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "must_be_grounded": {
          "type": "boolean"
        },
        "expected_outcome": {
          "type": [
            "string",
            "null"
          ]
        }
      }
    }
  }
}
```

 ## 9.2 Dataset 範例

```
{
  "id": "weather-001",
  "query": "台東明天會下雨嗎？",
  "expected_behavior": {
    "required_tools": [
      "weather"
    ],
    "forbidden_tools": [],
    "required_facts": [
      "forecast"
    ],
    "required_conditions": [
      "weather_data_available"
    ],
    "must_be_grounded": true,
    "expected_outcome": "回答必須依據 weather tool 的預報資料"
  }
}
```

---

 # 10\. Judge Prompt 設計

 檔案：

```
prompts/judge.md
```

 ## 10.1 Prompt Contract

 Judge 必須收到：

 1. User query
2. Agent final response
3. Tool calls
4. Tool results
5. Expected behavior
6. Evaluation criteria

 Judge 不應自行取得外部資料。

 ## 10.2 `prompts/judge.md`

```
# Role

You are the quality judge for Tiehua Tourism Agent Harness.

Your task is to evaluate whether an agent execution satisfies the
provided expected behavior and whether the final answer is grounded
in the available tool results.

# Rules

1. Evaluate only the provided execution evidence.
2. Do not invent external facts.
3. Distinguish missing evidence from incorrect evidence.
4. A response is not grounded when it makes factual claims that are
   unsupported by the supplied tool results.
5. Follow the expected behavior specification.
6. Return structured JSON only.

# Input

## User Query
{{user_query}}

## Expected Behavior
{{expected_behavior}}

## Tool Calls
{{tool_calls}}

## Tool Results
{{tool_results}}

## Agent Response
{{agent_response}}

# Output Schema

{
  "passed": true,
  "score": 0.0,
  "issues": [
    {
      "code": "string",
      "severity": "info|warning|error",
      "message": "string"
    }
  ],
  "retry": false,
  "reason": "string"
}
```

 ## 10.3 Judge 原則

 LLM Judge 是輔助 evaluator，不得取代 deterministic checks。

 優先順序：

```
Deterministic Validation
        ↓
LLM Judge
        ↓
Combined Evaluation Result
```

 Judge model 必須使用免費模型 configuration。

---

 # 11\. Deployment 指南

 ## 11.1 Dockerfile

 正式 image 應：

 - 使用 Python 3.11+
- 非 root user
- 安裝 pinned dependencies
- 不包含 secrets
- 提供 health endpoint

 基本結構：

```
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src ./src
COPY config ./config
COPY prompts ./prompts
COPY data ./data

RUN useradd --create-home appuser
USER appuser

CMD ["python", "-m", "src.main"]
```

 實際 production build 應依 `pyproject.toml` 的 dependency configuration 調整。

---

 ## 11.2 Docker Compose

 基本服務：

```
services:
  app:
    build: .
    env_file:
      - .env
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  chroma:
    image: chromadb/chroma
```

 Production 不得依賴 Compose 中的 hard-coded secrets。

---

 ## 11.3 Kubernetes Baseline

 至少提供：

```
Deployment
Service
ConfigMap
Secret
ReadinessProbe
LivenessProbe
```

 基本 lifecycle：

```
Pod start
   ↓
load config
   ↓
initialize dependencies
   ↓
readiness = true
   ↓
serve requests
   ↓
SIGTERM
   ↓
graceful shutdown
```

---

 ## 11.4 Health Check

 ### `/health`

 檢查 process 是否存活。

 ### `/ready`

 檢查：

 - configuration valid
- required dependencies initialized
- required model configuration exists

 External API 的短暫不可用不應直接使 liveness failure。

---

 # 12\. 開發原則 — Must Follow

 1. **LLM 不負責 execution control；Harness 才負責 execution control。**
2. **Tool 不得自行決定全域 retry policy。**
3. **所有 Tool input/output 必須具有明確 schema。**
4. **External failure 必須轉換為 structured error。**
5. **所有 retry 必須有上限。**
6. **Retry 必須保留 execution state，不得無條件重建 context。**
7. **Evaluator 必須與 Agent implementation 解耦。**
8. **Deterministic checks 優先於 LLM Judge。**
9. **LLM Judge 不得被視為唯一 correctness authority。**
10. **每次 execution 必須具有 `run_id`，每次 retry 必須具有 `attempt`。**
11. **所有 production configuration 必須由 configuration layer 管理，不得散落於 business logic。**
12. **Secrets 不得進入 source code、Git history、logs 或 evaluation dataset。**
13. **每個新增 failure mode 必須新增對應測試。**
14. **每個 phase 完成後必須同步文件、tests 與 executable demonstration。**
15. **不可為了讓測試通過而降低 evaluator 的實際驗證強度。**

---

 # 13\. Git Commit 規範

 採用 Conventional Commits：

```
<type>(<scope>): <description>
```

 ## 13.1 Types

 | Type | 用途 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | Bug fix |
| `refactor` | 重構 |
| `test` | 測試 |
| `docs` | 文件 |
| `chore` | 維護 |
| `perf` | 效能 |
| `build` | Build / dependency |
| `ci` | CI/CD |

## 13.2 範例

```
feat(harness): add stateful retry loop
```

```
feat(evaluation): add evaluator registry
```

```
fix(weather): classify forecast timeout correctly
```

```
test(evaluator): cover groundedness failure
```

```
docs: update development guide for P4
```

 ## 13.3 Commit 原則

 - 一個 commit 應代表一個邏輯變更。
- 不將 unrelated refactor 混入 feature commit。
- 測試應與其功能變更同步提交。
- 不提交 secrets、`.env`、cache 或 generated artifacts。
- Commit message 必須描述「變更內容」，而非實作過程。

---

 # 14\. 目前完成狀態

 ## 14.1 Baseline

 | Component | Status | 說明 |
| --- | --- | --- |
| Agent 主流程 | ✅ | `src/agent.py` |
| Function Calling | ✅ | 已完成 |
| Tool dispatch | ✅ | 已完成 |
| OpenRouter client | ✅ | timeout = 30s |
| AgentState | ✅ | 13 欄位、serialization、deep copy、query helpers |
| Weather Tool | ✅ | Open-Meteo、mapping、WMO、structured errors |
| Attractions Tool | ✅ | 58 筆、台東市、雙向匹配、provenance |
| Tourism JSON | ✅ | `data/processed/attractions.json` |
| Tourism data script | ✅ | CSV → JSON |
| Deterministic Evaluator v0.2 | ✅ | 5 checks |
| Evaluator tests | ✅ | 19 tests |
| Agent tests | ✅ | 22 tests |
| Weather tests | ✅ | 18 tests |
| Attractions tests | ✅ | 30 tests |
| **Baseline total** | **✅** | **89 tests 全通** |

## 14.2 Formal Architecture Status

 | Component | Status | 下一步 |
| --- | --- | --- |
| P0 Core Harness Loop | 🟡 | Runner、Loop、Retry Policy、Error Classification |
| P1 Evaluation Registry | 🟡 | Registry、Dataset、Judge、CI |
| P2 Standardized Tool API | 🟡 | `BaseTool`、LLM abstraction |
| P3 Observability | 🟡 | structlog、Prometheus、OpenTelemetry |
| P4 RAG | 🟡 | ingestion → retrieval → rerank |
| P5 Safety | 🟡 | Guardrails pipeline |
| P6 Deployment | 🟡 | Docker、Compose、K8s、Health Check |
| LLM Judge | 🟡 | 免費模型整合 |
| Chroma / PGVector | 🟡 | P4 |
| LangGraph | 🟡 | Phase 27 可選 |
| Production readiness | ❌ | 待 P0–P6 驗收 |

### 14.3 Status 定義

 - **✅ Complete**：功能與 tests 均完成，符合既定 contract。
- **🟡 In Progress / Planned**：架構或部分基礎已存在，但尚未達正式版驗收標準。
- **❌ Not Ready**：尚未實作或尚未具備正式版所需驗收條件。

---

 # Appendix A — Definition of Done

 一項功能只有在以下條件全部成立時才視為完成：

```
Implementation
    +
Schema / Contract
    +
Unit Tests
    +
Failure Tests
    +
Observability
    +
Documentation
    +
Executable Demonstration
    =
Done
```

 對 Harness 核心功能，另外必須驗證：

```
Normal Path
    +
Tool Failure
    +
Evaluation Failure
    +
Retry Path
    +
Retry Exhaustion
    +
Final Failure State
```

---

 # Appendix B — Formal Execution Contract

 一次完整 execution 必須遵循：

```
1. Create run_id
2. Initialize AgentState
3. Run input guardrails
4. Execute Agent
5. Dispatch tools
6. Record tool results
7. Generate final response
8. Run deterministic evaluators
9. Run LLM Judge when configured
10. Aggregate evaluation results
11. Classify errors
12. Build structured feedback
13. Apply RetryPolicy
14. Retry with updated AgentState when allowed
15. Finalize success or failure
16. Emit logs, metrics and traces
```

 核心 invariant：

```
No evaluation → no trusted success
No retry limit → invalid harness
No state preservation → invalid stateful retry
No structured tool result → invalid tool contract
No observable run_id → invalid production execution
```

---

 # Appendix C — Formal Quality Model

 Tiehua Tourism Agent Harness 的 execution quality 應拆解為：

```
Execution Quality
├── Tool Correctness
├── Tool Reliability
├── State Integrity
├── Evaluation Correctness
├── Groundedness
├── Recovery Behavior
├── Safety Compliance
└── Observability
```

 不以單一 LLM score 取代上述維度。

 正式版的核心目標不是讓 Agent 「看起來更聰明」，而是讓每次 execution 都具有：

```
可控制
    +
可驗證
    +
可觀察
    +
可恢復
    +
可重現
```

 這五項能力構成 Tiehua Tourism Agent Harness 的正式版工程基準。