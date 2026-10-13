# 🌿 Tiehua Tourism Agent Harness

### A Tool-Using Tourism Agent with Evaluation and Stateful Retry

> **LLM 負責 reasoning，Tools 負責 action，Evaluator 負責 verification，Harness 負責 control and recovery。**

## 🎯 Project

Tiehua Tourism Agent Harness 是一個以 **Harness Engineering** 為核心的旅遊 AI Agent。

與一般 Chatbot 不同，本專案讓 Agent 能夠：

* 使用 Tools 執行任務
* 使用 RAG 取得旅遊知識
* 評估自己的執行結果
* 根據 Structured Feedback 修正
* 在限制次數內 Retry
* 發生錯誤時進行 Re-plan / Re-tool / Re-retrieve

核心目標：

> **讓 LLM 不只是產生答案，而是在一個可控制、可驗證、可恢復的環境中完成任務。**

---

## 🏗️ Architecture

```text
User
 ↓
Agent Harness
 ↓
LLM
 ↓
Tools / RAG
 ↓
Result
 ↓
Evaluator
 ↓
Structured Feedback
 ↓
Error Classification
 ↓
Retry Policy
 ↓
Updated State
 ↓
Re-plan / Re-tool / Re-retrieve
 ↓
Final Answer
```

---

## 🔧 Tools

目前主要 Tools：

```text
get_weather(city)
search_attractions(city, category)
```

### Weather

使用 **Open-Meteo API** 取得即時天氣資訊。

### Tourism

從結構化旅遊資料 / RAG Knowledge Base 搜尋景點。

---

## 📚 RAG

旅遊資料經過：

```text
Raw Data
 ↓
Cleaning
 ↓
Chunking
 ↓
Embedding
 ↓
Vector Store
 ↓
Retrieval
 ↓
Agent
```

資料包含：

* 景點
* 美食
* 住宿
* 交通
* 活動
* 分類與標籤
* 資料來源

目的：

> **讓 Agent 的回答建立在可檢索的外部資料上。**

---

## 🔄 Evaluation & Retry

Agent 產生結果後，不直接視為成功。

```text
Agent Result
 ↓
Evaluator
 ↓
PASS → Final Answer

FAIL
 ↓
Structured Feedback
 ↓
Error Classification
 ↓
Retry Policy
 ↓
Updated State
 ↓
Retry
```

例如：

```json
{
  "status": "fail",
  "error_type": "missing_evidence",
  "retry_strategy": "retrieve_again",
  "feedback": [
    "缺少雨天適合性的證據"
  ]
}
```

Retry 不是重新開始，而是：

> **Previous State → Evaluation → Feedback → Updated State → Retry**

並透過 `MAX_RETRIES` 避免無限循環。

---

## 🧪 Evaluation

使用兩種方式測試 Agent：

### Deterministic Tests

使用 `pytest` 測試：

* Tool execution
* Tool parameters
* API errors
* Retry limit
* State management
* Safety rules

### Agent Evaluation

建立固定 Dataset，測試：

* Tool Routing
* Weather
* Tourism
* RAG
* Groundedness
* Completeness
* Safety

必要時使用 **LLM-as-a-Judge** 進行品質評估。

---

## 🛡️ Safety

Harness 同時處理：

* Prompt Injection
* Hallucination
* Invalid Tool Arguments
* Tool/API Failure
* Unsupported Claims
* Retry Abuse

Safety 不只依賴 Prompt，而是放在 Agent execution flow 中。

---

## 📁 Project Structure

```text
tiehua-agent-harness/
│
├── README.md
├── main.py
├── requirements.txt
│
├── src/
│   ├── agent.py
│   ├── agent_core/
│   ├── tools/
│   ├── rag/
│   ├── evaluation/
│   ├── harness/
│   ├── safety/
│   ├── errors/
│   ├── observability/
│   └── state.py
│
├── data/
├── prompts/
├── evaluation/
└── tests/
```

---

## 🛠️ Tech Stack

```text
Python
OpenRouter
Function Calling
Open-Meteo
RAG
Chroma
LangGraph
pytest
LLM-as-a-Judge
```

---

## 💡 Core Principle

傳統 Prompt Engineering：

```text
如何讓 LLM 回答得更好？
```

Harness Engineering：

```text
如何讓 Agent 在 LLM 不可靠的情況下，
仍然能夠可靠地完成任務？
```

因此本專案的核心是：

> **LLM 負責 intelligence，Harness 負責 control。**

**From Prompt Engineering → Agent Engineering → Harness Engineering**
