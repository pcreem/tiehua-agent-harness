"""Project documentation for Tiehua Tourism Agent Harness.

This file provides a comprehensive guide to understanding the codebase,
including architecture, key components, data flow, and known issues.
"""

# Tiehua Tourism Agent Harness - 專案文檔

## 專案概覽

Tiehua Tourism Agent Harness 是一個多語系旅遊 AI Agent 框架，採用 **Harness 控制迴路** 架構：
- **LLM 推理** → **Tool 執行** → **Evaluator 驗證** → **Harness 失敗處理/重試**

支援語言：繁體中文 (zh-TW)、英文 (en)、日文 (ja)

---

## 核心檔案導覽

### 📐 架構與設計文件
| 檔案 | 用途 |
|------|------|
| `ARCHITECTURE_EXTENSIONS.md` | 三大擴充架構設計：多語系/意圖路由、模組化 Prompt、QA 腳本 |
| `DEVELOPMENT_GUIDE.md` | 開發規範、Prompt Engineering 最佳實踐、Git/安全/觀測指南 |

### 🧠 核心執行流程 (Harness)
| 檔案 | 關鍵功能 |
|------|----------|
| `src/harness/graph.py` | **LangGraph 工作流**：`language_detection → intent_classification → agent_execution → evaluation → error_classification → feedback_builder → retry_decision → finalize` |
| `src/harness/graph_types.py` | LangGraph 共享狀態 schema (`HarnessGraphState`) |
| `src/harness/runner.py` | `HarnessGraphRunner` - 對外執行介面、`run()` / `run_streaming()` |
| `src/state.py` | `AgentState` - 單一事實來源 (含 `language`、`user_intent`、`intent_confidence`) |
| `src/harness/language_detector.py` | 啟發式語言偵測 (zh-TW/en/ja) |
| `src/harness/intent_classifier.py` | 關鍵字+LLM 意圖分類 (weather/attraction/general/mixed) |

### 🤖 Agent 核心
| 檔案 | 關鍵功能 |
|------|----------|
| `src/agent_core/agent_core.py` | `AgentCore` - 動態 Prompt 組裝、tool calls、state 語言注入 |
| `src/agent_core/tool_executor.py` | Tool 安全驗證、執行、錯誤正規化、observability |
| `src/agent_core/llm_client.py` | LLM 呼叫封裝 (OpenRouter) |
| `src/harness/prompt_manager.py` | **模組化 Prompt 組合**：role + style(語言) + format + safety，SHA-256 版本追蹤 |

### 🔧 Tools
| 檔案 | 功能 |
|------|------|
| `src/tools/weather.py` | 多語 Weather Tool (Open-Meteo API，支援 zh/en/ja) |
| `src/tools/attractions.py` | 多語景點搜尋 (本地 6,176 筆資料，22 縣市、6 類別) |
| `src/tools/__init__.py` | 自動註冊機制 |

### 📚 RAG 系統
| 檔案 | 功能 |
|------|------|
| `src/rag/ingestion.py` | 文件攝取管線 (chunking → embedding → vectorstore) |
| `src/rag/retrieval.py` | Vector/Hybrid/Reranked 檢索策略 |
| `src/rag/embeddings.py` | Embedding Factory (多模型支援) |
| `src/rag/vectorstore.py` | Chroma/pgvector 向量庫抽象層 |
| `src/rag/chunking.py` | 文本切分策略 |
| `src/rag/types.py` | 核心資料結構 (Document, Chunk, RetrievalResult 等) |

### 🛡️ 安全 Guardrails
| 檔案 | 功能 |
|------|------|
| `src/safety/prompt_injection.py` | Prompt Injection 偵測 (多訊號加權) |
| `src/safety/content_filter.py` | 內容過濾 (detoxify) |
| `src/safety/tool_guard.py` | Tool 參數白名單驗證 |
| `src/safety/pipeline.py` | 三階段 pipeline (input/tool/output) + metrics |

### 📊 評估與自動化
| 檔案 | 功能 |
|------|------|
| `src/evaluation/llm_judge.py` | LLM-as-a-Judge (groundedness/quality) |
| `src/evaluation/deterministic.py` | 確定性評估器 (required_tools/must_be_grounded 等) |
| `scripts/generate_qa_report.py` | QA 報表生成 (依語言/意圖統計 pass rate) |
| `scripts/audit_poi_content.py` | POI 稽核 (幻覺/格式/時效) |
| `evaluation/dataset_schema.json` | 評估 case schema 定義 |
| `evaluation/expected_behaviors/tourism.yaml` | 預期行為規則 (依類別定義 required_tools 等) |

### 🖥️ 介面
| 檔案 | 功能 |
|------|------|
| `main.py` | FastAPI 進入點 (port 8000) |
| `dashboard/app.py` | Evaluation Dashboard (Streamlit, port 8501) |
| `dashboard/chat.py` | 聊天介面 + 即時統計 (Streamlit, port 8502) |

### 📁 資料
| 檔案 | 內容 |
|------|------|
| `data/processed/attractions.json` | 相容舊工具格式 (7.9 MB, 6,176 筆) |
| `data/processed/attractions_v2.json` | **完整 schema v2** (18.5 MB) - 含多語、provenance、quality、attributes |
| `data/Attraction-json/` | 官方原始資料 (AttractionList/FeeList/ServiceTimeList) |
| `prompts/` | 模組化 Prompt 模板 (roles/styles/formats/system/judge/feedback) |

---

## 關鍵概念補充

### 1. State 流向
```
User Query 
  → language_detection (更新 state.language)
  → intent_classification (更新 state.user_intent, state.intent_confidence)  
  → agent_execution (PromptManager 依 state.language 組裝 system prompt)
  → Tool 呼叫 (ToolExecutor 驗證 → 執行 → 回傳 ToolResult)
  → evaluation (Deterministic + LLM Judge)
  → error_classification / feedback_builder / retry_decision
  → finalize → 回傳 AgentState
```

### 2. 多語系機制
- **偵測**：`LanguageDetector` 啟發式 (字元集 + 旅遊關鍵字)
- **Prompt 選擇**：`PromptManager._load_style(state.language)` 載入 `prompts/styles/{zh-TW,en,ja}.md`
- **Tool 參數**：Weather/Attractions 皆接受 `language` 參數
- **回應語言**：system prompt 指示「回答語言跟隨使用者查詢語言」

### 3. 資料 Schema v2 重點欄位
```json
{
  "id": "Attraction_xxx",
  "names": {"zh-TW": "...", "en": "", "ja": ""},
  "class_names": {"zh-TW": "森林遊樂區、自然風景區", "en": "...", "ja": "..."},
  "category": "nature",
  "all_categories": ["nature", "sightseeing"],
  "location": {
    "latitude": 24.55, "longitude": 121.49,
    "city": "宜蘭縣", "district": "大同鄉",
    "geo_precision": "address"
  },
  "description": {"zh-TW": "...", "en": "", "ja": ""},
  "attributes": {
    "fees": [{"name": "全票", "price": 80, ...}],
    "service_times": [{"Name": "平日", "StartTime": "06:00", "EndTime": "20:00"}],
    "parking": "...", "traffic_info": "...", "phone_numbers": [...]
  },
  "provenance": {"source": "交通部觀光署", "provider_id": "A15010000H", ...},
  "quality": {"confidence": 0.85, "has_coordinates": true, ...},
  "i18n_status": {"zh-TW": "complete", "en": "missing", "ja": "missing"}
}
```

### 4. 已知技術債 / 面試必備誠實點
| 項目 | 現況 | 解決方向 |
|------|------|----------|
| RAG 多語檢索 | Schema 有 `language` filter，但未接上 `state.language`；嵌入模型仍單語 | 下載官方 En/Ja 版資料 → 換多語 embedding → 開 filter |
| 評估資料集 | 僅 20 cases (smoke 10 + edge 5 + regression 5) | 擴充到 3語言×4意圖×3難度矩陣 |
| 11 個 agent 測試失敗 | `test-key` 打真實 API → 401 | 改用 mock LLM response |
| `/metrics` 端點 | 掛預設 registry，非自訂 | 接入自訂 Prometheus metrics |
| Chat 統計 | 僅 session state，重整即失效 | 持久化 JSONL |
| `generate_qa_report.py` | 未端到端驗證 LLMJudge 介面 | 驗證 `evaluate_conversation()` 方法 |

### 5. 啟動指令
```bash
# FastAPI
uvicorn main:app --host 127.0.0.1 --port 8000

# Evaluation Dashboard
streamlit run dashboard/app.py --server.port 8501

# Chat Interface
streamlit run dashboard/chat.py --server.port 8502

# 測試
pytest tests/test_attractions.py -v
pytest tests/ -v  # 11 個 agent 測試會因 401 失敗 (預期)
```

### 6. 向量庫重建
```bash
python scripts/rebuild_vectorstore_local.py
# 使用 intfloat/multilingual-e5-large 本地嵌入，避開 OpenRouter 限制
# 約 30-60 分鐘處理 6,176 筆
```

---

## Debug 記錄：向量庫重建腳本修復

### 問題
執行 `python scripts/rebuild_vectorstore_local.py` 時發生錯誤：
```
AttributeError: 'Document' object has no attribute 'document_id'
```

### 根因
`vector_store.add_chunks()` 期望接收 `Chunk` 物件列表，而非 `Document` 物件。`Chunk` 有 `document_id`、`start_char`、`end_char` 等欄位，需經過 chunking 步驟從 `Document` 轉換而來。

### 修復方案
使用 `IngestionPipeline` 進行完整的 ingestion 流程：Document → Chunker → Embedder → VectorStore。

### 修正後的腳本
```python
#!/usr/bin/env python3
"""
Rebuild vectorstore from attractions_v2.json (6,176 records).

Uses local multilingual embedding model (sentence-transformers) to avoid API limits.
Properly uses IngestionPipeline for Document -> Chunk -> VectorStore flow.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.ingestion import IngestionPipelineBuilder
from rag.types import Document, ChunkConfig, EmbeddingConfig, VectorStoreConfig
from sentence_transformers import SentenceTransformer


# Use multilingual model for cross-language support
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # 1024 dim, supports 100+ languages


class LocalEmbedder:
    """Local embedding using sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        print(f"Model loaded. Dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # E5 models need "query: " or "passage: " prefix for best results
        prefixed_texts = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(prefixed_texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()


def attractions_to_documents(attractions: List[Dict[str, Any]]) -> List[Document]:
    """Convert attraction records to Documents for vectorstore."""
    documents = []

    for attr in attractions:
        # Build searchable text combining key fields
        parts = []

        parts.append(f"名稱：{attr['names']['zh-TW']}")
        if attr['description']['zh-TW']:
            parts.append(f"描述：{attr['description']['zh-TW']}")

        parts.append(f"類別：{attr['category']}")
        if attr['class_names']['zh-TW']:
            parts.append(f"分類：{attr['class_names']['zh-TW']}")

        loc = attr['location']
        parts.append(f"地址：{loc['address']['zh-TW']}")
        parts.append(f"城市：{loc['city']}")
        if loc['district']:
            parts.append(f"行政區：{loc['district']}")

        attrs = attr['attributes']
        if attrs.get('fees'):
            fee_text = "、".join([
                f"{f.get('Name', f.get('name', ''))}{f.get('Price', f.get('price', ''))}元"
                if f.get('Price') or f.get('price')
                else f.get('Name', f.get('name', ''))
                for f in attrs['fees']
            ])
            parts.append(f"票價：{fee_text}")

        if attrs.get('service_times'):
            st_text = "、".join([
                f"{st.get('Name', st.get('name', ''))}{st.get('StartTime', st.get('startTime', ''))}-{st.get('EndTime', st.get('endTime', ''))}"
                if st.get('StartTime') or st.get('startTime')
                else st.get('Description', st.get('description', ''))
                for st in attrs['service_times']
            ])
            parts.append(f"營業時間：{st_text}")

        if attrs.get('parking'):
            parts.append(f"停車資訊：{attrs['parking']}")

        if attrs.get('traffic_info'):
            parts.append(f"交通資訊：{attrs['traffic_info'][:500]}")

        if attrs.get('phone_numbers'):
            parts.append(f"電話：{', '.join(attrs['phone_numbers'])}")

        content = "\n".join(parts)

        metadata = {
            "attraction_id": attr["id"],
            "name_zh": attr["names"]["zh-TW"],
            "category": attr["category"],
            "all_categories": attr["all_categories"],
            "city": loc["city"],
            "district": loc["district"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "language": "zh-TW",
            "source": attr["provenance"]["source"],
            "source_url": attr["provenance"]["source_url"],
            "confidence": attr["quality"]["confidence"],
            "has_fees": attr["quality"]["has_fees"],
            "has_service_times": attr["quality"]["has_service_times"],
        }

        doc = Document(
            id=attr["id"],
            content=content,
            metadata=metadata,
            source=attr["provenance"]["source_url"],
        )
        documents.append(doc)

    return documents


def main():
    print("Loading attractions_v2.json...")
    v2_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "attractions_v2.json"

    if not v2_path.exists():
        print(f"❌ File not found: {v2_path}")
        print("Run scripts/merge_attraction_data.py first")
        sys.exit(1)

    with open(v2_path, encoding="utf-8") as f:
        data = json.load(f)

    attractions = data["attractions"]
    print(f"Loaded {len(attractions)} attractions")

    print("\nConverting to documents...")
    documents = attractions_to_documents(attractions)
    print(f"Created {len(documents)} documents")

    print("\nInitializing local multilingual embedder...")
    embedder = LocalEmbedder()

    print("\nBuilding ingestion pipeline...")
    # Clear existing vectorstore
    import shutil
    vs_path = Path("./data/vectorstore")
    if vs_path.exists():
        shutil.rmtree(vs_path)
    vs_path.mkdir(parents=True, exist_ok=True)

    # Build pipeline with local embedder
    chunk_config = ChunkConfig(
        strategy="semantic",
        chunk_size=1000,
        chunk_overlap=100,
        min_chunk_size=50,
    )
    # Note: We'll use a custom embedder, so we need to monkey-patch or create custom pipeline
    # For simplicity, we'll use the pipeline but replace the embedder

    # Create vector store
    vector_config = VectorStoreConfig(
        provider="chroma",
        persist_dir=str(vs_path),
        collection_name="tiehua_rag",
    )

    # Use the builder but inject our local embedder
    builder = IngestionPipelineBuilder()
    builder._chunk_config = chunk_config
    builder._vector_config = vector_config
    # We'll create a custom embedder wrapper that matches the expected interface

    class EmbedderWrapper:
        def __init__(self, local_embedder):
            self.local_embedder = local_embedder

        def embed(self, texts: List[str]) -> List[List[float]]:
            return self.local_embedder.embed(texts)

    # Build with custom embedder
    from rag.chunking import create_chunker
    from rag.vectorstore import VectorStoreFactory

    chunker = create_chunker(chunk_config)
    vector_store = VectorStoreFactory.create(vector_config)
    embedder_wrapper = EmbedderWrapper(embedder)

    pipeline = IngestionPipeline(
        chunker=chunker,
        embedder=embedder_wrapper,
        vector_store=vector_store,
        chunk_config=chunk_config,
    )

    print("\nIngesting documents in batches (this will take 30-60 minutes)...")
    results = pipeline.ingest_batch(documents)

    success_count = sum(1 for r in results if r.success)
    total_chunks = sum(r.chunks_created for r in results if r.success)

    print(f"\n✅ Ingestion complete:")
    print(f"  Successful: {success_count}/{len(documents)}")
    print(f"  Total chunks created: {total_chunks}")

    if success_count < len(documents):
        print("\n⚠️  Failed documents:")
        for r in results:
            if not r.success:
                print(f"  - {r.document_id}: {r.error}")

    # Verify
    client = vector_store.client
    collection = client.get_collection("tiehua_rag")
    print(f"  Vectorstore count: {collection.count()}")


if __name__ == "__main__":
    main()
```

---

## 快速檢查清單

- [ ] `ARCHITECTURE_EXTENSIONS.md` - 理解三大擴充架構
- [ ] `src/harness/graph.py` - 理解 LangGraph 節點流向
- [ ] `src/harness/prompt_manager.py` - 理解模組化 Prompt 組裝
- [ ] `src/tools/attractions.py` - 理解多語景點搜尋工具
- [ ] `src/rag/ingestion.py` - 理解 RAG ingestion 流程
- [ ] `data/processed/attractions_v2.json` - 理解 schema v2 結構
- [ ] 執行 `pytest tests/test_attractions.py -v` 驗證工具正常
- [ ] 執行 `python scripts/rebuild_vectorstore_local.py` 重建向量庫
- [ ] 啟動三個服務 (FastAPI:8000, Dashboard:8501, Chat:8502) 驗證端到端