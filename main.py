"""Taiwan Tourism Agent Harness - FastAPI Entry Point"""

import os
import json
import time
import queue
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, AsyncGenerator
import uvicorn

from src.harness.runner import HarnessRunner
from src.agent import create_agent
from src.observability import get_metrics, get_logger, setup_logging


# Global runner instance
_runner: Optional[HarnessRunner] = None
_metrics = get_metrics()
_logger = get_logger("api")

# Default settings for the global runner
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_USE_LANGGRAPH = True

# Maximum allowed retry count to prevent abuse
MAX_RETRY_LIMIT = 10
MIN_RETRY_LIMIT = 0


class QueryRequest(BaseModel):
    """Request model for agent query."""
    query: str = Field(
        min_length=1,
        max_length=5000,
        description="User query in natural language",
    )
    max_retries: int = Field(
        default=_DEFAULT_MAX_RETRIES,
        ge=MIN_RETRY_LIMIT,
        le=MAX_RETRY_LIMIT,
        description="Maximum number of retry attempts",
    )
    use_langgraph: bool = Field(
        default=_DEFAULT_USE_LANGGRAPH,
        description="Use LangGraph-based execution",
    )


class QueryResponse(BaseModel):
    """Response model for agent query."""
    query: str
    final_answer: str
    status: str
    success: bool
    retry_count: int
    tool_calls: list
    tool_results: list
    evaluation: Optional[Dict[str, Any]] = None
    feedback: list
    run_id: str
    duration_ms: int


def _create_runner(max_retries: int, use_langgraph: bool) -> HarnessRunner:
    """Create a HarnessRunner with validated settings."""
    return HarnessRunner(
        max_retries=max_retries,
        use_langgraph=use_langgraph,
    )


def _get_runner(request: QueryRequest) -> HarnessRunner:
    """Return the global runner or create a custom one based on request settings.

    Since QueryRequest now uses non-Optional types with defaults,
    request.max_retries and request.use_langgraph are always concrete values.
    """
    if request.max_retries == _DEFAULT_MAX_RETRIES and request.use_langgraph == _DEFAULT_USE_LANGGRAPH:
        if _runner is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        return _runner
    return _create_runner(request.max_retries, request.use_langgraph)


def _sanitize_error(exc: Exception) -> str:
    """Return a safe, client-facing error message.

    Internal implementation details (tracebacks, file paths, stack frames)
    are logged server-side but never exposed to the client.
    """
    error_type = type(exc).__name__
    # Return a generic message that does not leak internal details
    return f"Internal server error: {error_type}"


# Pre-built Mermaid graph (lazily initialized on first request)
_mermaid_graph: Optional[str] = None


def _get_mermaid_graph() -> str:
    """Build or return the cached Mermaid diagram of the harness graph."""
    global _mermaid_graph
    if _mermaid_graph is not None:
        return _mermaid_graph

    from src.harness.graph import create_harness_graph
    from src.agent import create_agent

    agent = create_agent()
    graph = create_harness_graph(
        agent_fn=lambda s: agent.run_with_state(s)
    )
    _mermaid_graph = graph.draw_mermaid()
    return _mermaid_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _runner, _mermaid_graph
    # Startup
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    _runner = _create_runner(_DEFAULT_MAX_RETRIES, _DEFAULT_USE_LANGGRAPH)
    _logger.info("api_started", version="1.0.0")
    yield
    # Shutdown
    _logger.info("api_shutdown")


app = FastAPI(
    title="Taiwan Tourism Agent Harness",
    description="A Tool-Using Tourism Agent with Evaluation, Structured Feedback and Stateful Retry",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Liveness probe - process is alive."""
    return {"status": "healthy", "service": "taiwan-tourism-api"}


@app.get("/ready")
async def readiness_check():
    """Readiness probe - service can handle requests."""
    # Check if runner is initialized
    if _runner is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Check configuration
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not configured")
    
    return {
        "status": "ready",
        "service": "taiwan-tourism-api",
        "checks": {
            "runner": "ok",
            "api_key": "configured",
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query through the agent harness."""
    start_time = time.time()

    try:
        runner = _get_runner(request)

        # Execute harness in a thread pool to avoid blocking the event loop
        result = await run_in_threadpool(runner.run, request.query)

        duration_ms = int((time.time() - start_time) * 1000)

        # Record metrics
        _metrics.record_agent_run(result.status, duration_ms / 1000)

        return QueryResponse(
            query=result.query,
            final_answer=result.final_answer,
            status=result.status,
            success=result.success,
            retry_count=result.retry_count,
            tool_calls=result.tool_calls,
            tool_results=result.tool_results,
            evaluation=result.evaluation,
            feedback=result.feedback,
            run_id=result.state.run_id if result.state else "",
            duration_ms=duration_ms,
        )

    except HTTPException:
        # Re-raise HTTPException as-is to preserve original status code
        # (e.g. 503 from _get_runner when service is not ready)
        raise

    except Exception as e:
        _logger.error("query_failed", query=request.query[:100], error=str(e))
        duration_ms = int((time.time() - start_time) * 1000)
        _metrics.record_error(type(e).__name__, "api")
        _metrics.record_agent_run("failed", duration_ms / 1000)
        raise HTTPException(
            status_code=500,
            detail=_sanitize_error(e),
        ) from e


@app.post("/query/stream")
async def stream_query(request: QueryRequest) -> StreamingResponse:
    """Stream query execution steps via SSE (Server-Sent Events).

    Uses a background thread + queue pattern so that each LangGraph step
    is forwarded to the client as soon as it is produced, rather than
    buffered until the entire agent run completes.

    Requires use_langgraph=True.
    """
    if not request.use_langgraph:
        raise HTTPException(status_code=400, detail="Streaming only available with LangGraph")

    async def generate() -> AsyncGenerator[str, None]:
        events: "queue.Queue" = queue.Queue()
        sentinel = object()
        error_holder: list = []

        def _worker():
            """Run the synchronous LangGraph stream in a background thread."""
            try:
                runner = _create_runner(request.max_retries, use_langgraph=True)
                for step in runner.run_streaming(request.query, max_retries=request.max_retries):
                    events.put(("step", step))
            except Exception as exc:
                error_holder.append(exc)
            finally:
                events.put(("done", sentinel))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while True:
                kind, value = await run_in_threadpool(events.get)

                if kind == "done":
                    break

                if kind == "step":
                    agent_state = value.get("agent_state", {})
                    event = {
                        "node": value.get("current_node", "unknown"),
                        "status": agent_state.get("status", "running"),
                        "attempt": agent_state.get("attempt", 0),
                        "execution_path": value.get("execution_path", []),
                    }
                    if agent_state.get("tool_calls"):
                        event["tool_calls"] = agent_state["tool_calls"]
                    if agent_state.get("answer"):
                        event["partial_answer"] = agent_state["answer"]
                    yield f"data: {json.dumps(event)}\n\n"

            yield f"data: {json.dumps({'status': 'complete'})}\n\n"

        except Exception as e:
            _logger.error("stream_query_failed", query=request.query[:100], error=str(e))
            yield f"data: {json.dumps({'status': 'error', 'message': _sanitize_error(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/graph/visualize")
async def visualize_graph():
    """Get Mermaid diagram of the LangGraph harness.

    The graph is built once and cached for subsequent requests.
    """
    try:
        mermaid = _get_mermaid_graph()
        return {"mermaid": mermaid}
    except Exception as e:
        _logger.error("graph_visualize_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=_sanitize_error(e),
        ) from e


def main():
    """Run the API server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()