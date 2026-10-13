"""Taiwan Tourism Agent Harness - FastAPI Entry Point"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

from src.harness.runner import HarnessRunner
from src.agent import create_agent
from src.observability import get_metrics, get_logger, setup_logging


# Global runner instance
_runner: Optional[HarnessRunner] = None
_metrics = get_metrics()
_logger = get_logger("api")


class QueryRequest(BaseModel):
    """Request model for agent query."""
    query: str
    max_retries: Optional[int] = 2
    use_langgraph: Optional[bool] = True


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _runner
    # Startup
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    _runner = HarnessRunner(
        max_retries=2,
        use_langgraph=True,
    )
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
    import time
    start_time = time.time()
    
    try:
        # Create a new runner with custom settings if needed
        if request.max_retries != 2 or request.use_langgraph != True:
            runner = HarnessRunner(
                max_retries=request.max_retries,
                use_langgraph=request.use_langgraph,
            )
        else:
            runner = _runner
        
        # Execute harness
        result = runner.run(request.query)
        
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
    
    except Exception as e:
        _logger.error("query_failed", query=request.query[:100], error=str(e))
        duration_ms = int((time.time() - start_time) * 1000)
        _metrics.record_error(type(e).__name__, "api")
        _metrics.record_agent_run("failed", duration_ms / 1000)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """Stream query execution steps (LangGraph only)."""
    if not request.use_langgraph:
        raise HTTPException(status_code=400, detail="Streaming only available with LangGraph")
    
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate():
        runner = HarnessRunner(
            max_retries=request.max_retries,
            use_langgraph=True,
        )
        agent = create_agent()
        
        from src.state import initialize_state
        state = initialize_state(request.query, max_retries=request.max_retries)
        
        graph_runner = runner._run_with_langgraph(agent, state)
        
        # This would need async streaming support in the graph runner
        # For now, return the final result
        result = runner.run(request.query)
        yield f"data: {json.dumps({'status': 'complete', 'result': result.final_answer})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/graph/visualize")
async def visualize_graph():
    """Get Mermaid diagram of the LangGraph harness."""
    from src.harness.graph import create_harness_graph
    from src.agent import create_agent
    
    agent = create_agent()
    graph = create_harness_graph(agent_fn=lambda s: agent.run_with_state(s))
    
    return {"mermaid": graph.draw_mermaid()}


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