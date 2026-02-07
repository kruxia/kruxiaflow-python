# Kruxia Flow Python SDK

[![CI](https://github.com/kruxia/kruxiaflow/actions/workflows/main-ci.yml/badge.svg)](https://github.com/kruxia/kruxiaflow-python/actions/workflows/main-ci.yml)
[![Docker Image](https://img.shields.io/docker/image-size/kruxia/kruxiaflow-py-std/latest?label=docker%20image)](https://hub.docker.com/r/kruxia/kruxiaflow-py-std)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/1457098705214640333?logo=discord&label=Discord)](https://discord.gg/ZJAzygCq)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Python SDK for [Kruxia Flow](https://github.com/kruxia/kruxiaflow) — the AI-native durable workflow engine.

Define workflows in Python, deploy them to the Kruxia Flow server, and build custom Python workers to execute activities.

## Installation

The Python SDK is not yet published to PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/kruxia/kruxiaflow-python.git
```

For the standard Python worker with data science packages:

```bash
pip install "kruxiaflow-python[std] @ git+https://github.com/kruxia/kruxiaflow-python.git"
```

## Quick Start

### Define a Workflow

Workflows describe WHAT to run. They compile to YAML and deploy to the Kruxia Flow API server.

```python
from kruxiaflow import Activity, Workflow, workflow

# Step 1: Fetch weather data
fetch_weather = Activity(
    key="fetch_weather",
    worker="std",
    activity_name="http_request",
    parameters={
        "method": "GET",
        "url": "https://api.weather.gov/gridpoints/LOT/76,73/forecast",
    },
    outputs=["response"],
)

# Step 2: Send notification (depends on step 1)
send_notification = Activity(
    key="send_notification",
    worker="std",
    activity_name="http_request",
    parameters={
        "method": "POST",
        "url": "http://mailpit:8025/api/v1/send",
        "body": {
            "Subject": f"Weather Report - {workflow.id}",
            "Text": f"Forecast: {fetch_weather['response.body.properties.periods[0].detailedForecast']}",
        },
    },
    depends_on=["fetch_weather"],
)

# Build and print as YAML
weather_workflow = Workflow(
    name="weather_report",
    activities=[fetch_weather, send_notification],
)

print(weather_workflow)
```

### Deploy to Kruxia Flow

```python
from kruxiaflow import KruxiaFlow

client = KruxiaFlow(api_url="http://localhost:8080")
client.deploy(weather_workflow)

# Start a workflow instance
instance = client.start_workflow(
    "weather_report",
    inputs={"webhook_url": "https://httpbin.org/post"},
)
print(f"Started: {instance['workflow_id']}")
```

### Write Python Activities with ScriptActivity

Use `ScriptActivity` to write activities as Python functions that run on the py-std worker:

```python
from kruxiaflow import ScriptActivity, Workflow

@ScriptActivity.from_function(
    inputs={"sales_data": [{"product": "Laptop", "amount": 1200.00, "region": "North"}]},
)
async def load_data(sales_data):
    import pandas as pd
    df = pd.DataFrame(sales_data)
    return {"data": df.to_dict(), "row_count": len(df)}

@ScriptActivity.from_function(
    inputs={"data": load_data["data"]},
    depends_on=["load_data"],
)
async def sql_transform(data):
    import duckdb, pandas as pd
    df = pd.DataFrame(data)
    results = duckdb.sql("""
        SELECT region, SUM(amount) as total_sales
        FROM df GROUP BY region
    """).df()
    return {"results": results.to_dict()}

pipeline = Workflow(
    name="sales_pipeline",
    activities=[load_data, sql_transform],
)
print(pipeline)
```

## Architecture

The Python SDK has two distinct contexts:

```
┌──────────────────────────────────────────────────────────────────┐
│  WORKFLOW DEFINITION (my_workflow.py)                             │
│  Describes WHAT to run. Compiles to YAML and deploys to API.     │
│                                                                  │
│  from kruxiaflow import Activity, Workflow                       │
│  activity = Activity(key="process").with_worker("std", "analyze")│
└──────────────────────────────────────────────────────────────────┘
                              │ deploys to
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     KRUXIA FLOW API SERVER                        │
└──────────────────────────────────────────────────────────────────┘
                              ▲ polls for work
                              │
┌──────────────────────────────────────────────────────────────────┐
│  WORKER PROCESS (custom_worker.py)                               │
│  Contains the actual CODE that executes activities.              │
│                                                                  │
│  from kruxiaflow.worker import activity, ActivityResult          │
│  @activity()                                                     │
│  async def analyze(params, ctx): ...                             │
└──────────────────────────────────────────────────────────────────┘
```

- **`kruxiaflow.Activity`** — Workflow definition (describes what to run)
- **`kruxiaflow.ScriptActivity`** — Python function that runs on the py-std worker
- **`kruxiaflow.worker.Activity`** — Custom worker implementation (the code that runs)

## Features

### Workflow Definition API

Build workflows programmatically with full type safety:

```python
from kruxiaflow import Activity, Input, SecretRef, Dependency

# Workflow inputs and secrets
user_content = Input("user_content", type=str, required=True)
db_url = SecretRef("db_url")

# LLM activity with cost control
analyze = Activity(
    key="analyze",
    worker="std",
    activity_name="llm_prompt",
    parameters={
        "model": "anthropic/claude-haiku-4-5-20251001",
        "prompt": f"Analyze: {user_content!s}",
        "max_tokens": 500,
    },
    outputs=["result"],
    settings={
        "budget": {"limit": 0.10, "action": "abort"},
        "retry": {"max_attempts": 3, "strategy": "exponential"},
    },
)
```

### Expression System

SQLAlchemy-style operators for conditional dependencies:

```python
from kruxiaflow import Dependency

# Conditional dependency: only run if confidence > 0.8
save = Activity(
    key="save",
    ...
).with_dependencies(
    Dependency.on(analyze, analyze["confidence"] > 0.8)
)
```

### Output References

Access nested JSON paths from upstream activity outputs:

```python
# Dot-notation paths into activity outputs
analyze["result.content"]
analyze["result.usage.total_tokens"]
fetch["response.body.properties.periods[0].temperature"]
```

### YAML Export

Every workflow compiles to YAML that deploys directly to the Kruxia Flow API:

```python
# Print compiled YAML
print(workflow)

# Or get the YAML string
yaml_str = workflow.to_yaml()
```

### Async Client

```python
from kruxiaflow import AsyncKruxiaFlow

async def main():
    client = AsyncKruxiaFlow(api_url="http://localhost:8080")
    await client.deploy(workflow)
    instance = await client.start_workflow("my_workflow", inputs={...})
```

## Standard Python Worker (py-std)

The `py-std` worker provides a ready-to-use Docker image with pre-installed packages for common data processing, ML, and NLP tasks. Use it with `ScriptActivity` or inline scripts.

### Pre-installed Packages

| Category            | Packages                                       |
|---------------------|-------------------------------------------------|
| Core                | httpx, pydantic, orjson, pyyaml                |
| Data Processing     | pandas, polars, pyarrow, duckdb, numpy         |
| Machine Learning    | scikit-learn                                    |
| NLP                 | transformers, sentence-transformers, tiktoken  |

### Using with ScriptActivity

```python
from kruxiaflow import ScriptActivity

@ScriptActivity.from_function(worker="py-std")
async def process_data(records, threshold=0.5):
    import pandas as pd
    df = pd.DataFrame(records)
    filtered = df[df['value'] > threshold]
    return {"count": len(filtered), "total": filtered['value'].sum()}
```

### Using with Inline Scripts

```yaml
activities:
  - key: transform
    worker: py-std
    activity_name: script
    parameters:
      script: |
        import pandas as pd
        df = pd.DataFrame(INPUT["data"])
        OUTPUT = {"result": df.describe().to_dict()}
```

## Custom Workers

Build custom Python workers for activities that need specialized logic:

```python
from kruxiaflow.worker import (
    activity, ActivityContext, ActivityResult,
    ActivityRegistry, WorkerConfig, WorkerManager,
)

@activity()
async def analyze_sentiment(params: dict, ctx: ActivityContext) -> ActivityResult:
    text = params["text"]
    # Your custom logic here
    return ActivityResult.value("result", {"sentiment": "positive", "confidence": 0.85})

# Run the worker
async def main():
    config = WorkerConfig()  # reads from KRUXIAFLOW_* env vars
    registry = ActivityRegistry()
    registry.register(analyze_sentiment)
    manager = WorkerManager(config=config, registry=registry)
    await manager.run()
```

## Examples

15+ production-ready examples covering YAML workflows, Python SDK definitions, and ScriptActivity patterns:

| #  | Example                                               | Concepts                                          |
|----|-------------------------------------------------------|---------------------------------------------------|
| 1  | [Weather Report](examples/01_weather_report.py)       | Sequential workflow, HTTP requests, templates     |
| 2  | [User Validation](examples/02_user_validation.py)     | Conditional branching, PostgreSQL queries         |
| 3  | [Document Processing](examples/03_document_processing.py) | Parallel execution, fan-out/fan-in            |
| 4  | [Content Moderation](examples/04_moderate_content.py) | LLM cost tracking, budget control, retry          |
| 5  | [Research Assistant](examples/05_research_assistant.py) | Multi-model fallback, budget-aware selection    |
| 11 | [GitHub Health Check](examples/11_github_health_check.py) | ScriptActivity, HTTP API integration          |
| 12 | [Sales ETL Pipeline](examples/12_sales_etl_pipeline.py) | pandas, DuckDB SQL on DataFrames, Parquet      |
| 13 | [Customer Churn Prediction](examples/13_customer_churn_prediction.py) | Parallel ML training, LLM explanations |
| 14 | [Document Intelligence](examples/14_document_intelligence.py) | AI-powered document analysis              |
| 15 | [Content Moderation System](examples/15_content_moderation_system.py) | Multi-stage moderation pipeline       |
|    | [Custom Worker](examples/custom_worker.py)            | Worker SDK, ActivityRegistry, ActivityContext      |

Examples 11-15 include both `.py` (Python SDK) and `.yaml` (compiled output) files.

Run any example to see its compiled YAML:

```bash
python examples/01_weather_report.py
```

## Development

```bash
# Clone the repo
git clone https://github.com/kruxia/kruxiaflow-python.git
cd kruxiaflow-python

# Install dev dependencies
uv sync --dev

# Run tests (95% coverage required)
uv run pytest

# Lint and format
uv run ruff check
uv run ruff format

# Type check
uv run ty check
```

## Related

- **[Kruxia Flow](https://github.com/kruxia/kruxiaflow)** — The workflow engine server (Rust)
- **[kruxiaflow.com](https://kruxiaflow.com)** — Project homepage
- **[Discord](https://discord.gg/ZJAzygCq)** — Community chat

## License

[MIT](LICENSE)
