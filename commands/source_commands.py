import time
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from surreal_commands import CommandInput, CommandOutput, command

from open_notebook.database.repository import ensure_record_id
from open_notebook.domain.notebook import Source
from open_notebook.domain.transformation import Transformation
from open_notebook.exceptions import ConfigurationError
from open_notebook.utils.tracing import summarize_content_state

try:
    from open_notebook.graphs.source import source_graph
    from open_notebook.graphs.transformation import graph as transform_graph
except ImportError as e:
    logger.error(f"Failed to import graphs: {e}")
    raise ValueError("graphs not available")


def full_model_dump(model):
    if isinstance(model, BaseModel):
        return model.model_dump()
    elif isinstance(model, dict):
        return {k: full_model_dump(v) for k, v in model.items()}
    elif isinstance(model, list):
        return [full_model_dump(item) for item in model]
    else:
        return model


def get_command_id(input_data: CommandInput) -> str:
    """Extract command_id from input_data's execution context, or return 'unknown'."""
    if input_data.execution_context:
        return str(input_data.execution_context.command_id)
    return "unknown"


class SourceProcessingInput(CommandInput):
    source_id: str
    content_state: Dict[str, Any]
    notebook_ids: List[str]
    transformations: List[str]
    embed: bool
    trace_id: Optional[str] = None


class SourceProcessingOutput(CommandOutput):
    success: bool
    source_id: str
    embedded_chunks: int = 0
    insights_created: int = 0
    processing_time: float
    error_message: Optional[str] = None
    trace_id: Optional[str] = None


@command(
    "process_source",
    app="open_notebook",
    retry={
        "max_attempts": 15,  # Handle deep queues (workaround for SurrealDB v2 transaction conflicts)
        "wait_strategy": "exponential_jitter",
        "wait_min": 1,
        "wait_max": 120,  # Allow queue to drain
        "stop_on": [ValueError, ConfigurationError],  # Don't retry validation/config errors
        "retry_log_level": "debug",  # Avoid log noise during transaction conflicts
    },
)
async def process_source_command(
    input_data: SourceProcessingInput,
) -> SourceProcessingOutput:
    """
    Process source content using the source_graph workflow
    """
    start_time = time.time()
    trace_id = input_data.trace_id or "none"
    command_id = get_command_id(input_data)

    try:
        logger.info(
            "Starting source processing command: "
            f"trace_id={trace_id} command_id={command_id} "
            f"source_id={input_data.source_id} "
            f"notebook_count={len(input_data.notebook_ids or [])} "
            f"transformation_count={len(input_data.transformations or [])} "
            f"embed={input_data.embed} "
            f"content_state={summarize_content_state(input_data.content_state)}"
        )

        # 1. Load transformation objects from IDs
        transformations = []
        for trans_id in input_data.transformations:
            logger.info(
                "Loading transformation for source processing: "
                f"trace_id={trace_id} command_id={command_id} "
                f"source_id={input_data.source_id} transformation_id={trans_id}"
            )
            transformation = await Transformation.get(trans_id)
            if not transformation:
                raise ValueError(f"Transformation '{trans_id}' not found")
            transformations.append(transformation)

        logger.info(
            "Loaded source transformations: "
            f"trace_id={trace_id} command_id={command_id} "
            f"source_id={input_data.source_id} count={len(transformations)}"
        )

        # 2. Get existing source record to update its command field
        source = await Source.get(input_data.source_id)
        if not source:
            raise ValueError(f"Source '{input_data.source_id}' not found")

        # Update source with command reference
        source.command = (
            ensure_record_id(input_data.execution_context.command_id)
            if input_data.execution_context
            else None
        )
        await source.save()

        logger.info(
            "Updated source with command reference: "
            f"trace_id={trace_id} command_id={command_id} source_id={source.id}"
        )

        # 3. Process source with all notebooks
        logger.info(
            "Invoking source processing graph: "
            f"trace_id={trace_id} command_id={command_id} "
            f"source_id={input_data.source_id}"
        )

        # Execute source_graph with all notebooks
        graph_started_at = time.time()
        result = await source_graph.ainvoke(
            {  # type: ignore[arg-type]
                "content_state": input_data.content_state,
                "notebook_ids": input_data.notebook_ids,  # Use notebook_ids (plural) as expected by SourceState
                "apply_transformations": transformations,
                "embed": input_data.embed,
                "source_id": input_data.source_id,  # Add the source_id to the state
                "trace_id": trace_id,
            }
        )
        logger.info(
            "Source processing graph completed: "
            f"trace_id={trace_id} command_id={command_id} "
            f"source_id={input_data.source_id} "
            f"duration={time.time() - graph_started_at:.2f}s"
        )

        processed_source = result["source"]

        # 4. Gather processing results (notebook associations handled by source_graph)
        # Note: embedding is fire-and-forget (async job), so we can't query the
        # count here — it hasn't completed yet. The embed_source_command logs
        # the actual count when it finishes.
        insights_list = await processed_source.get_insights()
        insights_created = len(insights_list)

        processing_time = time.time() - start_time
        embed_status = "submitted" if input_data.embed else "skipped"
        logger.info(
            "Successfully processed source: "
            f"trace_id={trace_id} command_id={command_id} "
            f"source_id={processed_source.id} duration={processing_time:.2f}s "
            f"insights_created={insights_created} embedding={embed_status}"
        )

        return SourceProcessingOutput(
            success=True,
            source_id=str(processed_source.id),
            embedded_chunks=0,
            insights_created=insights_created,
            processing_time=processing_time,
            trace_id=trace_id,
        )

    except ValueError as e:
        # Validation errors are permanent failures - don't retry
        processing_time = time.time() - start_time
        logger.opt(exception=e).error(
            "Source processing failed permanently: "
            f"trace_id={trace_id} command_id={command_id} "
            f"source_id={input_data.source_id} duration={processing_time:.2f}s "
            f"content_state={summarize_content_state(input_data.content_state)}"
        )
        return SourceProcessingOutput(
            success=False,
            source_id=input_data.source_id,
            processing_time=processing_time,
            error_message=str(e),
            trace_id=trace_id,
        )
    except Exception as e:
        # Transient failure - will be retried (surreal-commands logs final failure)
        logger.opt(exception=e).warning(
            "Transient error processing source; command will retry: "
            f"trace_id={trace_id} command_id={command_id} "
            f"source_id={input_data.source_id} "
            f"duration={time.time() - start_time:.2f}s"
        )
        raise


# =============================================================================
# RUN TRANSFORMATION COMMAND
# =============================================================================


class RunTransformationInput(CommandInput):
    """Input for running a transformation on an existing source."""

    source_id: str
    transformation_id: str


class RunTransformationOutput(CommandOutput):
    """Output from transformation command."""

    success: bool
    source_id: str
    transformation_id: str
    processing_time: float
    error_message: Optional[str] = None


@command(
    "run_transformation",
    app="open_notebook",
    retry={
        "max_attempts": 5,
        "wait_strategy": "exponential_jitter",
        "wait_min": 1,
        "wait_max": 60,
        "stop_on": [ValueError, ConfigurationError],  # Don't retry validation/config errors
        "retry_log_level": "debug",
    },
)
async def run_transformation_command(
    input_data: RunTransformationInput,
) -> RunTransformationOutput:
    """
    Run a transformation on an existing source to generate an insight.

    This command runs the transformation graph which:
    1. Loads the source and transformation
    2. Calls the LLM to generate insight content
    3. Creates the insight via create_insight command (fire-and-forget)

    Use this command for UI-triggered insight generation to avoid blocking
    the HTTP request while the LLM processes.

    Retry Strategy:
    - Retries up to 5 times for transient failures (network, timeout, etc.)
    - Uses exponential-jitter backoff (1-60s)
    - Does NOT retry permanent failures (ValueError for validation errors)
    """
    start_time = time.time()

    try:
        logger.info(
            f"Running transformation {input_data.transformation_id} "
            f"on source {input_data.source_id}"
        )

        # Load source
        source = await Source.get(input_data.source_id)
        if not source:
            raise ValueError(f"Source '{input_data.source_id}' not found")

        # Load transformation
        transformation = await Transformation.get(input_data.transformation_id)
        if not transformation:
            raise ValueError(
                f"Transformation '{input_data.transformation_id}' not found"
            )

        # Run transformation graph (includes LLM call + insight creation)
        await transform_graph.ainvoke(
            input=dict(source=source, transformation=transformation)
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Successfully ran transformation {input_data.transformation_id} "
            f"on source {input_data.source_id} in {processing_time:.2f}s"
        )

        return RunTransformationOutput(
            success=True,
            source_id=input_data.source_id,
            transformation_id=input_data.transformation_id,
            processing_time=processing_time,
        )

    except ValueError as e:
        # Validation errors are permanent failures - don't retry
        processing_time = time.time() - start_time
        logger.error(
            f"Failed to run transformation {input_data.transformation_id} "
            f"on source {input_data.source_id}: {e}"
        )
        return RunTransformationOutput(
            success=False,
            source_id=input_data.source_id,
            transformation_id=input_data.transformation_id,
            processing_time=processing_time,
            error_message=str(e),
        )
    except Exception as e:
        # Transient failure - will be retried (surreal-commands logs final failure)
        logger.debug(
            f"Transient error running transformation {input_data.transformation_id} "
            f"on source {input_data.source_id}: {e}"
        )
        raise
