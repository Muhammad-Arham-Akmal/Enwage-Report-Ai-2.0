import operator
import time
from typing import Any, Dict, List, Optional

from content_core import extract_content
from content_core.common import ProcessSourceState
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger
from typing_extensions import Annotated, NotRequired, TypedDict

from open_notebook.ai.models import Model, ModelManager
from open_notebook.domain.content_settings import ContentSettings
from open_notebook.domain.notebook import Asset, Source
from open_notebook.domain.transformation import Transformation
from open_notebook.graphs.transformation import graph as transform_graph
from open_notebook.utils.tracing import summarize_content_state


class SourceState(TypedDict):
    content_state: ProcessSourceState
    apply_transformations: List[Transformation]
    source_id: str
    notebook_ids: List[str]
    source: Source
    transformation: Annotated[list, operator.add]
    embed: bool
    trace_id: NotRequired[Optional[str]]


class TransformationState(TypedDict):
    source: Source
    transformation: Transformation
    trace_id: NotRequired[Optional[str]]


def _trace_id(state: dict) -> str:
    return state.get("trace_id") or "none"


async def content_process(state: SourceState) -> dict:
    trace_id = _trace_id(state)
    source_id = state.get("source_id")
    started_at = time.time()
    content_settings = ContentSettings(
        default_content_processing_engine_doc="auto",
        default_content_processing_engine_url="auto",
        default_embedding_option="ask",
        auto_delete_files="yes",
        youtube_preferred_languages=[
            "en",
            "pt",
            "es",
            "de",
            "nl",
            "en-GB",
            "fr",
            "hi",
            "ja",
        ],
    )
    content_state: Dict[str, Any] = state["content_state"]  # type: ignore[assignment]

    logger.info(
        "Content extraction starting: "
        f"trace_id={trace_id} source_id={source_id} "
        f"content_state={summarize_content_state(content_state)}"
    )

    content_state["url_engine"] = (
        content_settings.default_content_processing_engine_url or "auto"
    )
    content_state["document_engine"] = (
        content_settings.default_content_processing_engine_doc or "auto"
    )
    content_state["output_format"] = "markdown"

    # Add speech-to-text model configuration from Default Models
    try:
        model_manager = ModelManager()
        defaults = await model_manager.get_defaults()
        if defaults.default_speech_to_text_model:
            stt_model = await Model.get(defaults.default_speech_to_text_model)
            if stt_model:
                content_state["audio_provider"] = stt_model.provider
                content_state["audio_model"] = stt_model.name
                logger.debug(
                    "Using speech-to-text model: "
                    f"trace_id={trace_id} source_id={source_id} "
                    f"model={stt_model.provider}/{stt_model.name}"
                )
    except Exception as e:
        logger.opt(exception=e).warning(
            "Failed to retrieve speech-to-text model configuration: "
            f"trace_id={trace_id} source_id={source_id}"
        )
        # Continue without custom audio model (content-core will use its default)

    logger.info(
        "Content extraction configured: "
        f"trace_id={trace_id} source_id={source_id} "
        f"content_state={summarize_content_state(content_state)}"
    )

    try:
        processed_state = await extract_content(content_state)
    except Exception as e:
        logger.opt(exception=e).error(
            "Content extraction failed: "
            f"trace_id={trace_id} source_id={source_id} "
            f"duration={time.time() - started_at:.2f}s "
            f"content_state={summarize_content_state(content_state)}"
        )
        raise

    if not processed_state.content or not processed_state.content.strip():
        logger.error(
            "Content extraction returned empty content: "
            f"trace_id={trace_id} source_id={source_id} "
            f"duration={time.time() - started_at:.2f}s "
            f"processed_state={summarize_content_state(processed_state)}"
        )
        url = processed_state.url or ""
        if url and ("youtube.com" in url or "youtu.be" in url):
            raise ValueError(
                "Could not extract content from this YouTube video. "
                "No transcript or subtitles are available. "
                "Try configuring a Speech-to-Text model in Settings "
                "to transcribe the audio instead."
            )
        raise ValueError(
            "Could not extract any text content from this source. "
            "The content may be empty, inaccessible, or in an unsupported format."
        )

    logger.info(
        "Content extraction completed: "
        f"trace_id={trace_id} source_id={source_id} "
        f"duration={time.time() - started_at:.2f}s "
        f"processed_state={summarize_content_state(processed_state)} "
        f"title_present={bool(processed_state.title)}"
    )

    return {"content_state": processed_state}


async def save_source(state: SourceState) -> dict:
    trace_id = _trace_id(state)
    started_at = time.time()
    content_state = state["content_state"]

    logger.info(
        "Saving processed source content: "
        f"trace_id={trace_id} source_id={state['source_id']} "
        f"content_state={summarize_content_state(content_state)}"
    )

    try:
        # Get existing source using the provided source_id
        source = await Source.get(state["source_id"])
        if not source:
            raise ValueError(f"Source with ID {state['source_id']} not found")

        # Update the source with processed content
        source.asset = Asset(url=content_state.url, file_path=content_state.file_path)
        source.full_text = content_state.content

        # Preserve user-set title; only overwrite placeholder or empty titles
        if content_state.title and (not source.title or source.title == "Processing..."):
            source.title = content_state.title

        await source.save()
        logger.info(
            "Processed source content saved: "
            f"trace_id={trace_id} source_id={source.id} "
            f"title={source.title!r} content_chars={len(source.full_text or '')} "
            f"duration={time.time() - started_at:.2f}s"
        )

        # NOTE: Notebook associations are created by the API immediately for UI responsiveness
        # No need to create them here to avoid duplicate edges

        if state["embed"]:
            if source.full_text and source.full_text.strip():
                logger.info(
                    "Submitting source embedding job: "
                    f"trace_id={trace_id} source_id={source.id} "
                    f"content_chars={len(source.full_text or '')}"
                )
                embed_command_id = await source.vectorize(trace_id=trace_id)
                logger.info(
                    "Source embedding job submitted: "
                    f"trace_id={trace_id} source_id={source.id} "
                    f"command_id={embed_command_id}"
                )
            else:
                logger.warning(
                    "Source has no text content to embed, skipping vectorization: "
                    f"trace_id={trace_id} source_id={source.id}"
                )
        else:
            logger.info(
                "Source embedding skipped by request: "
                f"trace_id={trace_id} source_id={source.id}"
            )

        return {"source": source}
    except Exception as e:
        logger.opt(exception=e).error(
            "Failed to save processed source content: "
            f"trace_id={trace_id} source_id={state['source_id']} "
            f"duration={time.time() - started_at:.2f}s"
        )
        raise


def trigger_transformations(state: SourceState, config: RunnableConfig) -> List[Send]:
    if len(state["apply_transformations"]) == 0:
        return []

    to_apply = state["apply_transformations"]
    trace_id = _trace_id(state)
    logger.info(
        "Scheduling source transformations: "
        f"trace_id={trace_id} source_id={state['source'].id} "
        f"count={len(to_apply)} "
        f"transformations={[getattr(t, 'id', None) or t.name for t in to_apply]}"
    )

    return [
        Send(
            "transform_content",
            {
                "source": state["source"],
                "transformation": t,
                "trace_id": trace_id,
            },
        )
        for t in to_apply
    ]


async def transform_content(state: TransformationState) -> Optional[dict]:
    trace_id = _trace_id(state)
    started_at = time.time()
    source = state["source"]
    content = source.full_text
    if not content:
        logger.warning(
            "Skipping source transformation because content is empty: "
            f"trace_id={trace_id} source_id={source.id}"
        )
        return None
    transformation: Transformation = state["transformation"]

    logger.info(
        "Applying source transformation: "
        f"trace_id={trace_id} source_id={source.id} "
        f"transformation_id={transformation.id} name={transformation.name!r}"
    )
    try:
        result = await transform_graph.ainvoke(
            dict(input_text=content, transformation=transformation)  # type: ignore[arg-type]
        )
        insight_command_id = await source.add_insight(transformation.title, result["output"])
        logger.info(
            "Source transformation completed: "
            f"trace_id={trace_id} source_id={source.id} "
            f"transformation_id={transformation.id} "
            f"output_chars={len(result['output'])} "
            f"insight_command_id={insight_command_id} "
            f"duration={time.time() - started_at:.2f}s"
        )
        return {
            "transformation": [
                {
                    "output": result["output"],
                    "transformation_name": transformation.name,
                }
            ]
        }
    except Exception as e:
        logger.opt(exception=e).error(
            "Source transformation failed: "
            f"trace_id={trace_id} source_id={source.id} "
            f"transformation_id={transformation.id} "
            f"duration={time.time() - started_at:.2f}s"
        )
        raise


# Create and compile the workflow
workflow = StateGraph(SourceState)

# Add nodes
workflow.add_node("content_process", content_process)
workflow.add_node("save_source", save_source)
workflow.add_node("transform_content", transform_content)
# Define the graph edges
workflow.add_edge(START, "content_process")
workflow.add_edge("content_process", "save_source")
workflow.add_conditional_edges(
    "save_source", trigger_transformations, ["transform_content"]
)
workflow.add_edge("transform_content", END)

# Compile the graph
source_graph = workflow.compile()
