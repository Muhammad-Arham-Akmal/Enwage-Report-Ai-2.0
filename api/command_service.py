from typing import Any, Dict, List, Optional

from loguru import logger
from surreal_commands import get_command_status, submit_command

from open_notebook.utils.tracing import summarize_content_state


def _summarize_command_args(command_args: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"keys": sorted(command_args.keys())}
    for key in ("source_id", "trace_id", "embed"):
        if key in command_args:
            summary[key] = command_args[key]

    if "notebook_ids" in command_args:
        summary["notebook_count"] = len(command_args.get("notebook_ids") or [])
    if "transformations" in command_args:
        summary["transformation_count"] = len(command_args.get("transformations") or [])
    if "content_state" in command_args:
        summary["content_state"] = summarize_content_state(command_args["content_state"])

    return summary


class CommandService:
    """Generic service layer for command operations"""

    @staticmethod
    async def submit_command_job(
        module_name: str,  # Actually app_name for surreal-commands
        command_name: str,
        command_args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a generic command job for background processing"""
        trace_id = (context or {}).get("trace_id") or command_args.get("trace_id")
        try:
            logger.info(
                "Submitting command job: "
                f"trace_id={trace_id} app={module_name} command={command_name} "
                f"args={_summarize_command_args(command_args)} context={context or {}}"
            )

            # Ensure command modules are imported before submitting
            # This is needed because submit_command validates against local registry
            try:
                import commands.podcast_commands  # noqa: F401
            except ImportError as import_err:
                logger.opt(exception=import_err).error(
                    "Failed to import command modules before command submit: "
                    f"trace_id={trace_id} app={module_name} command={command_name}"
                )
                raise ValueError("Command modules not available")

            # surreal-commands expects: submit_command(app_name, command_name, args)
            cmd_id = submit_command(
                module_name,  # This is actually the app name (e.g., "open_notebook")
                command_name,  # Command name (e.g., "process_text")
                command_args,  # Input data
            )
            # Convert RecordID to string if needed
            if not cmd_id:
                raise ValueError("Failed to get cmd_id from submit_command")
            cmd_id_str = str(cmd_id)
            logger.info(
                "Submitted command job: "
                f"trace_id={trace_id} command_id={cmd_id_str} "
                f"app={module_name} command={command_name}"
            )
            return cmd_id_str

        except Exception as e:
            logger.opt(exception=e).error(
                "Failed to submit command job: "
                f"trace_id={trace_id} app={module_name} command={command_name} "
                f"args={_summarize_command_args(command_args)}"
            )
            raise

    @staticmethod
    async def get_command_status(job_id: str) -> Dict[str, Any]:
        """Get status of any command job"""
        try:
            logger.debug(f"Fetching command status: job_id={job_id}")
            status = await get_command_status(job_id)
            return {
                "job_id": job_id,
                "status": status.status if status else "unknown",
                "result": status.result if status else None,
                "error_message": getattr(status, "error_message", None)
                if status
                else None,
                "created": str(status.created)
                if status and hasattr(status, "created") and status.created
                else None,
                "updated": str(status.updated)
                if status and hasattr(status, "updated") and status.updated
                else None,
                "progress": getattr(status, "progress", None) if status else None,
            }
        except Exception as e:
            logger.opt(exception=e).error(
                f"Failed to get command status: job_id={job_id}"
            )
            raise

    @staticmethod
    async def list_command_jobs(
        module_filter: Optional[str] = None,
        command_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List command jobs with optional filtering"""
        # This will be implemented with proper SurrealDB queries
        # For now, return empty list as this is foundation phase
        return []

    @staticmethod
    async def cancel_command_job(job_id: str) -> bool:
        """Cancel a running command job"""
        try:
            # Implementation depends on surreal-commands cancellation support
            # For now, just log the attempt
            logger.info(f"Attempting to cancel job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel command job: {e}")
            raise
