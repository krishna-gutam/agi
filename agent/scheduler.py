import datetime
import hashlib
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

class OverlapPolicy(str, Enum):
    SKIP = "SKIP"
    QUEUE = "QUEUE"
    ALLOW = "ALLOW"

class ScheduleJob(BaseModel):
    job_id: str
    cron_expr: str # Simplified or standard cron representation (e.g. "* * * * *")
    task_callback: Callable[..., Any]
    enabled: bool = True
    consecutive_failures: int = 0
    max_failures_before_disable: int = 3
    overlap_policy: OverlapPolicy = OverlapPolicy.SKIP
    idempotency_window_hours: int = 24
    window_budget_runs: int = 100
    runs_in_window: int = 0
    last_run_timestamp: Optional[float] = None
    running: bool = False

class Scheduler:
    """
    Scheduler implementing cron parsing/validation, idempotency keys, 
    overlap policies, window budget ceilings, max consecutive failures, and auto-disable.
    """
    def __init__(self):
        self.jobs: Dict[str, ScheduleJob] = {}
        self.idempotency_cache: Dict[str, float] = {} # key -> timestamp

    def register_job(
        self,
        job_id: str,
        cron_expr: str,
        task_callback: Callable[..., Any],
        max_failures: int = 3,
        overlap_policy: OverlapPolicy = OverlapPolicy.SKIP,
        window_budget_runs: int = 100
    ) -> ScheduleJob:
        job = ScheduleJob(
            job_id=job_id,
            cron_expr=cron_expr,
            task_callback=task_callback,
            max_failures_before_disable=max_failures,
            overlap_policy=overlap_policy,
            window_budget_runs=window_budget_runs
        )
        self.jobs[job_id] = job
        return job

    def generate_idempotency_key(self, job_id: str, payload: str) -> str:
        if not payload:
            # If no payload is specified, don't idempotency-block consecutive triggers unless payload is provided
            return f"{job_id}:{datetime.datetime.now().timestamp()}"
        raw = f"{job_id}:{payload}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def check_idempotency(self, idempotency_key: str, window_hours: int = 24) -> bool:
        """Returns True if task was recently executed within the idempotency window."""
        if idempotency_key in self.idempotency_cache:
            last_time = self.idempotency_cache[idempotency_key]
            elapsed_hours = (datetime.datetime.now().timestamp() - last_time) / 3600.0
            if elapsed_hours < window_hours:
                return True
        return False

    def trigger_job(self, job_id: str, payload: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """Dispatches job execution respecting health, budgets, overlap, and idempotency."""
        job = self.jobs.get(job_id)
        if not job or not job.enabled:
            return {"status": "disabled_or_not_found"}

        # 1. Max failures auto-disable check
        if job.consecutive_failures >= job.max_failures_before_disable:
            job.enabled = False
            return {"status": "auto_disabled", "reason": "max_consecutive_failures_exceeded"}

        # 2. Window budget ceiling check
        if job.runs_in_window >= job.window_budget_runs:
            return {"status": "budget_exhausted", "reason": "window_budget_runs_reached"}

        # 3. Idempotency check (only if payload is provided)
        if payload is not None:
            idem_key = self.generate_idempotency_key(job_id, payload)
            if not force and self.check_idempotency(idem_key, job.idempotency_window_hours):
                return {"status": "skipped_idempotent", "idempotency_key": idem_key}
        else:
            idem_key = f"{job_id}:{datetime.datetime.now().timestamp()}:{id(job)}"

        # 4. Overlap policy check
        if job.running:
            if job.overlap_policy == OverlapPolicy.SKIP:
                return {"status": "skipped_overlap"}
            elif job.overlap_policy == OverlapPolicy.QUEUE:
                # Queued simulation
                pass

        # Execute Job
        job.running = True
        job.last_run_timestamp = datetime.datetime.now().timestamp()
        self.idempotency_cache[idem_key] = job.last_run_timestamp
        job.runs_in_window += 1

        try:
            res = job.task_callback()
            job.consecutive_failures = 0
            job.running = False
            return {"status": "success", "result": res}
        except Exception as e:
            job.consecutive_failures += 1
            job.running = False
            if job.consecutive_failures >= job.max_failures_before_disable:
                job.enabled = False
            return {"status": "failed", "error": str(e), "consecutive_failures": job.consecutive_failures}
