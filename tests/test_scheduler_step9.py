import pytest
from agent.scheduler import Scheduler, OverlapPolicy

def test_scheduler_basic_execution():
    scheduler = Scheduler()
    called = []
    
    def dummy_task():
        called.append(1)
        return "ok"

    job = scheduler.register_job("job_1", "* * * * *", dummy_task)
    res = scheduler.trigger_job("job_1")
    assert res["status"] == "success"
    assert res["result"] == "ok"
    assert len(called) == 1

def test_scheduler_idempotency():
    scheduler = Scheduler()
    call_count = 0

    def task():
        nonlocal call_count
        call_count += 1
        return call_count

    scheduler.register_job("job_idem", "* * * * *", task)
    
    # First trigger -> executes
    res1 = scheduler.trigger_job("job_idem", payload="same_payload")
    assert res1["status"] == "success"
    assert call_count == 1

    # Second trigger with same payload within idempotency window -> skipped idempotent
    res2 = scheduler.trigger_job("job_idem", payload="same_payload")
    assert res2["status"] == "skipped_idempotent"
    assert call_count == 1

    # Forced trigger -> executes again
    res3 = scheduler.trigger_job("job_idem", payload="same_payload", force=True)
    assert res3["status"] == "success"
    assert call_count == 2

def test_scheduler_auto_disable_on_failures():
    scheduler = Scheduler()
    
    def failing_task():
        raise RuntimeError("Task failed!")

    job = scheduler.register_job("job_fail", "* * * * *", failing_task, max_failures=2)
    
    # Failure 1
    res1 = scheduler.trigger_job("job_fail")
    assert res1["status"] == "failed"
    assert job.consecutive_failures == 1
    assert job.enabled

    # Failure 2 -> reaches max failures (2), should auto-disable
    res2 = scheduler.trigger_job("job_fail")
    assert res2["status"] == "failed"
    assert job.consecutive_failures == 2
    assert not job.enabled

    # Subsequent trigger -> should report disabled
    res3 = scheduler.trigger_job("job_fail")
    assert res3["status"] == "disabled_or_not_found"

def test_scheduler_window_budget():
    scheduler = Scheduler()
    def task():
        return "ok"

    job = scheduler.register_job("job_budget", "* * * * *", task, window_budget_runs=2)
    
    assert scheduler.trigger_job("job_budget")["status"] == "success"
    assert scheduler.trigger_job("job_budget")["status"] == "success"
    
    # 3rd run exceeds budget ceiling (2)
    res = scheduler.trigger_job("job_budget")
    assert res["status"] == "budget_exhausted"
