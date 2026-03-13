Flaky tests are a common pain point — tests that sometimes pass and sometimes fail due to timing, network, or other non-deterministic factors. Users currently need a third-party plugin (`pytest-rerunfailures`) to retry them.

Add a built-in `@pytest.mark.flaky(retries=N)` marker that automatically retries a failing test up to N times before reporting it as failed. If the test passes on any retry, it should be reported as passed (but with a note that it needed retries). The retry count and which attempt passed should be visible in the test output.

Look at how existing markers like `@pytest.mark.skip` and `@pytest.mark.xfail` integrate with the runner and reporting system in `src/_pytest/skipping.py` for patterns to follow.
