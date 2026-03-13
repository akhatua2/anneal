Tests that accidentally hang (infinite loops, deadlocks, slow network calls) can block entire CI pipelines. Users want a simple way to set a per-test timeout so that a hanging test fails fast with a clear message instead of running forever.

Add a built-in `@pytest.mark.timeout(seconds)` marker that fails a test if it exceeds the given duration. The timeout should only apply during the test call itself, not setup/teardown. Think about how this integrates with pytest's marker and plugin system — look at how `src/_pytest/skipping.py` registers markers and hooks into the test runner for a pattern to follow.

Consider what happens on platforms where `signal.alarm` isn't available. The new plugin should be registered the same way other builtins are in `src/_pytest/config/__init__.py`.
