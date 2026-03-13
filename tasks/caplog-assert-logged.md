Users frequently want to assert that a specific message was logged at a specific level during a test. Right now they have to manually loop through `caplog.records` and check each one, which is verbose and error-prone.

Add a convenience method to the `caplog` fixture that lets users assert a log message was recorded, matching on level and a message pattern. When the assertion fails, the error should be helpful — show what was expected vs what was actually logged.

Look at how `pytest.raises` and `pytest.warns` handle the `match=` parameter for API design inspiration. The method belongs in `LogCaptureFixture` in `src/_pytest/logging.py`. Make sure it plays nicely with `caplog.set_level()` and `caplog.clear()`.
