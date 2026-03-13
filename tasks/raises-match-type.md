`pytest.raises(ValueError)` catches any subclass of `ValueError`, which is usually what you want. But sometimes users need to assert that the *exact* type was raised — not a subclass. This matters when testing exception hierarchies where parent and child exceptions have different semantics.

Right now users have to do `with pytest.raises(SomeError) as exc_info:` and then manually check `type(exc_info.value) is SomeError`, which is clunky and produces poor failure messages.

Add a way to `pytest.raises` to optionally enforce exact type matching. The default behavior must not change. The implementation lives in `RaisesContext` in `src/_pytest/python_api.py` — look at how the existing `match=` parameter works for the pattern to follow.
