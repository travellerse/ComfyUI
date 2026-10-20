import pytest


@pytest.fixture(autouse=True)
def _reset_fused_attention_stats():
    """``fused_attention.stats`` is a module-level counter that outlives a test.

    Several suites drive the same kernel, so a case that asserts an exact call count would
    otherwise see whatever the previous file left behind. Both sides are reset; the module is
    only touched once some test has imported it, so suites that never load it pay nothing.
    """
    import sys

    def reset():
        module = sys.modules.get("comfy.ldm.qwen_image21.fused_attention")
        if module is not None:
            for key in module.stats:
                module.stats[key] = 0

    reset()
    yield
    reset()


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    """The real file database: WAL mode, both engines and their hooks, in ``tmp_path``."""
    from app.database import db as db_module

    db_path = str(tmp_path / "comfyui.db")
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    db_module._init_file_db(db_module.args.database_url)
    yield db_path
    db_module.Session.kw["bind"].dispose()
    db_module.WriteSession.kw["bind"].dispose()
    db_module._db_lock.release(force=True)
