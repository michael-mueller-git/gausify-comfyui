"""Worker of gausify/isolated.py: runs one library function inside the core venv.

    <venv>/bin/python -I gausify/isolated_worker.py <job.pkl>

The job pickle holds {"module", "qualname", "args", "kwargs", "result"}; callables
of the caller are replaced by `isolated.CbRef` and called back over the protocol.
Protocol on stdout, one JSON object per line:
  {"type": "cb", "id": n, "args": b64, "kwargs": b64, "reply": bool}
  {"type": "result"} | {"type": "error", "message", "traceback", "exc": b64|null}
Replies to `reply` callbacks come on stdin: {"ok": b64} or {"err": text}.
Everything else printed goes to stderr.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import pickle
import sys
import time
import traceback

_OUT = None
POLL_INTERVAL = 0.25


def _b64(obj) -> str:
    return base64.b64encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _emit(**ev) -> None:
    _OUT.write(json.dumps(ev) + "\n")
    _OUT.flush()


class Proxy:
    def __init__(self, ref):
        self.id, self.reply = ref.id, ref.reply
        self._last = (0.0, None)

    def __call__(self, *args, **kwargs):
        if not self.reply:
            _emit(type="cb", id=self.id, args=_b64(args), kwargs=_b64(kwargs), reply=False)
            return None
        if not args and not kwargs:  # a cancel poll: throttle it
            t, v = self._last
            if time.monotonic() - t < POLL_INTERVAL:
                return v
        _emit(type="cb", id=self.id, args=_b64(args), kwargs=_b64(kwargs), reply=True)
        line = sys.stdin.readline()
        if not line:
            raise InterruptedError("the caller went away")
        msg = json.loads(line)
        if "err" in msg:
            raise RuntimeError(f"callback failed in the caller: {msg['err']}")
        v = pickle.loads(base64.b64decode(msg["ok"]))
        if not args and not kwargs:
            self._last = (time.monotonic(), v)
        return v


def _sub(x, CbRef):
    if isinstance(x, CbRef):
        return Proxy(x)
    return x


def main() -> int:
    global _OUT
    _OUT, sys.stdout = sys.stdout, sys.stderr
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.append(root)  # appended: the venv's packages keep precedence
    from gausify.isolated import CbRef

    job_path = sys.argv[1]
    try:
        with open(job_path, "rb") as f:
            job = pickle.load(f)
        fn = importlib.import_module(job["module"])
        for part in job["qualname"].split("."):
            fn = getattr(fn, part)
        args = [_sub(a, CbRef) for a in job["args"]]
        kwargs = {k: _sub(v, CbRef) for k, v in job["kwargs"].items()}
        result = fn(*args, **kwargs)
        # Containers the function may have filled in (stats dicts, caches): sent back.
        back = {
            "result": result,
            "args": [a if isinstance(a, (dict, list)) else None for a in args],
            "kwargs": {k: v for k, v in kwargs.items() if isinstance(v, (dict, list))},
        }
        with open(job["result"], "wb") as f:
            pickle.dump(back, f, protocol=pickle.HIGHEST_PROTOCOL)
        _emit(type="result")
        return 0
    except BaseException as e:  # noqa: BLE001
        try:
            exc = _b64(e)
            pickle.loads(base64.b64decode(exc))
        except Exception:  # noqa: BLE001 - not picklable
            exc = None
        _emit(type="error", message=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(), exc=exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
