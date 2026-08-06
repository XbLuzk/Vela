from __future__ import annotations

import asyncio
import os
import signal
from contextlib import suppress


async def stop_subprocess(proc: asyncio.subprocess.Process) -> None:
    """Stop a subprocess and its process group, escalating when needed."""

    if proc.returncode is not None:
        return
    with suppress(ProcessLookupError):
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=1)
    except TimeoutError:
        with suppress(ProcessLookupError):
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        with suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=1)
