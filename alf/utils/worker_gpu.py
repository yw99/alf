# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Explicit GPU assignment and supervision for single-host DDP workers."""

from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import re
import signal
import time


def resolve_worker_gpus(mapping, visible_devices, device_count,
                        distributed='multi-gpu', num_gpus_per_worker=1):
    """Resolve logical indices to CUDA visibility tokens, preserving UUIDs.

    ``None`` selects the legacy launcher and does not validate its settings.
    ``device_count`` must be CUDA's count under the original visibility mask.
    """
    if mapping is None:
        return None
    if distributed != 'multi-gpu' or num_gpus_per_worker != 1:
        raise ValueError('--worker_gpus requires --distributed=multi-gpu and '
                         '--num_gpus_per_ddp_worker=1')
    if not re.fullmatch(r'[0-9]+(?:,[0-9]+)*', mapping):
        raise ValueError('--worker_gpus must be a nonempty comma-separated '
                         'list of nonnegative logical GPU indices')
    devices = ([str(i) for i in range(device_count)] if visible_devices is None
               else visible_devices.split(','))
    if (device_count < 1 or len(devices) != device_count
            or len(set(devices)) != len(devices)):
        raise ValueError('CUDA_VISIBLE_DEVICES must identify distinct available '
                         'GPUs; repeat indices in --worker_gpus instead')
    indices = [int(i) for i in mapping.split(',')]
    if max(indices) >= device_count:
        raise ValueError(f'--worker_gpus index {max(indices)} is unavailable; '
                         f'only {device_count} GPU(s) are visible')
    return [devices[i] for i in indices]


@contextmanager
def temporary_environment(**values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# Only the owning worker may report through this endpoint. Forked environment
# children can inherit module state, but must not report as their parent rank.
_failure_owner = None
_failure_connection = None
_failure_reported = False


def report_worker_failure(exc):
    """Notify the coordinator of the first failure, before teardown can block."""
    global _failure_reported
    if isinstance(exc, SystemExit) and exc.code in (None, 0):
        return
    if (_failure_owner is None or _failure_owner[0] != os.getpid()
            or _failure_reported):
        return
    _failure_reported = True
    pid, rank = _failure_owner
    try:
        description = str(exc)[:1024]
    except Exception:
        description = '<exception message unavailable>'
    message = (f'DDP rank {rank} (PID {pid}): '
               f'{type(exc).__name__}: {description}')
    # One bounded message per dedicated pipe fits in a Linux atomic pipe write,
    # avoiding a partial payload if the worker dies while notifying the parent.
    payload = message.encode('utf-8', errors='replace')[:2048]
    try:
        _failure_connection.send_bytes(payload)
    except OSError:
        # Preserve the original exception if the coordinator already exited.
        pass


def _worker_entry(target, rank, world_size, args, failure_connection):
    global _failure_owner, _failure_connection, _failure_reported
    _failure_owner = (os.getpid(), rank)
    _failure_connection = failure_connection
    _failure_reported = False
    try:
        # Environment descendants belong to this worker's session so the
        # coordinator can clean them up even if their parent fails.
        os.setsid()
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        target(rank, world_size, *args)
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
            report_worker_failure(exc)
        raise
    finally:
        failure_connection.close()
        _failure_owner = None
        _failure_connection = None


def _signal_worker(process, sig):
    if process.pid is None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        # The process may still be importing modules, before setsid().
        if process.is_alive():
            try:
                os.kill(process.pid, sig)
            except ProcessLookupError:
                pass


def _cleanup_workers(processes):
    for process in processes:
        _signal_worker(process, signal.SIGTERM)
    deadline = time.monotonic() + 5
    for process in processes:
        if process.pid is not None:
            process.join(max(0, deadline - time.monotonic()))
    # Also kill descendants whose leader already exited; joining a leader alone
    # does not establish that its environment subprocesses have exited.
    for process in processes:
        _signal_worker(process, signal.SIGKILL)
    for process in processes:
        if process.pid is not None:
            process.join(timeout=1)


def run_workers(devices, target, args=(), with_queue=False):
    """Run one fresh process per mapping entry and propagate worker failures.

    Linux only. The caller owns rendezvous settings. With ``with_queue``, a
    managed queue is appended to target arguments and cleaned up on exit.
    GPU visibility is set *before spawn*, including for rank zero, because ALF
    imports can query CUDA before the target function runs.
    """
    if not hasattr(os, 'setsid'):
        raise ValueError('--worker_gpus currently requires POSIX process groups')
    ctx = mp.get_context('spawn')
    processes = []
    pipes = []
    manager = None
    previous_handlers = {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        if with_queue:
            manager = ctx.Manager()
            args = (*args, manager.Queue())
        for rank, device in enumerate(devices):
            reader, writer = ctx.Pipe(duplex=False)
            pipes.append((reader, writer))
            with temporary_environment(CUDA_VISIBLE_DEVICES=device):
                process = ctx.Process(target=_worker_entry,
                                      args=(target, rank, len(devices), args, writer),
                                      name=f'DDP_worker-{rank}')
                processes.append(process)
                try:
                    process.start()
                finally:
                    writer.close()
        pending = {process.sentinel: process for process in processes}
        channels = {reader: process
                    for (reader, _), process in zip(pipes, processes)}
        while pending:
            ready = wait([*channels, *pending])
            # Read notifications before exit codes to retain the first error
            # even if worker teardown raises a different exception afterwards.
            for connection in [item for item in ready if item in channels]:
                channels.pop(connection)
                try:
                    message = connection.recv_bytes(2048).decode(
                        'utf-8', errors='replace')
                except EOFError:
                    # EOF does not mean success: the exit sentinel still owns
                    # completion, including os._exit() and fatal signals.
                    continue
                finally:
                    connection.close()
                raise ChildProcessError(message)
            for sentinel in [item for item in ready if item in pending]:
                process = pending.pop(sentinel)
                process.join()
                if process.exitcode != 0:
                    raise ChildProcessError(
                        f'{process.name} (PID {process.pid}) exited with '
                        f'code {process.exitcode}')

    finally:
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            _cleanup_workers(processes)
        finally:
            for reader, writer in pipes:
                reader.close()
                writer.close()
            try:
                if manager is not None:
                    manager.shutdown()
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
