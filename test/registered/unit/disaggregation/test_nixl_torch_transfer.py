"""CPU tests for the experimental PyTorch transfer compatibility boundary."""

# Thread targets intentionally retain BaseException for process-control fault
# injection; nested contexts keep the interruption point visually explicit.
# ruff: noqa: BLE001, SIM117

from __future__ import annotations

import ast
import threading
import time
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sglang.srt.disaggregation.nixl.torch_transfer import (
    TorchTransferAgent,
    _load_transfer_api,
    _register_nixl_backend,
    _WorkHandle,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _WorkState(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    SUCCEEDED = COMPLETED
    FAILED = auto()
    CANCELLED = auto()


class _TransferOp(Enum):
    READ = "read"
    WRITE = "write"


class _Resource:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0
        self.close_errors: list[BaseException] = []

    def close(self) -> None:
        self.close_calls += 1
        if self.close_errors:
            raise self.close_errors.pop(0)
        self.closed = True


class _RawSpan:
    instances: ClassVar[list[_RawSpan]] = []

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.__class__.instances.append(self)


class _Registration(_Resource):
    def __init__(self, span: _RawSpan, name: str) -> None:
        super().__init__()
        self.span = span
        self.name = name
        self.region_calls = []

    def region(self, offset: int, nbytes: int, **kwargs):
        region = ("local", self.name, offset, nbytes, kwargs)
        self.region_calls.append(region)
        return region


class _Peer(_Resource):
    def __init__(self, name: str = "remote-endpoint") -> None:
        super().__init__()
        self.name = name
        self.endpoint_id = "remote-id"
        self.incarnation = "remote-incarnation"
        self.region_calls = []

    def unsafe_region(self, address: int, nbytes: int, **kwargs):
        region = ("remote", address, nbytes, kwargs)
        self.region_calls.append(region)
        return region


class _Plan(_Resource):
    def __init__(self, endpoint, kwargs: dict) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.kwargs = kwargs
        self.select_indices_calls = []
        self.submit_and_poll_calls = []

    def select_indices(
        self,
        operation,
        *,
        local_indices,
        remote_indices,
    ):
        self.select_indices_calls.append((operation, local_indices, remote_indices))
        raise AssertionError("one-shot SGLang transfers must not create IndexSelection")

    def submit_indices_and_poll(
        self,
        operation,
        *,
        local_indices,
        remote_indices,
        **kwargs,
    ):
        local_snapshot = np.asarray(local_indices, dtype=np.int32).copy()
        remote_snapshot = np.asarray(remote_indices, dtype=np.int32).copy()
        call = {
            "operation": operation,
            "local_indices": tuple(local_snapshot),
            "remote_indices": tuple(remote_snapshot),
            **kwargs,
        }
        self.submit_and_poll_calls.append(call)
        submission_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"max_polls", "timeout_ns"}
        }
        work = self.endpoint._submit(
            operation,
            self,
            {
                "local_indices": local_snapshot,
                "remote_indices": remote_snapshot,
                **submission_kwargs,
            },
        )
        endpoint = self.endpoint
        state = (
            endpoint.next_fused_state
            if endpoint.capabilities.fused_prevalidated_submit_poll
            else _WorkState.PENDING
        )
        work.done = state in (
            _WorkState.COMPLETED,
            _WorkState.FAILED,
            _WorkState.CANCELLED,
        )
        work.state = state
        work.error = endpoint.next_fused_error
        endpoint._raise_postcommit_error()
        return work, state


class _Selection:
    def __init__(self, plan, operation, local_indices, remote_indices) -> None:
        self.plan = plan
        self.operation = operation
        self.local_indices = np.asarray(local_indices, dtype=np.int32).copy()
        self.remote_indices = np.asarray(remote_indices, dtype=np.int32).copy()
        self.submit_and_poll_calls = []

    def submit_and_poll(self, **kwargs):
        call = dict(kwargs)
        self.submit_and_poll_calls.append(call)
        self.plan.submit_and_poll_calls.append((self, call))
        submission_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"max_polls", "timeout_ns"}
        }
        work = self.plan.endpoint._submit(
            self.operation,
            self.plan,
            {
                "local_indices": self.local_indices,
                "remote_indices": self.remote_indices,
                **submission_kwargs,
            },
        )
        endpoint = self.plan.endpoint
        state = (
            endpoint.next_fused_state
            if endpoint.capabilities.fused_prevalidated_submit_poll
            else _WorkState.PENDING
        )
        work.done = state in (
            _WorkState.COMPLETED,
            _WorkState.FAILED,
            _WorkState.CANCELLED,
        )
        work.state = state
        work.error = endpoint.next_fused_error
        endpoint._raise_postcommit_error()
        return work, state


class _Work(_Resource):
    def __init__(self, recovery_token=None) -> None:
        super().__init__()
        self.recovery_token = recovery_token
        self.done = False
        self._state = _WorkState.PENDING
        self.error = None
        self.cancel_completes = True
        self.test_error: Exception | None = None
        self.test_calls = 0
        self.state_hook = None

    @property
    def state(self) -> _WorkState:
        self.test_calls += 1
        if self.state_hook is not None:
            self.state_hook()
        if self.test_error is not None:
            raise self.test_error
        return self._state

    @state.setter
    def state(self, value: _WorkState) -> None:
        self._state = value

    def test(self) -> bool:
        self.test_calls += 1
        if self.test_error is not None:
            raise self.test_error
        return self.done

    def cancel(self) -> bool:
        if self.cancel_completes:
            self.done = True
            self.state = _WorkState.CANCELLED
        return self.cancel_completes

    def close(self) -> None:
        super().close()
        if self.closed:
            self.recovery_token = None


class _ProgressMode(Enum):
    MANUAL = auto()
    BACKGROUND = auto()


class _ThreadMode(Enum):
    CALLER_SERIALIZED = auto()
    SERIALIZED = auto()
    MULTIPLE = auto()


class _Endpoint(_Resource):
    instances: ClassVar[list[_Endpoint]] = []
    synchronous_notification_send = True
    constructor_cut: tuple[BaseException, bool, list[BaseException]] | None = None
    missing_capability: str | None = None

    def __init__(
        self,
        name: str,
        *,
        backend: str,
        endpoint_id: str,
        progress_mode: _ProgressMode,
        thread_mode: _ThreadMode,
        options: dict,
    ) -> None:
        super().__init__()
        self.name = name
        self.backend = backend
        self.endpoint_id = endpoint_id
        self.progress_mode = progress_mode
        self.thread_mode = thread_mode
        self.options = options
        self.capabilities = SimpleNamespace(
            notifications=True,
            attached_notifications=True,
            standalone_notifications=True,
            raw_spans=True,
            strided_regions=True,
            write=True,
            fused_prevalidated_submit_poll=True,
            synchronous_notification_send=self.__class__.synchronous_notification_send,
        )
        self.registrations = []
        self.peers = []
        self.plans = []
        self.bound_transfers = []
        self.transfer_slots = []
        self.submissions = []
        self.notifications = []
        self.synchronous_notifications = []
        self.incoming_notifications = []
        self.progress_calls = 0
        self.next_fused_result = False
        self.next_fused_state = _WorkState.PENDING
        self.next_fused_error = None
        self.poison_index_submit = False
        self.submit_attempts = 0
        self.submit_error: BaseException | None = None
        self.postcommit_error: BaseException | None = None
        self.postcommit_hook = None
        self.open_works_errors: list[BaseException] = []
        self.next_work_done = False
        self.next_work_state = _WorkState.PENDING
        self._open_works = []
        self.open_works_reads = 0
        self.construction_snapshot_reads = 0
        self.poison_construction_snapshots = False
        missing_capability = self.__class__.missing_capability
        if missing_capability is not None:
            setattr(self.capabilities, missing_capability, False)
        self.__class__.instances.append(self)
        constructor_cut = self.__class__.constructor_cut
        if constructor_cut is not None:
            self.__class__.constructor_cut = None
            error, mismatch, close_errors = constructor_cut
            self.close_errors.extend(close_errors)
            if mismatch:
                self.endpoint_id = f"foreign-{endpoint_id}"
            error.recovery_endpoint = self
            raise error

    @property
    def open_registrations(self):
        self._record_construction_snapshot()
        return tuple(item for item in self.registrations if not item.closed)

    @property
    def open_peers(self):
        self._record_construction_snapshot()
        return tuple(item for item in self.peers if not item.closed)

    @property
    def open_plans(self):
        self._record_construction_snapshot()
        return tuple(item for item in self.plans if not item.closed)

    @property
    def open_bound_transfers(self):
        self._record_construction_snapshot()
        return tuple(item for item in self.bound_transfers if not item.closed)

    @property
    def open_transfer_slots(self):
        self._record_construction_snapshot()
        return tuple(item for item in self.transfer_slots if not item.closed)

    def _record_construction_snapshot(self):
        self.construction_snapshot_reads += 1
        if self.poison_construction_snapshots:
            raise AssertionError("clean path read a Core construction snapshot")

    @property
    def open_works(self):
        self.open_works_reads += 1
        if self.open_works_errors:
            raise self.open_works_errors.pop(0)
        return tuple(work for work in self._open_works if not work.closed)

    def register(self, span, *, name):
        registration = _Registration(span, name)
        self.registrations.append(registration)
        return registration

    def export_metadata(self, registrations):
        assert registrations == list(self.open_registrations)
        return b"endpoint-metadata"

    def import_peer(self, metadata, **kwargs):
        assert metadata == b"peer-metadata"
        if kwargs:
            assert kwargs == {"expected_name": "decode-agent"}
        peer = _Peer()
        self.peers.append(peer)
        return peer

    def prepare(self, **kwargs):
        plan = _Plan(self, kwargs)
        self.plans.append(plan)
        return plan

    def _submit(self, operation, plan, kwargs):
        self.submit_attempts += 1
        if self.submit_error is not None:
            raise self.submit_error
        work = _Work(kwargs.get("recovery_token"))
        work.done = self.next_work_done
        work.state = self.next_work_state
        self.submissions.append((operation, plan, kwargs, work))
        self._open_works.append(work)
        return work

    def _raise_postcommit_error(self):
        if self.postcommit_hook is not None:
            self.postcommit_hook()
        error, self.postcommit_error = self.postcommit_error, None
        if error is not None:
            raise error

    def write(self, plan, **kwargs):
        work = self._submit("WRITE", plan, kwargs)
        self._raise_postcommit_error()
        return work

    def read(self, plan, **kwargs):
        work = self._submit("READ", plan, kwargs)
        self._raise_postcommit_error()
        return work

    def progress(self):
        self.progress_calls += 1

    def send_notification(self, peer, payload, *, recovery_token=None):
        work = _Work(recovery_token)
        self.notifications.append((peer, payload, recovery_token, work))
        self._open_works.append(work)
        self._raise_postcommit_error()
        return work

    def send_notification_sync(self, peer, payload):
        self.synchronous_notifications.append((peer, payload))

    def poll_notifications(self):
        result = self.incoming_notifications
        self.incoming_notifications = []
        return result


def _make_api():
    return SimpleNamespace(
        BACKEND_FACTORY_API_VERSION=2,
        BackendFactoryV2=object,
        Endpoint=_Endpoint,
        ProgressMode=_ProgressMode,
        RawSpan=_RawSpan,
        ThreadMode=_ThreadMode,
        TransferOp=_TransferOp,
        TransferPlan=_Plan,
        IndexSelection=_Selection,
        WorkState=_WorkState,
    )


def _reset_fake_api() -> None:
    _Endpoint.instances.clear()
    _Endpoint.constructor_cut = None
    _Endpoint.missing_capability = None
    _Endpoint.synchronous_notification_send = True
    _RawSpan.instances.clear()


def _make_uncreated_agent():
    _reset_fake_api()
    with patch(
        "sglang.srt.disaggregation.nixl.torch_transfer._load_transfer_api",
        return_value=_make_api(),
    ):
        return TorchTransferAgent(
            "local-endpoint",
            owner=object(),
            background_progress=False,
            endpoint_backend="fake",
        )


def _make_agent(
    *,
    background_progress: bool = False,
    synchronous_notification_send: bool = True,
):
    _reset_fake_api()
    _Endpoint.synchronous_notification_send = synchronous_notification_send
    api = _make_api()
    owner = object()
    with (
        patch(
            "sglang.srt.disaggregation.nixl.torch_transfer._load_transfer_api",
            return_value=api,
        ),
        patch(
            "sglang.srt.disaggregation.nixl.torch_transfer._register_nixl_backend"
        ) as register_nixl,
    ):
        agent = TorchTransferAgent(
            "local-endpoint",
            owner=owner,
            background_progress=background_progress,
            num_threads=8 if background_progress else 0,
        )
        agent.create_backend("UCX", {"num_threads": "8"})
    register_nixl.assert_called_once_with()
    return agent, _Endpoint.instances[-1], owner


def _register_and_add_peer(agent):
    registrations = agent.register_memory([(0x1000, 0x400, 0, "")], "VRAM")
    agent.add_remote_agent(b"peer-metadata", peer_name="decode-agent")
    return registrations


def test_raw_registration_keeps_owner_and_exports_metadata():
    agent, endpoint, owner = _make_agent(background_progress=True)

    registrations = agent.register_memory(
        [(0x1000, 0x100, 2, ""), (0x2000, 0x80, 0, "")], "VRAM"
    )

    assert endpoint.progress_mode is _ProgressMode.BACKGROUND
    assert endpoint.thread_mode is _ThreadMode.MULTIPLE
    assert endpoint.options == {
        "backends": ["UCX"],
        "backend_init_params": {"UCX": {"num_threads": "8"}},
        "transfer_backends": ["UCX"],
        "num_threads": 8,
    }
    assert len(registrations) == 2
    assert _RawSpan.instances[0].device == "cuda:2"
    assert _RawSpan.instances[0].owner is owner
    assert agent.get_agent_metadata() == b"endpoint-metadata"


def test_prepared_write_reuses_indexed_plan_and_attaches_notification():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.prep_xfer_dlist("", [(0x1000, 64, 0), (0x1040, 64, 0)], "VRAM")
    remote = agent.prep_xfer_dlist(
        "decode-agent", [(0x5000, 64, 0), (0x5040, 64, 0)], "VRAM"
    )

    source_indices = np.asarray([0, 1], dtype=np.int32)
    destination_indices = np.asarray([1, 0], dtype=np.int32)
    notification = bytearray(b"7_kv_0_1_0")
    first = agent.make_prepped_xfer(
        "WRITE",
        local,
        source_indices,
        remote,
        destination_indices,
        notification,
    )
    second = agent.make_prepped_xfer("WRITE", local, [1], remote, [0])

    assert len(endpoint.plans) == 1
    assert endpoint.plans[0].kwargs["indexed"] is True
    assert endpoint.submissions == []
    assert endpoint.submit_attempts == 0
    assert first.work is None
    assert second.work is None
    assert endpoint.plans[0].select_indices_calls == []
    assert first.local_indices is source_indices
    assert first.remote_indices is destination_indices
    assert source_indices.flags.writeable is False
    assert destination_indices.flags.writeable is False

    with pytest.raises(ValueError, match="read-only"):
        source_indices[:] = [1, 1]
    with pytest.raises(ValueError, match="read-only"):
        destination_indices[:] = [0, 0]
    notification[:] = b"changed_tag"

    assert agent.transfer(first) == "PROC"
    first_submission = endpoint.submissions[0][2]
    assert tuple(first_submission["local_indices"]) == (0, 1)
    assert tuple(first_submission["remote_indices"]) == (1, 0)
    assert first_submission["local_indices"].dtype == np.int32
    assert first_submission["remote_indices"].dtype == np.int32
    assert first_submission["notification"] == b"7_kv_0_1_0"
    assert endpoint.plans[0].submit_and_poll_calls[0] == {
        "operation": _TransferOp.WRITE,
        "local_indices": (0, 1),
        "remote_indices": (1, 0),
        "notification": b"7_kv_0_1_0",
        "max_polls": 0,
        "timeout_ns": None,
        "recovery_token": first,
    }
    assert first.local_indices is None
    assert first.remote_indices is None
    assert endpoint.plans[0].select_indices_calls == []
    assert endpoint.open_works_reads == 0
    assert first.work.test_calls == 0
    first.work.done = True
    first.work.state = _WorkState.SUCCEEDED
    assert agent.check_xfer_state(first) == "DONE"
    assert first.work.test_calls == 1
    assert first.work.closed is True

    assert agent.transfer(second) == "PROC"
    assert second.work.test_calls == 0
    assert endpoint.submit_attempts == 2
    assert agent.transfer(second) == "PROC"
    assert second.work.test_calls == 1
    assert endpoint.submit_attempts == 2
    second.work.done = True
    second.work.state = _WorkState.SUCCEEDED
    assert agent.check_xfer_state(second) == "DONE"
    assert second.work.test_calls == 2
    agent.release_dlist_handle(local)
    assert endpoint.plans[0].closed is True


@pytest.mark.parametrize(
    ("state", "error", "expected"),
    [
        (_WorkState.SUCCEEDED, None, "DONE"),
        (_WorkState.FAILED, RuntimeError("fused failure"), "ERR"),
    ],
)
def test_indexed_fused_submit_consumes_immediate_terminal_without_state_reread(
    state, error, expected
):
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    endpoint.next_fused_result = True
    endpoint.next_fused_state = state
    endpoint.next_fused_error = error
    endpoint.poison_index_submit = True
    local = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
    remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")

    handle = agent.make_prepped_xfer("WRITE", local, [0], remote, [0], b"fused")
    assert endpoint.submissions == []
    assert handle.work is None
    assert agent.transfer(handle) == expected
    assert endpoint.plans[-1].select_indices_calls == []
    assert endpoint.plans[-1].submit_and_poll_calls == [
        {
            "operation": _TransferOp.WRITE,
            "local_indices": (0,),
            "remote_indices": (0,),
            "notification": b"fused",
            "max_polls": 0,
            "timeout_ns": None,
            "recovery_token": handle,
        }
    ]
    assert handle.local_indices is None
    assert handle.remote_indices is None
    assert handle.work.test_calls == 0
    if expected == "ERR":
        assert handle.work.closed is True
        assert id(handle) not in agent._works
    assert agent.check_xfer_state(handle) == expected
    assert handle.work.test_calls == 0
    assert handle.work.closed is True


def test_work_state_is_one_authoritative_read_then_cached():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")
    assert handle.work is None
    assert endpoint.submissions == []
    endpoint.next_work_done = True
    endpoint.next_work_state = _WorkState.SUCCEEDED

    assert agent.transfer(handle) == "DONE"
    handle.work.test = MagicMock(side_effect=AssertionError("test() was queried"))
    assert handle.work.test_calls == 1
    assert handle.work.test.call_count == 0
    assert agent.check_xfer_state(handle) == "DONE"
    assert handle.work.test_calls == 1
    assert handle.work.closed is True


def test_indexed_submit_fallback_is_owned_by_core_api():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    endpoint.capabilities.fused_prevalidated_submit_poll = False
    local = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
    remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")

    handle = agent.make_prepped_xfer("WRITE", local, [0], remote, [0], b"fallback")
    assert endpoint.submissions == []
    assert endpoint.plans[-1].submit_and_poll_calls == []
    assert handle.first_state is None
    assert agent.transfer(handle) == "PROC"
    assert endpoint.plans[-1].select_indices_calls == []
    assert endpoint.plans[-1].submit_and_poll_calls == [
        {
            "operation": _TransferOp.WRITE,
            "local_indices": (0,),
            "remote_indices": (0,),
            "notification": b"fallback",
            "max_polls": 0,
            "timeout_ns": None,
            "recovery_token": handle,
        }
    ]
    assert handle.first_state is None
    assert handle.work.test_calls == 0
    assert agent.transfer(handle) == "PROC"
    assert handle.work.test_calls == 1
    assert endpoint.submit_attempts == 1


def test_plan_close_failure_quarantines_cache_and_peer_until_retry():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
    remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")
    handle = agent.make_prepped_xfer("WRITE", local, [0], remote, [0])
    assert agent.transfer(handle) == "PROC"
    handle.work.done = True
    handle.work.state = _WorkState.SUCCEEDED
    assert agent.check_xfer_state(handle) == "DONE"
    plan = endpoint.plans[-1]
    plan.close_errors.append(RuntimeError("partially committed close"))

    with pytest.raises(RuntimeError, match="partially committed"):
        agent.remove_remote_agent("decode-agent")

    assert not agent._plan_cache
    assert agent._closing_plans
    assert "decode-agent" in agent._retiring_peers
    with pytest.raises(RuntimeError, match="teardown is in progress"):
        agent.make_prepped_xfer("WRITE", local, [0], remote, [0])

    agent.remove_remote_agent("decode-agent")
    assert not agent._closing_plans
    assert "decode-agent" not in agent._retiring_peers
    assert plan.closed is True


def test_compact_strided_catalog_preserves_flattened_indices():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.prep_xfer_dlist("", [(0x1000, 32, 0, 64, 8)], "VRAM")
    remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0, 64, 8)], "VRAM")

    handle = agent.make_prepped_xfer("WRITE", local, [7, 1], remote, [2, 6], b"compact")

    registration = endpoint.registrations[0]
    assert registration.region_calls[-1][4] == {"stride": 64, "count": 8}
    assert endpoint.peers[0].region_calls[-1][3] == {
        "device": "cuda:0",
        "memory_type": "VRAM",
        "stride": 64,
        "count": 8,
    }
    assert endpoint.plans[-1].kwargs["local"] == local.regions
    assert endpoint.plans[-1].kwargs["remote"] == remote.regions
    assert endpoint.submissions == []
    assert agent.transfer(handle) == "PROC"
    assert tuple(endpoint.submissions[-1][2]["local_indices"]) == (7, 1)
    assert tuple(endpoint.submissions[-1][2]["remote_indices"]) == (2, 6)
    assert handle.work.test_calls == 0
    handle.work.done = True
    handle.work.state = _WorkState.SUCCEEDED
    agent.progress()
    assert handle.work.test_calls == 0
    assert agent.check_xfer_state(handle) == "DONE"
    assert handle.work.test_calls == 1
    assert handle.work.closed is True


def test_status_observation_failure_stays_nonterminal():
    agent, _, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")
    assert agent.transfer(handle) == "PROC"
    handle.work.test_error = RuntimeError("temporary status failure")

    assert agent.check_xfer_state(handle) == "PROC"
    assert handle.terminal_state is None
    assert handle.work.closed is False

    handle.work.test_error = None
    handle.work.done = True
    handle.work.state = _WorkState.SUCCEEDED
    agent.progress()
    assert handle.work.test_calls == 2
    assert agent.check_xfer_state(handle) == "DONE"
    assert handle.terminal_state == "DONE"
    assert handle.work.closed is True


def test_batch_progress_leaves_one_status_probe_per_active_work():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handles = [
        agent.initialize_xfer("WRITE", local, remote, "decode-agent") for _ in range(3)
    ]
    assert [agent.transfer(handle) for handle in handles] == ["PROC"] * 3
    handles[0].work.done = True
    handles[0].work.state = _WorkState.SUCCEEDED

    agent.progress()
    assert [handle.work.test_calls for handle in handles] == [1, 1, 1]
    states = [agent.check_xfer_state(handle) for handle in handles]

    assert endpoint.progress_calls == 1
    assert states == ["DONE", "PROC", "PROC"]
    assert [handle.work.test_calls for handle in handles] == [2, 2, 2]
    assert handles[0].work.closed is True
    assert handles[0].owned_plan.closed is True
    assert handles[1].work.closed is False
    assert handles[1].owned_plan.closed is False
    assert handles[2].work.closed is False
    assert handles[2].owned_plan.closed is False
    assert set(agent._works) == {id(handles[1]), id(handles[2])}

    handles[1].work.done = True
    handles[1].work.state = _WorkState.SUCCEEDED
    agent.progress()
    assert [handle.work.test_calls for handle in handles] == [2, 2, 2]
    states = [agent.check_xfer_state(handle) for handle in handles[1:]]

    assert endpoint.progress_calls == 2
    assert states == ["DONE", "PROC"]
    assert [handle.work.test_calls for handle in handles] == [2, 3, 3]
    assert handles[1].work.closed is True
    assert handles[1].owned_plan.closed is True
    assert handles[2].work.closed is False
    assert handles[2].owned_plan.closed is False
    assert set(agent._works) == {id(handles[2])}


def test_detached_notification_work_is_still_reaped_by_progress():
    agent, endpoint, _ = _make_agent(synchronous_notification_send=False)
    _register_and_add_peer(agent)
    handle = agent.send_notif("decode-agent", b"abort")
    assert endpoint.notifications[-1][2] is handle
    assert endpoint.open_works_reads == 0
    handle.work.done = True
    handle.work.state = _WorkState.SUCCEEDED

    agent.progress()

    assert endpoint.progress_calls == 1
    assert handle.work.test_calls == 1
    assert handle.work.closed is True
    assert handle.terminal_state == "DONE"
    assert agent._works == {}
    assert agent._detached_works == []


def test_uncancellable_sibling_is_retained_then_reaped_when_terminal():
    agent, _, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    failed, sibling, foreground = [
        agent.initialize_xfer("WRITE", local, remote, "decode-agent") for _ in range(3)
    ]
    assert [agent.transfer(handle) for handle in (failed, sibling, foreground)] == [
        "PROC",
        "PROC",
        "PROC",
    ]
    failed.work.done = True
    failed.work.state = _WorkState.FAILED
    sibling.work.cancel_completes = False

    agent.cancel_handles([failed, sibling])

    assert failed.work.closed is True
    assert sibling.work.closed is False
    assert id(sibling) in agent._works
    assert id(sibling) in agent._retained_work_ids
    sweep_thread = agent._sweep_thread
    assert sweep_thread is not None
    foreground_probe_count = foreground.work.test_calls

    sibling.work.done = True
    sibling.work.state = _WorkState.SUCCEEDED
    deadline = time.monotonic() + 1
    while not sibling.work.closed and time.monotonic() < deadline:
        time.sleep(0.01)

    assert sibling.work.closed is True
    sweep_thread.join(timeout=1)
    assert sweep_thread.is_alive() is False
    assert agent._retained_work_ids == set()
    assert foreground.work.test_calls == foreground_probe_count
    foreground.work.done = True
    foreground.work.state = _WorkState.SUCCEEDED
    assert agent.check_xfer_state(foreground) == "DONE"


def test_cancel_handles_and_wait_does_not_return_before_uncancellable_work_is_safe():
    agent, _, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    failed, sibling = [
        agent.initialize_xfer("WRITE", local, remote, "decode-agent") for _ in range(2)
    ]
    assert agent.transfer(failed) == "PROC"
    assert agent.transfer(sibling) == "PROC"
    failed.work.done = True
    failed.work.state = _WorkState.FAILED
    sibling.work.cancel_completes = False

    def complete_sibling() -> None:
        time.sleep(0.01)
        sibling.work.done = True
        sibling.work.state = _WorkState.SUCCEEDED

    completion = threading.Thread(target=complete_sibling)
    completion.start()
    agent.cancel_handles_and_wait([failed, sibling])
    completion.join(timeout=1)

    assert completion.is_alive() is False
    assert failed.work.closed is True
    assert sibling.work.closed is True
    assert agent._works == {}


def test_background_progress_does_not_call_manual_progress():
    agent, endpoint, _ = _make_agent(background_progress=True)
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")

    class _UnexpectedLock:
        def __enter__(self):
            raise AssertionError("empty background progress acquired the lock")

        def __exit__(self, *args):
            return False

    with patch.object(agent, "_lock", _UnexpectedLock()):
        agent.progress()

    assert endpoint.progress_calls == 0
    with pytest.raises(RuntimeError, match="has not been posted"):
        agent.check_xfer_state(handle)
    assert endpoint.submissions == []
    assert agent.transfer(handle) == "PROC"
    assert handle.work.test_calls == 1

    handle.work.done = True
    handle.work.state = _WorkState.SUCCEEDED
    agent.progress()

    assert endpoint.progress_calls == 0
    assert handle.work.test_calls == 1
    assert agent.check_xfer_state(handle) == "DONE"
    assert handle.work.test_calls == 2
    assert handle.work.closed is True
    assert handle.owned_plan.closed is True


def test_multiple_mode_admission_gate_drains_progress_before_close():
    agent, endpoint, _ = _make_agent(background_progress=False)
    progress_entered = threading.Event()
    progress_release = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    failures: list[BaseException] = []
    original_progress = endpoint.progress

    def blocking_progress():
        progress_entered.set()
        if not progress_release.wait(1):
            raise RuntimeError("test progress release timed out")
        original_progress()

    endpoint.progress = blocking_progress

    def run_progress():
        try:
            agent.progress()
        except BaseException as error:
            failures.append(error)

    def run_close():
        close_started.set()
        try:
            agent.close()
        except BaseException as error:
            failures.append(error)
        finally:
            close_finished.set()

    progress_thread = threading.Thread(target=run_progress)
    close_thread = threading.Thread(target=run_close)
    progress_thread.start()
    assert progress_entered.wait(1)
    close_thread.start()
    assert close_started.wait(1)
    assert not close_finished.wait(0.05)
    progress_release.set()
    progress_thread.join(timeout=1)
    close_thread.join(timeout=1)

    assert not progress_thread.is_alive()
    assert not close_thread.is_alive()
    assert failures == []
    assert endpoint.closed is True


def test_distinct_handle_submits_overlap_without_adapter_locks():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
    remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")
    handles = [
        agent.make_prepped_xfer("WRITE", local, [0], remote, [0]) for _ in range(2)
    ]
    barrier = threading.Barrier(2)
    counter_lock = threading.Lock()
    active = 0
    max_active = 0
    original_submit = endpoint._submit

    def overlapping_submit(operation, plan, kwargs):
        nonlocal active, max_active
        assert not agent._lock._is_owned()
        assert not kwargs["recovery_token"]._state_lock.locked()
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=1)
            return original_submit(operation, plan, kwargs)
        finally:
            with counter_lock:
                active -= 1

    endpoint._submit = overlapping_submit
    results = []
    failures = []

    def submit(handle):
        try:
            results.append(agent.transfer(handle))
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=submit, args=(handle,)) for handle in handles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert sorted(results) == ["PROC", "PROC"]
    assert max_active == 2


def test_distinct_handle_status_refreshes_overlap_without_adapter_locks():
    agent, _, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handles = [
        agent.initialize_xfer("WRITE", local, remote, "decode-agent") for _ in range(2)
    ]
    assert [agent.transfer(handle) for handle in handles] == ["PROC", "PROC"]
    barrier = threading.Barrier(2)
    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    def make_hook(handle):
        def hook():
            nonlocal active, max_active
            assert not agent._lock._is_owned()
            assert not handle._state_lock.locked()
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                barrier.wait(timeout=1)
            finally:
                with counter_lock:
                    active -= 1

        return hook

    for handle in handles:
        handle.work.state_hook = make_hook(handle)
    results = []
    failures = []

    def poll(handle):
        try:
            results.append(agent.check_xfer_state(handle))
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=poll, args=(handle,)) for handle in handles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert sorted(results) == ["PROC", "PROC"]
    assert max_active == 2


def test_teardown_gate_never_refreshes_status_under_agent_lock():
    agent, _, _ = _make_agent()
    registrations = _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")
    assert agent.transfer(handle) == "PROC"
    with handle._state_lock:
        handle.submission_interrupted = True
    probes_before = handle.work.test_calls
    open_works_before = agent._endpoint.open_works_reads
    handle.work.state_hook = lambda: pytest.fail(
        "teardown gate must not perform a Core status read"
    )

    with pytest.raises(RuntimeError, match="submission.*unsettled"):
        agent.deregister_memory(registrations)

    assert handle.work.test_calls == probes_before
    assert agent._endpoint.open_works_reads == open_works_before


def test_concurrent_interrupted_recovery_has_one_core_snapshot_owner():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    handle = _WorkHandle(phase="recovery_pending")
    handle.submission_interrupted = True
    agent._track_work(handle)
    recovered = _Work(handle)
    endpoint._open_works.append(recovered)
    entered = threading.Event()
    release = threading.Event()
    original_open_works = type(endpoint).open_works.fget

    def blocking_open_works(instance):
        entered.set()
        assert release.wait(1)
        return original_open_works(instance)

    results = []
    failures = []

    def recover():
        try:
            results.append(agent._recover_interrupted_post_work(handle))
        except BaseException as error:
            failures.append(error)

    with patch.object(type(endpoint), "open_works", property(blocking_open_works)):
        owner = threading.Thread(target=recover)
        contender = threading.Thread(target=recover)
        owner.start()
        assert entered.wait(1)
        contender.start()
        contender.join(timeout=1)
        release.set()
        owner.join(timeout=1)

    assert not owner.is_alive()
    assert not contender.is_alive()
    assert failures == []
    assert sorted(results) == ["pending", "recovered"]
    assert endpoint.open_works_reads == 1
    assert handle.work is recovered
    assert handle.phase == "posted"


def test_manual_progress_does_not_hold_adapter_state_lock():
    agent, endpoint, _ = _make_agent(background_progress=False)
    observed = []

    def inspect_progress_lock():
        observed.append(agent._lock._is_owned())

    endpoint.progress = inspect_progress_lock
    agent.progress()
    assert observed == [False]


def test_close_adopts_authoritative_closed_state_before_child_queries():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    snapshots_before = endpoint.construction_snapshot_reads
    endpoint.poison_construction_snapshots = True
    endpoint.closed = True

    agent.close()

    assert endpoint.construction_snapshot_reads == snapshots_before
    assert agent._endpoint is None
    assert agent._works == {}
    assert agent._peers == {}


def test_close_commit_lost_return_converges_on_retry():
    agent, endpoint, _ = _make_agent()

    def committed_then_interrupted():
        endpoint.close_calls += 1
        endpoint.closed = True
        raise KeyboardInterrupt("close return lost")

    endpoint.close = committed_then_interrupted
    with pytest.raises(KeyboardInterrupt, match="close return lost"):
        agent.close()

    assert agent._endpoint is None
    assert endpoint.close_calls == 1
    agent.close()
    assert endpoint.close_calls == 1


def test_background_progress_reaps_nonempty_detached_work():
    agent, endpoint, _ = _make_agent(
        background_progress=True, synchronous_notification_send=False
    )
    _register_and_add_peer(agent)
    handle = agent.send_notif("decode-agent", b"abort")
    handle.work.done = True
    handle.work.state = _WorkState.SUCCEEDED

    agent.progress()

    assert endpoint.progress_calls == 0
    assert handle.work.test_calls == 1
    assert handle.work.closed is True
    assert agent._detached_works == []


def test_conn_enables_background_progress_for_prefill_and_decode_roles():
    repo_root = Path(__file__).resolve().parents[4]
    source = (repo_root / "python/sglang/srt/disaggregation/nixl/conn.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TorchTransferAgent"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert isinstance(keywords["background_progress"], ast.Constant)
    assert keywords["background_progress"].value is True
    assert isinstance(keywords["num_threads"], ast.Name)
    assert keywords["num_threads"].id == "num_threads"

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "num_threads"
            for target in node.targets
        )
    ]
    role_assignment = next(
        node for node in assignments if isinstance(node.value, ast.IfExp)
    )
    assert isinstance(role_assignment.value.body, ast.Constant)
    assert role_assignment.value.body.value == 8
    assert isinstance(role_assignment.value.orelse, ast.Constant)
    assert role_assignment.value.orelse.value == 0


def test_registration_names_do_not_collide_after_deregister():
    agent, endpoint, _ = _make_agent()
    first = agent.register_memory([(0x1000, 0x100, 0, "")], "VRAM")
    second = agent.register_memory([(0x2000, 0x100, 0, "")], "VRAM")
    agent.deregister_memory(first)
    third = agent.register_memory([(0x3000, 0x100, 0, "")], "VRAM")

    assert second[0].name == "sglang_region_1"
    assert third[0].name == "sglang_region_2"
    assert len({registration.name for registration in endpoint.registrations}) == 3
    assert agent._registration_groups == [second, third]


@pytest.mark.parametrize("operation", ["READ", "WRITE"])
def test_one_off_transfer_orients_local_and_remote_regions(operation):
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1080, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5080, 32, 0)], "VRAM")
    source, destination = (remote, local) if operation == "READ" else (local, remote)

    handle = agent.initialize_xfer(
        operation, source, destination, "decode-agent", b"done"
    )

    plan = endpoint.plans[-1]
    assert plan.kwargs["indexed"] is False
    assert plan.kwargs["local"][0][0] == "local"
    assert plan.kwargs["remote"][0][0] == "remote"
    assert endpoint.submissions == []
    assert agent.transfer(handle) == "PROC"
    assert endpoint.submissions[-1][0] == operation
    assert endpoint.submissions[-1][2]["notification"] == b"done"
    assert endpoint.submissions[-1][2]["recovery_token"] is handle
    assert endpoint.open_works_reads == 0
    handle.work.done = True
    handle.work.state = _WorkState.SUCCEEDED
    assert agent.check_xfer_state(handle) == "DONE"
    assert plan.closed is True


def test_standalone_notifications_preserve_peer_alias():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    notification = SimpleNamespace(
        source_endpoint_id="remote-id",
        source_incarnation="remote-incarnation",
        payload=b"9_aux",
    )
    endpoint.incoming_notifications.append(notification)

    payload = b"abort"
    result = agent.send_notif("decode-agent", payload)

    assert agent.get_new_notifs() == {"decode-agent": [b"9_aux"]}
    assert endpoint.progress_calls == 0
    assert result is None
    assert endpoint.synchronous_notifications[0][1] is payload
    assert endpoint.notifications == []
    assert agent._works == {}
    assert agent._detached_works == []


def test_notification_without_imported_source_uses_explicit_sentinel():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    endpoint.incoming_notifications.append(
        SimpleNamespace(
            source_endpoint_id=None,
            source_incarnation=None,
            payload=b"9_aux",
        )
    )

    assert agent.get_new_notifs() == {"<unknown-pytorch-transfer-source>": [b"9_aux"]}


def test_notification_from_stale_incarnation_is_rejected():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    endpoint.incoming_notifications.append(
        SimpleNamespace(
            source_endpoint_id="remote-id",
            source_incarnation="old-incarnation",
            payload=b"9_aux",
        )
    )

    with pytest.raises(RuntimeError, match="unknown or stale"):
        agent.get_new_notifs()


def test_notification_batches_group_payloads_and_accept_anonymous_source():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)

    class CountingAliases(dict):
        lookups = 0

        def __getitem__(self, key):
            self.lookups += 1
            return super().__getitem__(key)

    aliases = CountingAliases(agent._source_aliases)
    agent._source_aliases = aliases
    endpoint.poll_notification_batches = lambda: [
        SimpleNamespace(
            source_endpoint_id="remote-id",
            source_incarnation="remote-incarnation",
            payloads=(b"first", b"second"),
        ),
        SimpleNamespace(
            source_endpoint_id=None,
            source_incarnation=None,
            payloads=(b"anonymous",),
        ),
    ]

    def reject_scalar_poll():
        raise AssertionError("grouped receive must not retry scalar polling")

    endpoint.poll_notifications = reject_scalar_poll

    assert agent.get_new_notifs() == {
        "decode-agent": [b"first", b"second"],
        "<unknown-pytorch-transfer-source>": [b"anonymous"],
    }
    assert aliases.lookups == 1


def test_notification_batch_from_stale_incarnation_is_rejected():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    endpoint.poll_notification_batches = lambda: [
        SimpleNamespace(
            source_endpoint_id="remote-id",
            source_incarnation="old-incarnation",
            payloads=(b"stale",),
        )
    ]

    with pytest.raises(RuntimeError, match="unknown or stale"):
        agent.get_new_notifs()


def test_notification_batch_failure_is_never_retried_as_scalar():
    agent, endpoint, _ = _make_agent()
    batch_calls = 0
    scalar_calls = 0

    def fail_batch_poll():
        nonlocal batch_calls
        batch_calls += 1
        raise RuntimeError("destructive batch failure")

    def count_scalar_poll():
        nonlocal scalar_calls
        scalar_calls += 1
        return []

    endpoint.poll_notification_batches = fail_batch_poll
    endpoint.poll_notifications = count_scalar_poll

    with pytest.raises(RuntimeError, match="destructive batch failure"):
        agent.get_new_notifs()
    assert batch_calls == 1
    assert scalar_calls == 0


def test_active_work_prevents_unsafe_close():
    agent, _, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")
    assert agent.transfer(handle) == "PROC"
    handle.work.cancel_completes = False

    with pytest.raises(RuntimeError, match="active transfer"):
        agent.close()

    handle.work.done = True
    handle.work.state = _WorkState.CANCELLED
    agent.close()


def test_prepared_release_and_agent_close_never_post():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local_prepared = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
    remote_prepared = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")
    indexed = agent.make_prepped_xfer(
        "WRITE", local_prepared, [0], remote_prepared, [0], b"never-posted"
    )
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    owned = agent.initialize_xfer(
        "WRITE", local, remote, "decode-agent", b"also-never-posted"
    )

    agent.release_xfer_handle(owned)
    assert owned.owned_plan.closed is True
    assert endpoint.submit_attempts == 0
    assert indexed.local_indices == (0,)
    assert indexed.remote_indices == (0,)
    agent.close()

    assert indexed.phase == "released"
    assert indexed.local_indices is None
    assert indexed.remote_indices is None
    assert endpoint.submit_attempts == 0
    assert all(plan.closed for plan in endpoint.plans)


def test_post_failure_is_terminal_and_never_retried():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
    remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")
    handle = agent.make_prepped_xfer("WRITE", local, [0], remote, [0])
    endpoint.submit_error = RuntimeError("post failed")

    with pytest.raises(RuntimeError, match="post failed"):
        agent.transfer(handle)
    assert endpoint.submit_attempts == 1
    assert handle.terminal_state == "ERR"
    assert handle.phase == "released"
    assert agent.transfer(handle) == "ERR"
    assert endpoint.submit_attempts == 1


@pytest.mark.parametrize("indexed", [True, False], ids=["indexed", "one-off"])
def test_postcommit_base_exception_recovers_exact_core_work(indexed):
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    endpoint.postcommit_error = KeyboardInterrupt("submit return interrupted")

    if indexed:
        endpoint.next_fused_state = _WorkState.COMPLETED
        local = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
        remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")
        handle = agent.make_prepped_xfer("WRITE", local, [0], remote, [0])
    else:
        endpoint.next_work_done = True
        endpoint.next_work_state = _WorkState.COMPLETED
        local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
        remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
        handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")

    with pytest.raises(KeyboardInterrupt, match="submit return interrupted"):
        agent.transfer(handle)

    recovered = endpoint.submissions[-1][3]
    assert handle.work is recovered
    assert recovered.close_calls == 1
    assert recovered.closed is True
    assert endpoint.open_works == ()
    assert endpoint.open_works_reads == 2
    assert endpoint.submit_attempts == 1
    assert endpoint.submissions[-1][2]["recovery_token"] is handle
    assert handle.terminal_state == "ERR"
    assert handle.phase == "released"
    assert id(handle) not in agent._works
    assert agent.transfer(handle) == "ERR"
    assert endpoint.submit_attempts == 1
    if indexed:
        assert endpoint.plans[-1].closed is False
    else:
        assert handle.owned_plan.close_calls == 1
        assert handle.owned_plan.closed is True


class _IdentityBombHandle(_WorkHandle):
    def __hash__(self):
        raise AssertionError("recovery token was hashed")

    def __eq__(self, other):
        raise AssertionError("recovery token equality was evaluated")

    def __bool__(self):
        raise AssertionError("recovery token truthiness was evaluated")


class _UnprintableInterrupt(KeyboardInterrupt):
    def __str__(self):
        raise RuntimeError("exception formatting is hostile")


def test_interrupted_post_ignores_distractor_and_uses_identity_only():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    distractor = _Work(object())
    endpoint._open_works.append(distractor)
    endpoint.next_work_done = True
    endpoint.next_work_state = _WorkState.COMPLETED
    endpoint.postcommit_error = KeyboardInterrupt("submit return interrupted")
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    with patch(
        "sglang.srt.disaggregation.nixl.torch_transfer._WorkHandle",
        _IdentityBombHandle,
    ):
        handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")

    with pytest.raises(KeyboardInterrupt, match="submit return interrupted"):
        agent.transfer(handle)

    recovered = endpoint.submissions[-1][3]
    assert endpoint.submissions[-1][2]["recovery_token"] is handle
    assert recovered.closed is True
    assert distractor.closed is False
    assert handle.terminal_state == "ERR"
    assert handle.phase == "released"


def test_ambiguous_token_recovery_is_retained_and_retryable():
    agent, endpoint, _ = _make_agent()
    registrations = _register_and_add_peer(agent)
    endpoint.postcommit_error = KeyboardInterrupt("submit return interrupted")
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")
    duplicate = _Work(handle)

    def add_duplicate():
        endpoint._open_works.append(duplicate)

    endpoint.postcommit_hook = add_duplicate
    with patch.object(agent, "_start_retained_work_sweeper") as start_sweeper:
        assert agent.transfer(handle) == "PROC"

        assert handle.phase == "recovery_pending"
        assert handle.work is None
        assert id(handle) in agent._works
        assert id(handle) in agent._retained_work_ids
        assert start_sweeper.call_count == 1
        assert agent.transfer(handle) == "PROC"
        with pytest.raises(RuntimeError, match="recovery is pending"):
            agent.release_xfer_handle(handle)
        with pytest.raises(RuntimeError, match="submission.*unsettled"):
            agent.deregister_memory(registrations)
        with pytest.raises(RuntimeError, match="submission.*unsettled"):
            agent.remove_remote_agent("decode-agent")
        with pytest.raises(RuntimeError, match="active transfer"):
            agent.close()

    duplicate.close()
    assert agent.transfer(handle) == "ERR"
    assert endpoint.submit_attempts == 1
    assert handle.phase == "released"
    assert endpoint.submissions[-1][3].closed is True
    agent.remove_remote_agent("decode-agent")
    agent.deregister_memory(registrations)


def test_interrupted_open_works_lookup_retries_without_resubmission():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    endpoint.postcommit_error = KeyboardInterrupt("submit return interrupted")
    endpoint.open_works_errors.append(RuntimeError("snapshot interrupted"))
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")

    with patch.object(agent, "_start_retained_work_sweeper"):
        assert agent.transfer(handle) == "PROC"
        assert handle.phase == "recovery_pending"
        assert agent.transfer(handle) == "ERR"

    assert endpoint.submit_attempts == 1
    assert endpoint.open_works_reads == 2
    assert handle.phase == "released"
    assert "KeyboardInterrupt: submit return interrupted" in str(
        agent.pop_xfer_error(handle)
    )


def test_exact_recovery_keeps_teardown_blocked_until_work_is_terminal():
    agent, endpoint, _ = _make_agent()
    registrations = _register_and_add_peer(agent)
    endpoint.postcommit_error = KeyboardInterrupt("submit return interrupted")

    def make_recovered_work_uncancellable():
        endpoint.submissions[-1][3].cancel_completes = False

    endpoint.postcommit_hook = make_recovered_work_uncancellable
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")

    with patch.object(agent, "_start_retained_work_sweeper"):
        assert agent.transfer(handle) == "PROC"
        assert handle.phase == "posted"
        assert handle.work.closed is False
        with pytest.raises(RuntimeError, match="submission.*unsettled"):
            agent.deregister_memory(registrations)
        with pytest.raises(RuntimeError, match="submission.*unsettled"):
            agent.remove_remote_agent("decode-agent")

        handle.work.done = True
        handle.work.state = _WorkState.COMPLETED
        assert agent.transfer(handle) == "ERR"

    assert endpoint.submit_attempts == 1
    assert handle.work.closed is True
    assert "KeyboardInterrupt: submit return interrupted" in str(
        agent.pop_xfer_error(handle)
    )
    agent.remove_remote_agent("decode-agent")
    agent.deregister_memory(registrations)


def test_recovery_precedes_hostile_exception_diagnostics_and_batch_owns_handle():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    endpoint.postcommit_error = _UnprintableInterrupt()

    def make_recovered_work_uncancellable():
        endpoint.submissions[-1][3].cancel_completes = False

    endpoint.postcommit_hook = make_recovered_work_uncancellable
    handles = []
    agent.begin_handle_batch(handles)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")

    with patch.object(agent, "_start_retained_work_sweeper"):
        assert handles == [handle]
        assert agent.transfer(handle) == "PROC"
        assert endpoint.submit_attempts == 1
        assert handle.work.closed is False

        handle.work.done = True
        handle.work.state = _WorkState.COMPLETED
        assert agent.check_xfer_state(handle) == "ERR"

    agent.end_handle_batch(handles)
    assert handle.work.closed is True
    assert "diagnostic formatting unavailable" in str(agent.pop_xfer_error(handle))


def test_async_notification_lost_return_recovers_by_token():
    agent, endpoint, _ = _make_agent(synchronous_notification_send=False)
    _register_and_add_peer(agent)
    distractor = _Work(object())
    endpoint._open_works.append(distractor)
    endpoint.postcommit_error = KeyboardInterrupt("notification return interrupted")

    with pytest.raises(KeyboardInterrupt, match="notification return interrupted"):
        agent.send_notif("decode-agent", b"abort")

    _, payload, token, work = endpoint.notifications[-1]
    assert payload == b"abort"
    assert token.work is work
    assert token.terminal_state == "ERR"
    assert token.phase == "released"
    assert work.closed is True
    assert distractor.closed is False
    assert endpoint.open_works_reads == 1
    assert agent._works == {}
    assert agent._detached_works == []


def test_async_notification_tracking_interruption_rolls_back_hidden_handle():
    agent, endpoint, _ = _make_agent(synchronous_notification_send=False)
    _register_and_add_peer(agent)
    original_track = agent._track_work

    def store_then_interrupt(handle):
        original_track(handle)
        raise KeyboardInterrupt("tracking return interrupted")

    with patch.object(agent, "_track_work", side_effect=store_then_interrupt):
        with pytest.raises(KeyboardInterrupt, match="tracking return interrupted"):
            agent.send_notif("decode-agent", b"abort")

    assert endpoint.notifications == []
    assert agent._works == {}
    assert agent._detached_works == []


def test_prepared_plan_teardown_quarantines_then_rejects_stale_post():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
    remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")
    handle = agent.make_prepped_xfer("WRITE", local, [0], remote, [0])
    plan = endpoint.plans[-1]

    with pytest.raises(RuntimeError, match="referenced by a transfer handle"):
        agent.remove_remote_agent("decode-agent")
    assert plan.closed is False
    assert "decode-agent" in agent._retiring_peers
    assert endpoint.submit_attempts == 0

    with pytest.raises(RuntimeError, match="teardown is in progress"):
        agent.transfer(handle)
    assert endpoint.submit_attempts == 0
    assert endpoint.open_works_reads == 0
    assert handle.terminal_state == "ERR"
    agent.remove_remote_agent("decode-agent")
    assert plan.closed is True
    assert "decode-agent" not in agent._retiring_peers


@pytest.mark.parametrize("version", [None, True, 1])
def test_core_factory_v2_gate_rejects_absent_bool_and_old_versions(version):
    api = _make_api()
    api.BACKEND_FACTORY_API_VERSION = version

    with patch(
        "sglang.srt.disaggregation.nixl.torch_transfer.importlib.import_module",
        return_value=api,
    ):
        with pytest.raises(ImportError, match="FactoryV2"):
            _load_transfer_api()


def test_core_factory_v2_gate_requires_marker_and_accepts_newer_core():
    api = _make_api()
    del api.BackendFactoryV2
    with patch(
        "sglang.srt.disaggregation.nixl.torch_transfer.importlib.import_module",
        return_value=api,
    ):
        with pytest.raises(ImportError, match="FactoryV2"):
            _load_transfer_api()

    api = _make_api()
    api.BACKEND_FACTORY_API_VERSION = 3
    with patch(
        "sglang.srt.disaggregation.nixl.torch_transfer.importlib.import_module",
        return_value=api,
    ):
        assert _load_transfer_api() is api


@pytest.mark.parametrize("version", [None, True, 1, 3])
def test_provider_factory_v2_gate_requires_exact_integer_abi(version):
    register = MagicMock()
    provider = SimpleNamespace(
        TORCH_TRANSFER_FACTORY_API_VERSION=version,
        register_torch_backend=register,
    )
    with patch(
        "sglang.srt.disaggregation.nixl.torch_transfer.importlib.import_module",
        return_value=provider,
    ):
        with pytest.raises(ImportError, match="exact FactoryV2"):
            _register_nixl_backend()
    register.assert_not_called()

    provider.TORCH_TRANSFER_FACTORY_API_VERSION = 2
    with patch(
        "sglang.srt.disaggregation.nixl.torch_transfer.importlib.import_module",
        return_value=provider,
    ):
        _register_nixl_backend()
    register.assert_called_once_with()


@pytest.mark.parametrize("mismatch", [False, True], ids=["exact", "foreign"])
def test_constructor_recovery_is_identity_gated_and_cleanup_is_retryable(mismatch):
    class HostileCleanup(RuntimeError):
        def __str__(self):
            raise AssertionError("cleanup diagnostics evaluated __str__")

    agent = _make_uncreated_agent()
    error = KeyboardInterrupt("constructor cut")
    _Endpoint.constructor_cut = (
        error,
        mismatch,
        (
            [AssertionError("mismatched endpoint must not close")]
            if mismatch
            else [HostileCleanup()]
        ),
    )

    with pytest.raises(KeyboardInterrupt, match="constructor cut"):
        agent.create_backend("UCX", {})

    candidate = _Endpoint.instances[-1]
    assert agent._endpoint is None
    if mismatch:
        assert candidate.close_calls == 0
        assert candidate.closed is False
        assert agent._endpoint_recovery is None
        assert any("non-matching" in note for note in error.__notes__)
        candidate.close_errors.clear()
        candidate.close()
    else:
        assert candidate.close_calls == 1
        assert candidate.closed is False
        assert agent._endpoint_recovery is candidate
        assert any(
            "cleanup also raised:" in note and "HostileCleanup" in note
            for note in error.__notes__
        )
        agent.create_backend("UCX", {})
        assert candidate.close_calls == 2
        assert candidate.closed is True
        assert agent._endpoint_recovery is None


def test_postpublication_constructor_cut_rolls_back_all_publications():
    agent = _make_uncreated_agent()
    armed = True

    def store_then_interrupt(self, name, value):
        nonlocal armed
        object.__setattr__(self, name, value)
        if self is agent and name == "_endpoint" and type(value) is _Endpoint and armed:
            armed = False
            value.close_errors.append(RuntimeError("close cut"))
            raise KeyboardInterrupt("after endpoint publication")

    with patch.object(TorchTransferAgent, "__setattr__", store_then_interrupt):
        with pytest.raises(KeyboardInterrupt, match="after endpoint publication"):
            agent.create_backend("UCX", {})

    failed_endpoint = _Endpoint.instances[-1]
    assert agent._endpoint is None
    assert agent._transport_backend is None
    assert agent._send_notification_sync is None
    assert agent._endpoint_recovery is failed_endpoint
    agent.create_backend("UCX", {})
    assert failed_endpoint.closed is True
    assert agent._endpoint is _Endpoint.instances[-1]
    assert agent._endpoint is not failed_endpoint


def test_capability_failure_rolls_back_endpoint_before_publication():
    agent = _make_uncreated_agent()
    _Endpoint.missing_capability = "write"

    with pytest.raises(NotImplementedError, match="write"):
        agent.create_backend("UCX", {})

    failed_endpoint = _Endpoint.instances[-1]
    assert failed_endpoint.closed is True
    assert agent._endpoint is None
    assert agent._transport_backend is None
    assert agent._send_notification_sync is None


def test_registration_and_peer_lost_returns_close_only_exact_orphans():
    agent, endpoint, _ = _make_agent()
    known_registration = agent.register_memory([(0x1000, 0x100, 0, "")], "VRAM")
    original_register = endpoint.register
    orphan_registration = None

    def register_then_interrupt(*args, **kwargs):
        nonlocal orphan_registration
        orphan_registration = original_register(*args, **kwargs)
        raise KeyboardInterrupt("registration return cut")

    with patch.object(endpoint, "register", side_effect=register_then_interrupt):
        with pytest.raises(KeyboardInterrupt, match="registration return cut"):
            agent.register_memory([(0x2000, 0x100, 0, "")], "VRAM")

    assert orphan_registration.closed is True
    assert known_registration[0].closed is False
    assert endpoint.open_registrations == known_registration
    assert agent._construction_dirty is False

    original_import = endpoint.import_peer
    orphan_peer = None

    def import_then_interrupt(*args, **kwargs):
        nonlocal orphan_peer
        orphan_peer = original_import(*args, **kwargs)
        raise KeyboardInterrupt("peer return cut")

    with patch.object(endpoint, "import_peer", side_effect=import_then_interrupt):
        with pytest.raises(KeyboardInterrupt, match="peer return cut"):
            agent.add_remote_agent(b"peer-metadata", peer_name="decode-agent")

    assert orphan_peer.closed is True
    assert endpoint.open_peers == ()
    assert agent._peers == {}
    assert agent._construction_dirty is False


@pytest.mark.parametrize("indexed", [True, False], ids=["cached", "one-off"])
def test_plan_lost_return_preserves_known_child_and_closes_exact_orphan(indexed):
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    known_local = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
    known_remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")
    agent.make_prepped_xfer("WRITE", known_local, [0], known_remote, [0])
    known_plan = endpoint.plans[-1]
    original_prepare = endpoint.prepare
    orphan_plan = None

    def prepare_then_interrupt(**kwargs):
        nonlocal orphan_plan
        orphan_plan = original_prepare(**kwargs)
        raise KeyboardInterrupt("plan return cut")

    with patch.object(endpoint, "prepare", side_effect=prepare_then_interrupt):
        with pytest.raises(KeyboardInterrupt, match="plan return cut"):
            if indexed:
                local = agent.prep_xfer_dlist("", [(0x1040, 32, 0)], "VRAM")
                remote = agent.prep_xfer_dlist(
                    "decode-agent", [(0x5040, 32, 0)], "VRAM"
                )
                agent.make_prepped_xfer("WRITE", local, [0], remote, [0])
            else:
                local = agent.get_xfer_descs([(0x1040, 32, 0)], "VRAM")
                remote = agent.get_xfer_descs([(0x5040, 32, 0)], "VRAM")
                agent.initialize_xfer("WRITE", local, remote, "decode-agent")

    assert orphan_plan.closed is True
    assert known_plan.closed is False
    assert endpoint.open_plans == (known_plan,)
    assert agent._construction_dirty is False


def test_dirty_plan_cleanup_cut_is_retried_before_next_construction():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    original_prepare = endpoint.prepare
    orphan_plan = None

    def prepare_then_interrupt(**kwargs):
        nonlocal orphan_plan
        orphan_plan = original_prepare(**kwargs)
        orphan_plan.close_errors.append(RuntimeError("first close cut"))
        raise KeyboardInterrupt("plan return cut")

    with patch.object(endpoint, "prepare", side_effect=prepare_then_interrupt):
        with pytest.raises(KeyboardInterrupt) as caught:
            agent.initialize_xfer("WRITE", local, remote, "decode-agent")

    assert orphan_plan.closed is False
    assert orphan_plan.close_calls == 1
    assert agent._construction_dirty is True
    assert any("cleanup also raised" in note for note in caught.value.__notes__)

    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")
    assert orphan_plan.closed is True
    assert orphan_plan.close_calls == 2
    assert handle.owned_plan is endpoint.plans[-1]
    assert agent._construction_dirty is False


def test_reconcile_scans_future_slot_and_binding_children_first():
    agent, endpoint, _ = _make_agent()
    close_order = []

    class OrderedResource(_Resource):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def close(self):
            close_order.append(self.name)
            super().close()

    slot = OrderedResource("slot")
    binding = OrderedResource("binding")
    plan = OrderedResource("plan")
    peer = OrderedResource("peer")
    registration = OrderedResource("registration")
    endpoint.transfer_slots.append(slot)
    endpoint.bound_transfers.append(binding)
    endpoint.plans.append(plan)
    endpoint.peers.append(peer)
    endpoint.registrations.append(registration)
    agent._construction_dirty = True

    agent.register_memory([(0x1000, 0x100, 0, "")], "VRAM")

    assert slot.closed is True
    assert binding.closed is True
    assert plan.closed is True
    assert peer.closed is True
    assert registration.closed is True
    assert close_order == ["slot", "binding", "plan", "peer", "registration"]
    assert agent._construction_dirty is False


def test_clean_one_off_and_cached_hit_never_read_construction_snapshots():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    endpoint.construction_snapshot_reads = 0
    endpoint.poison_construction_snapshots = True
    local_raw = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote_raw = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    agent.initialize_xfer("WRITE", local_raw, remote_raw, "decode-agent")
    local = agent.prep_xfer_dlist("", [(0x1040, 32, 0)], "VRAM")
    remote = agent.prep_xfer_dlist("decode-agent", [(0x5040, 32, 0)], "VRAM")
    agent.make_prepped_xfer("WRITE", local, [0], remote, [0])
    agent.make_prepped_xfer("WRITE", local, [0], remote, [0])

    assert endpoint.construction_snapshot_reads == 0


def test_peer_postpublication_cut_retries_without_duplicate_core_child():
    class StoreThenInterrupt(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise KeyboardInterrupt("peer map STORE cut")

    agent, endpoint, _ = _make_agent()
    agent._peers = StoreThenInterrupt()

    with pytest.raises(KeyboardInterrupt, match="peer map STORE cut"):
        agent.add_remote_agent(b"peer-metadata", peer_name="decode-agent")

    peer = endpoint.open_peers[0]
    assert agent._peers["decode-agent"] is peer
    assert agent._peer_wire_metadata == {"decode-agent": b"peer-metadata"}
    assert agent._source_aliases["remote-id"] == "decode-agent"
    assert agent._source_aliases[("remote-id", "remote-incarnation")] == "decode-agent"
    assert agent._construction_dirty is False
    assert (
        agent.add_remote_agent(b"peer-metadata", peer_name="decode-agent")
        == "decode-agent"
    )
    assert endpoint.open_peers == (peer,)


def test_cached_plan_postpublication_cut_retries_without_duplicate_core_child():
    class StoreThenInterrupt(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise KeyboardInterrupt("plan cache STORE cut")

    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    local = agent.prep_xfer_dlist("", [(0x1000, 32, 0)], "VRAM")
    remote = agent.prep_xfer_dlist("decode-agent", [(0x5000, 32, 0)], "VRAM")
    agent._plan_cache = StoreThenInterrupt()

    with pytest.raises(KeyboardInterrupt, match="plan cache STORE cut"):
        agent.make_prepped_xfer("WRITE", local, [0], remote, [0])

    plan = endpoint.open_plans[0]
    cached = next(iter(agent._plan_cache.values()))
    assert cached.plan is plan
    assert agent._construction_dirty is False
    handle = agent.make_prepped_xfer("WRITE", local, [0], remote, [0])
    assert handle.plan is plan
    assert endpoint.open_plans == (plan,)


def test_tracking_store_cut_cannot_hide_handle_from_worker_batch():
    class StoreThenInterrupt(dict):
        armed = True

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt("work store cut")

    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    agent._works = StoreThenInterrupt()
    handles = []
    agent.begin_handle_batch(handles)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")

    with pytest.raises(KeyboardInterrupt, match="work store cut"):
        agent.initialize_xfer("WRITE", local, remote, "decode-agent")

    assert handles == []
    assert agent._works == {}
    assert endpoint.plans[-1].closed is True
    agent.end_handle_batch(handles)


def test_begin_batch_store_cut_rolls_back_both_publications():
    class StoreThenInterruptLocal:
        def __init__(self):
            object.__setattr__(self, "armed", True)

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "value" and self.armed:
                object.__setattr__(self, "armed", False)
                raise KeyboardInterrupt("batch TLS store cut")

    agent, _, _ = _make_agent()
    handles = []
    agent._batch_handles = StoreThenInterruptLocal()

    with pytest.raises(KeyboardInterrupt, match="batch TLS store cut"):
        agent.begin_handle_batch(handles)

    assert agent._active_handle_batches == {}
    assert getattr(agent._batch_handles, "value", None) is None


def test_post_tracking_cleanup_cut_is_retained_and_reaped():
    agent, endpoint, _ = _make_agent()
    _register_and_add_peer(agent)
    handles = []
    agent.begin_handle_batch(handles)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    original_prepare = endpoint.prepare
    original_track = agent._track_work

    def prepare_with_one_close_cut(**kwargs):
        plan = original_prepare(**kwargs)
        plan.close_errors.append(RuntimeError("plan close cut"))
        return plan

    def track_then_interrupt(handle):
        original_track(handle)
        raise KeyboardInterrupt("tracking return cut")

    with (
        patch.object(endpoint, "prepare", side_effect=prepare_with_one_close_cut),
        patch.object(agent, "_track_work", side_effect=track_then_interrupt),
        patch.object(agent, "_start_retained_work_sweeper"),
    ):
        with pytest.raises(KeyboardInterrupt, match="tracking return cut"):
            agent.initialize_xfer("WRITE", local, remote, "decode-agent")

        handle = handles[0]
        assert agent._works[id(handle)] is handle
        assert id(handle) in agent._retained_work_ids
        assert handle.owned_plan.closed is False
        agent.end_handle_batch(handles)
        with agent._lock:
            agent._reap_retained_works()

    assert handle.phase == "released"
    assert handle.owned_plan.closed is True
    assert agent._retained_work_ids == set()


def test_whole_batch_retention_reaps_without_per_handle_journal():
    agent, _, _ = _make_agent()
    _register_and_add_peer(agent)
    handles = []
    agent.begin_handle_batch(handles)
    local = agent.get_xfer_descs([(0x1000, 32, 0)], "VRAM")
    remote = agent.get_xfer_descs([(0x5000, 32, 0)], "VRAM")
    handle = agent.initialize_xfer("WRITE", local, remote, "decode-agent")
    assert handles == [handle]
    assert agent.transfer(handle) == "PROC"
    handle.work.cancel_completes = False

    with patch.object(agent, "_start_retained_work_sweeper"):
        agent.retain_handle_batch(handles)
        agent.end_handle_batch(handles)
        assert agent._retained_handle_batches[id(handles)] is handles
        assert id(handle) not in agent._retained_work_ids
        handle.work.done = True
        handle.work.state = _WorkState.COMPLETED
        with agent._lock:
            agent._reap_retained_works()

    assert handle.phase == "released"
    assert agent._retained_handle_batches == {}


def test_retiring_peer_cleanup_retries_before_same_metadata_reimport():
    agent, endpoint, _ = _make_agent()
    agent.add_remote_agent(b"peer-metadata", peer_name="decode-agent")
    old_peer = endpoint.peers[-1]
    old_peer.close_errors.append(KeyboardInterrupt("peer close cut"))

    with pytest.raises(KeyboardInterrupt, match="peer close cut"):
        agent.rollback_remote_agent("decode-agent", b"peer-metadata")

    assert "decode-agent" in agent._retiring_peers
    assert (
        agent.add_remote_agent(b"peer-metadata", peer_name="decode-agent")
        == "decode-agent"
    )
    assert old_peer.closed is True
    assert endpoint.peers[-1] is not old_peer
    assert "decode-agent" not in agent._retiring_peers
