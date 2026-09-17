"""Experimental PyTorch endpoint-transfer adapter for SGLang NIXL PD."""

# Process-control exceptions are deliberately part of the ownership protocol:
# every BaseException boundary below reconciles or durably retains native state.
# ruff: noqa: BLE001

from __future__ import annotations

import importlib
import itertools
import logging
import operator
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

_UNKNOWN_NOTIFICATION_SOURCE = "<unknown-pytorch-transfer-source>"
_RETAINED_WORK_SWEEP_INTERVAL_S = 0.01
_PENDING_STATES = frozenset({"CREATED", "PENDING", "RUNNING", "SUBMITTED"})
_DONE_STATES = frozenset({"COMPLETE", "COMPLETED", "DONE", "SUCCEEDED"})
_FAILED_STATES = frozenset({"CANCELLED", "FAILED", "TIMED_OUT"})
_UNKNOWN_WORK_OUTCOME = object()
_FACTORY_API_VERSION = 2


def _load_transfer_api() -> Any:
    try:
        module = importlib.import_module("torch.distributed._transfer")
    except ImportError as exc:
        raise ImportError(
            "PyTorch endpoint transfer requires torch.distributed._transfer"
        ) from exc
    factory_version = getattr(module, "BACKEND_FACTORY_API_VERSION", None)
    if (
        type(factory_version) is not int
        or factory_version < _FACTORY_API_VERSION
        or not hasattr(module, "BackendFactoryV2")
        or not hasattr(module, "Endpoint")
        or not hasattr(module, "RawSpan")
        or not hasattr(module, "TransferOp")
        or not callable(
            getattr(
                getattr(module, "TransferPlan", None),
                "submit_indices_and_poll",
                None,
            )
        )
    ):
        raise ImportError(
            "PyTorch endpoint transfer requires FactoryV2 Core, Endpoint, RawSpan, "
            "TransferOp, and allocation-minimal exact-state indexed submission "
            "support"
        )
    return module


def _register_nixl_backend() -> None:
    """Load the external provider only after explicit SGLang opt-in."""
    try:
        module = importlib.import_module("nixl.torch_transfer")
    except ImportError as exc:
        raise ImportError(
            "The PyTorch NIXL transfer backend requires nixl.torch_transfer"
        ) from exc
    register = getattr(module, "register_torch_backend", None)
    factory_version = getattr(module, "TORCH_TRANSFER_FACTORY_API_VERSION", None)
    if type(factory_version) is not int or factory_version != _FACTORY_API_VERSION:
        raise ImportError(
            "nixl.torch_transfer must register the exact FactoryV2 provider ABI"
        )
    if not callable(register):
        raise ImportError(  # noqa: TRY004
            "nixl.torch_transfer does not expose register_torch_backend()"
        )
    register()


@dataclass(frozen=True)
class _RawDescriptor:
    address: int
    nbytes: int
    device_id: int
    stride: int | None = None
    count: int | None = None

    @property
    def extent(self) -> int:
        if self.stride is None or self.count is None:
            return self.nbytes
        return (self.count - 1) * self.stride + self.nbytes


@dataclass(frozen=True)
class _RawDescriptorList:
    descriptors: tuple[_RawDescriptor, ...]
    memory_type: str


@dataclass(frozen=True)
class _PreparedList:
    regions: tuple[object, ...]
    remote: bool
    peer_name: str | None = None


@dataclass(frozen=True)
class _CachedPlan:
    plan: object
    peer_name: str
    prepared_ids: frozenset[int]


@dataclass
class _RegistrationRecord:
    address: int
    nbytes: int
    device_id: int
    memory_type: str
    registration: object


def _freeze_index_vector(indices: Sequence[int]) -> Sequence[int]:
    """Take immutable ownership without copying SGLang's native-int32 fast path.

    SGLang creates fresh, owning NumPy int32 arrays immediately before
    ``make_prepped_xfer``.  Mark those arrays read-only and retain them until
    Core snapshots them during the adjacent ``transfer`` call.  Non-owning
    buffers are copied before freezing, and general Python sequences become an
    immutable normalized tuple.  Thus no caller-visible mutation can alter an
    admitted request while the production fast path adds no index memcpy.
    """

    try:
        view = memoryview(indices)
    except TypeError:
        return tuple(operator.index(value) for value in indices)
    try:
        native_int32 = (
            view.ndim == 1
            and view.c_contiguous
            and view.itemsize == 4
            and view.format in {"i", "@i", "=i"}
        )
        if not native_int32:
            return tuple(operator.index(value) for value in indices)
        if view.readonly:
            return indices
        flags = getattr(indices, "flags", None)
        setflags = getattr(indices, "setflags", None)
        if (
            flags is not None
            and bool(getattr(flags, "owndata", False))
            and callable(setflags)
        ):
            setflags(write=False)
            return indices
        copy = getattr(indices, "copy", None)
        if callable(copy):
            frozen = copy()
            freeze = getattr(frozen, "setflags", None)
            if callable(freeze):
                freeze(write=False)
                return frozen
        return tuple(operator.index(value) for value in indices)
    finally:
        view.release()


@dataclass(eq=False)
class _WorkHandle:
    work: object | None = None
    owned_plan: object | None = None
    terminal_state: str | None = None
    first_state: object | None = None
    operation: Literal["READ", "WRITE"] | None = None
    plan: object | None = None
    local_indices: Sequence[int] | None = None
    remote_indices: Sequence[int] | None = None
    notification: bytes | None = None
    peer_name: str | None = None
    cache_key: tuple[str, int, int] | None = None
    # The production token is the already allocated handle itself. This avoids
    # a second per-attempt allocation; Core Work.close breaks the temporary
    # Work-to-handle edge before the adapter drops its registry ownership.
    submission_interrupted: bool = False
    _state_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _status_in_progress: bool = field(default=False, init=False, repr=False)
    _recovery_in_progress: bool = field(default=False, init=False, repr=False)
    _release_in_progress: bool = field(default=False, init=False, repr=False)
    _native_submit_entered: bool = field(default=False, init=False, repr=False)
    phase: Literal[
        "prepared",
        "posting",
        "recovery_pending",
        "posted",
        "terminal",
        "released",
    ] = "prepared"


class TorchTransferAgent:
    """NIXL-agent-shaped boundary around ``torch.distributed._transfer``.

    It exists to validate the current SGLang data plane without spreading an
    experimental PyTorch API through ``conn.py``. Core ``MULTIPLE`` mode lets
    independent transfer-worker calls overlap while a short lifecycle gate
    drains every admitted call before close. Submission
    deliberately uses Core's ``caller_ready`` mode: the common transfer worker
    consumes the early-send CUDA event before staging, packing, or direct source
    access, and it waits for terminal Work before any chunk is reusable.
    """

    def __init__(
        self,
        name: str,
        *,
        owner: object,
        background_progress: bool,
        num_threads: int = 0,
        endpoint_backend: str = "nixl",
    ) -> None:
        self.name = name
        self._owner = owner
        self._background_progress = background_progress
        self._num_threads = num_threads
        self._endpoint_backend = endpoint_backend
        self._api = _load_transfer_api()
        self._core_transfer_operations = {
            "READ": self._api.TransferOp.READ,
            "WRITE": self._api.TransferOp.WRITE,
        }
        self._work_state_outcomes = self._complete_work_state_outcomes(self._api)
        self._lock = threading.RLock()
        self._lifecycle_cv = threading.Condition(self._lock)
        self._admission_local = threading.local()
        self._active_calls = 0
        self._closing = False
        self._close_owner: int | None = None
        self._endpoint: Any = None
        self._endpoint_recovery: Any = None
        self._transport_backend: str | None = None
        self._send_notification_sync: Callable[[object, bytes], None] | None = None
        self._registrations: list[_RegistrationRecord] = []
        self._registration_groups: list[tuple[object, ...]] = []
        self._registration_ids = itertools.count()
        self._peers: dict[str, object] = {}
        self._peer_wire_metadata: dict[str, bytes] = {}
        self._source_aliases: dict[object, str] = {}
        self._plan_cache: dict[tuple[str, int, int], _CachedPlan] = {}
        self._closing_plans: dict[tuple[str, int, int], _CachedPlan] = {}
        self._retiring_peers: set[str] = set()
        self._works: dict[int, _WorkHandle] = {}
        self._batch_handles = threading.local()
        self._active_handle_batches: dict[int, list[_WorkHandle]] = {}
        self._retained_handle_batches: dict[int, list[_WorkHandle]] = {}
        self._detached_works: list[_WorkHandle] = []
        self._retained_work_ids: set[int] = set()
        self._sweep_wakeup = threading.Event()
        self._sweep_thread: threading.Thread | None = None
        # Set before entering any Core child constructor and cleared only after
        # authoritative adapter publication or a complete identity reconcile.
        self._construction_dirty = False

    def create_backend(self, backend: str, backend_params: Mapping[str, str]) -> None:
        with self._lock:
            self._retry_endpoint_cleanup_unlocked()
            if self._endpoint is not None:
                raise RuntimeError("PyTorch transfer endpoint is already initialized")
            options = {
                "backends": [backend],
                "backend_init_params": {backend: dict(backend_params)},
                "transfer_backends": [backend],
                "num_threads": self._num_threads,
            }
            if self._endpoint_backend == "nixl":
                _register_nixl_backend()
            endpoint_id = str(uuid.uuid4())
            progress_mode = (
                self._api.ProgressMode.BACKGROUND
                if self._background_progress
                else self._api.ProgressMode.MANUAL
            )
            thread_mode = self._api.ThreadMode.MULTIPLE
            endpoint = None
            try:
                endpoint = self._api.Endpoint(
                    self.name,
                    backend=self._endpoint_backend,
                    endpoint_id=endpoint_id,
                    progress_mode=progress_mode,
                    # Core/provider synchronization owns native concurrency.
                    # Adapter locks protect only short Python state transitions.
                    thread_mode=thread_mode,
                    options=options,
                )
                required_capabilities = (
                    "raw_spans",
                    "strided_regions",
                    "write",
                    "attached_notifications",
                    "standalone_notifications",
                )
                missing = [
                    name
                    for name in required_capabilities
                    if not self._supports_capability(endpoint, name)
                ]
                if missing:
                    raise NotImplementedError(
                        "The selected PyTorch transfer backend is missing required "
                        f"capabilities: {', '.join(missing)}"
                    )
                send_notification_sync = getattr(
                    endpoint, "send_notification_sync", None
                )
                if not (
                    self._supports_capability(endpoint, "synchronous_notification_send")
                    and callable(send_notification_sync)
                ):
                    send_notification_sync = None
                # Derived publication precedes the authoritative Endpoint owner.
                self._send_notification_sync = send_notification_sync
                self._transport_backend = backend
                self._endpoint = endpoint
            except BaseException as error:
                # A process-control exception may land after the authoritative
                # assignment but before this method returns. The failed call
                # still owns rollback: never leave a usable Endpoint published.
                if endpoint is not None and self._endpoint is endpoint:
                    self._endpoint = None
                self._send_notification_sync = None
                self._transport_backend = None

                candidate = endpoint
                if candidate is None:
                    try:
                        recovery_endpoint = getattr(error, "recovery_endpoint", None)
                    except BaseException as lookup_error:
                        recovery_endpoint = None
                        self._add_exception_note(
                            error,
                            "reading recovery_endpoint also raised: "
                            f"{type(lookup_error).__qualname__}",
                        )
                    if recovery_endpoint is not None:
                        try:
                            trusted = self._matches_recovery_endpoint(
                                recovery_endpoint,
                                endpoint_id=endpoint_id,
                                progress_mode=progress_mode,
                                thread_mode=thread_mode,
                            )
                        except BaseException as match_error:
                            trusted = False
                            self._add_exception_note(
                                error,
                                "validating recovery_endpoint also raised: "
                                f"{type(match_error).__qualname__}",
                            )
                        if trusted:
                            candidate = recovery_endpoint
                        else:
                            # Exact identity fields are the ownership proof.
                            # Even another real Core Endpoint can belong to an
                            # unrelated live caller and must not be mutated.
                            self._add_exception_note(
                                error,
                                "ignored a non-matching recovery_endpoint during "
                                "SGLang endpoint construction",
                            )
                if candidate is not None:
                    self._endpoint_recovery = candidate
                try:
                    self._retry_endpoint_cleanup_unlocked()
                except BaseException as cleanup_error:
                    self._add_exception_note(
                        error,
                        "endpoint construction cleanup also raised: "
                        f"{type(cleanup_error).__qualname__}",
                    )
                raise

    def get_plugin_list(self) -> list[str]:
        with self._lock:
            return [self._transport_backend] if self._transport_backend else []

    def register_memory(
        self, descriptors: Sequence[Sequence[object]], memory_type: str
    ) -> tuple[object, ...]:
        with self._lock:
            self._reconcile_construction_state_unlocked()
            endpoint = self._require_endpoint()
            self._construction_dirty = True
            registrations = []
            records = []
            group = None
            try:
                for raw in self._normalize_descriptors(descriptors):
                    span = self._api.RawSpan(
                        address=raw.address,
                        nbytes=raw.extent,
                        device=self._device(memory_type, raw.device_id),
                        memory_type=memory_type,
                        owner=self._owner,
                    )
                    name = f"sglang_region_{next(self._registration_ids)}"
                    registration = endpoint.register(span, name=name)
                    registrations.append(registration)
                    records.append(
                        _RegistrationRecord(
                            address=raw.address,
                            nbytes=raw.extent,
                            device_id=raw.device_id,
                            memory_type=memory_type,
                            registration=registration,
                        )
                    )
                group = tuple(registrations)
                # The complete registration graph is authoritative only after
                # both indexes publish. Exception recovery removes either partial.
                self._registration_groups.append(group)
                self._registrations.extend(records)
                self._construction_dirty = False
                return group
            except BaseException as error:
                self._registrations = [
                    record
                    for record in self._registrations
                    if not any(record is candidate for candidate in records)
                ]
                if group is not None:
                    self._registration_groups = [
                        candidate
                        for candidate in self._registration_groups
                        if candidate is not group
                    ]
                self._reconcile_after_construction_error_unlocked(error, "registration")
                raise

    def deregister_memory(self, registrations: Sequence[object]) -> None:
        with self._lock:
            self._require_endpoint()
            self._reconcile_construction_state_unlocked()
            self._require_no_unsettled_interrupted_submission("deregister memory")
            for registration in reversed(tuple(registrations)):
                registration.close()
            removed = set(map(id, registrations))
            self._registrations = [
                record
                for record in self._registrations
                if id(record.registration) not in removed
            ]
            self._registration_groups = [
                remaining
                for group in self._registration_groups
                if (
                    remaining := tuple(
                        registration
                        for registration in group
                        if id(registration) not in removed
                    )
                )
            ]

    def get_agent_metadata(self) -> bytes:
        with self._lock:
            endpoint = self._require_endpoint()
            metadata = endpoint.export_metadata(
                [record.registration for record in self._registrations]
            )
            if isinstance(metadata, bytes):
                return metadata
            to_bytes = getattr(metadata, "to_bytes", None)
            if callable(to_bytes):
                return to_bytes()
            try:
                return bytes(metadata)
            except (TypeError, ValueError) as exc:
                raise TypeError("Endpoint metadata is not bytes-compatible") from exc

    def add_remote_agent(self, metadata: bytes, peer_name: str | None = None) -> str:
        with self._lock:
            self._reconcile_construction_state_unlocked()
            metadata = bytes(metadata)
            if peer_name is not None and peer_name in self._retiring_peers:
                self.remove_remote_agent(peer_name)
            if peer_name is not None and peer_name in self._peers:
                if self._peer_wire_metadata.get(peer_name) == metadata:
                    return peer_name
                raise ValueError(
                    f"Peer {peer_name!r} is already registered with different metadata"
                )
            import_options = {} if peer_name is None else {"expected_name": peer_name}
            self._construction_dirty = True
            peer = None
            alias = None
            aliases = None
            previous_aliases = self._source_aliases
            try:
                peer = self._require_endpoint().import_peer(metadata, **import_options)
                source_ids = self._peer_source_ids(peer)
                source_id = source_ids[0]
                alias = peer_name or source_id
                if alias in self._peers:
                    peer.close()
                    raise ValueError(f"Peer {alias!r} is already registered")
                if not callable(getattr(peer, "unsafe_region", None)):
                    peer.close()
                    raise NotImplementedError(
                        "Imported peer does not support provider-validated raw regions"
                    )
                aliases = dict(previous_aliases)
                for source in source_ids:
                    aliases[source] = alias
                incarnation = getattr(peer, "incarnation", None)
                if incarnation is not None:
                    aliases[(source_id, str(incarnation))] = alias
                self._source_aliases = aliases
                self._peer_wire_metadata[alias] = metadata
                # The exact Core peer is the authoritative adapter owner and
                # commits last, after every derived identity index.
                self._peers[alias] = peer
                self._construction_dirty = False
                return alias
            except BaseException as error:
                if alias is None or self._peers.get(alias) is not peer:
                    if aliases is not None and self._source_aliases is aliases:
                        self._source_aliases = previous_aliases
                    if alias is not None:
                        self._peer_wire_metadata.pop(alias, None)
                    self._reconcile_after_construction_error_unlocked(error, "peer")
                else:
                    # A dict STORE may commit and then raise. Exact map identity
                    # proves adoption, so retry is idempotent and no scan is due.
                    self._construction_dirty = False
                raise

    def rollback_remote_agent(self, peer_name: str, metadata: bytes) -> bool:
        """Remove only the peer carrying this exact bootstrap wire metadata."""

        with self._lock:
            self._reconcile_construction_state_unlocked()
            if self._peer_wire_metadata.get(peer_name) != bytes(metadata):
                return False
            self.remove_remote_agent(peer_name)
            return True

    def remove_remote_agent(self, peer_name: str) -> None:
        with self._lock:
            self._require_endpoint()
            self._reconcile_construction_state_unlocked()
            peer = self._peers.get(peer_name)
            if peer is None:
                self._retiring_peers.discard(peer_name)
                return
            # Quarantine before the first provider close. If close partially
            # commits and raises, no future submission may reuse this peer.
            self._retiring_peers.add(peer_name)
            self._require_no_unsettled_interrupted_submission("remove a remote agent")
            self._release_cached_plans(lambda cached: cached.peer_name == peer_name)
            if not getattr(peer, "closed", False):
                peer.close()
            self._source_aliases = {
                source: alias
                for source, alias in self._source_aliases.items()
                if alias != peer_name
            }
            self._peer_wire_metadata.pop(peer_name, None)
            self._peers.pop(peer_name, None)
            self._retiring_peers.discard(peer_name)

    def get_xfer_descs(
        self, descriptors: Sequence[Sequence[object]], memory_type: str
    ) -> _RawDescriptorList:
        return _RawDescriptorList(self._normalize_descriptors(descriptors), memory_type)

    def prep_xfer_dlist(
        self,
        peer_name: str,
        descriptors: Sequence[Sequence[object]],
        memory_type: str,
    ) -> _PreparedList:
        with self._lock:
            self._require_endpoint()
            raw = self._normalize_descriptors(descriptors)
            if peer_name:
                regions = self._resolve_remote(peer_name, raw, memory_type)
                return _PreparedList(regions, remote=True, peer_name=peer_name)
            return _PreparedList(self._resolve_local(raw, memory_type), remote=False)

    def release_dlist_handle(self, handle: _PreparedList) -> None:
        with self._lock:
            self._require_endpoint()
            prepared_id = id(handle)
            self._release_cached_plans(
                lambda cached: prepared_id in cached.prepared_ids
            )

    def make_prepped_xfer(
        self,
        operation: str,
        source: _PreparedList,
        source_indices: Sequence[int],
        destination: _PreparedList,
        destination_indices: Sequence[int],
        notification: bytes = b"",
    ) -> _WorkHandle:
        with self._admitted_call():
            operation = self._normalize_operation(operation)
            local, remote, local_indices, remote_indices = self._orient_prepared(
                operation,
                source,
                source_indices,
                destination,
                destination_indices,
            )
            local_indices = _freeze_index_vector(local_indices)
            remote_indices = _freeze_index_vector(remote_indices)
            key = (operation, id(local), id(remote))
            wire_notification = bytes(notification) if notification else None
            handle = None
            with self._lock:
                if remote.peer_name in self._retiring_peers:
                    raise RuntimeError(
                        f"Peer {remote.peer_name!r} teardown is in progress"
                    )
                if key in self._closing_plans:
                    raise RuntimeError("Prepared transfer plan teardown is in progress")
                cached = self._plan_cache.get(key)
                if cached is None:
                    assert remote.peer_name is not None
                    self._reconcile_construction_state_unlocked()
                    self._construction_dirty = True
                    cached = None
                    try:
                        plan = self._require_endpoint().prepare(
                            local=local.regions,
                            remote=remote.regions,
                            indexed=True,
                        )
                        cached = _CachedPlan(
                            plan=plan,
                            peer_name=remote.peer_name,
                            prepared_ids=frozenset({id(local), id(remote)}),
                        )
                        self._plan_cache[key] = cached
                        self._construction_dirty = False
                    except BaseException as error:
                        if cached is None or self._plan_cache.get(key) is not cached:
                            self._reconcile_after_construction_error_unlocked(
                                error, "indexed plan"
                            )
                        else:
                            # Exact cache identity proves the STORE committed
                            # even when its return boundary was interrupted.
                            self._construction_dirty = False
                        raise
                # Publish the plan and immutable index ownership in one state
                # transition. Core snapshots and validates the indices during
                # its allocation-minimal direct one-shot submit.
                handle = _WorkHandle(
                    operation=operation,
                    plan=cached.plan,
                    local_indices=local_indices,
                    remote_indices=remote_indices,
                    notification=wire_notification,
                    peer_name=remote.peer_name,
                    cache_key=key,
                )
                return self._track_work(handle)

    def initialize_xfer(
        self,
        operation: str,
        source: _RawDescriptorList,
        destination: _RawDescriptorList,
        peer_name: str,
        notification: bytes = b"",
    ) -> _WorkHandle:
        with self._lock:
            self._require_endpoint()
            operation = self._normalize_operation(operation)
            if peer_name in self._retiring_peers:
                raise RuntimeError(f"Peer {peer_name!r} teardown is in progress")
            if operation == "WRITE":
                local_regions = self._resolve_local(
                    source.descriptors, source.memory_type
                )
                remote_regions = self._resolve_remote(
                    peer_name, destination.descriptors, destination.memory_type
                )
            else:
                local_regions = self._resolve_local(
                    destination.descriptors, destination.memory_type
                )
                remote_regions = self._resolve_remote(
                    peer_name, source.descriptors, source.memory_type
                )
            self._reconcile_construction_state_unlocked()
            self._construction_dirty = True
            plan = None
            handle = None
            try:
                plan = self._require_endpoint().prepare(
                    local=local_regions,
                    remote=remote_regions,
                    indexed=False,
                )
                handle = _WorkHandle(
                    owned_plan=plan,
                    operation=operation,
                    plan=plan,
                    notification=bytes(notification) if notification else None,
                    peer_name=peer_name,
                )
                tracked = self._track_work(handle)
                self._construction_dirty = False
                return tracked
            except BaseException as error:
                if handle is not None and self._works.get(id(handle)) is handle:
                    try:
                        self._release_work(handle)
                    except BaseException as cleanup_error:
                        self._retained_work_ids.add(id(handle))
                        self._start_retained_work_sweeper()
                        self._add_exception_note(
                            error,
                            "one-off handle cleanup also raised: "
                            f"{type(cleanup_error).__qualname__}",
                        )
                self._reconcile_after_construction_error_unlocked(error, "one-off plan")
                raise

    def transfer(self, handle: _WorkHandle) -> str:
        with self._admitted_call() as endpoint:
            should_post = False
            with handle._state_lock:
                if handle.terminal_state is not None:
                    return handle.terminal_state
                if handle.phase == "released":
                    raise RuntimeError("PyTorch transfer handle is released")
                if handle.phase == "posting":
                    raise RuntimeError(
                        "PyTorch transfer submission is already in progress"
                    )
                if handle.phase == "prepared":
                    # Publish submission ownership before native entry. The lock
                    # is released before submit_and_poll/write/read.
                    handle.phase = "posting"
                    should_post = True
                recovery_pending = handle.phase == "recovery_pending"
            if recovery_pending:
                self._retry_interrupted_post_recovery(handle, schedule=True)
                with handle._state_lock:
                    recovery_pending = handle.phase == "recovery_pending"
                if recovery_pending:
                    return "PROC"
            if should_post:
                try:
                    self._post(handle, endpoint)
                except BaseException as error:
                    # Recover provider ownership before any diagnostic formatting:
                    # an exception may itself have hostile __str__/repr hooks.
                    with handle._state_lock:
                        native_submit_entered = handle._native_submit_entered
                    if native_submit_entered:
                        self._handle_interrupted_submission(handle, error)
                    else:
                        # Validation/teardown rejection precedes provider entry,
                        # so there is no hidden Work to reconcile.
                        with handle._state_lock:
                            handle.terminal_state = "ERR"
                            handle.phase = "terminal"
                        self._release_work(handle)
                    with self._lock:
                        retained = id(handle) in self._works
                    with handle._state_lock:
                        retained_phase = handle.phase
                    if retained and retained_phase in (
                        "posted",
                        "recovery_pending",
                    ):
                        # Preserve immutable diagnostics only. Retaining the
                        # exception would retain this stack through a long DMA.
                        try:
                            details = (type(error).__qualname__, str(error))
                        except BaseException:
                            details = (
                                "BaseException",
                                "diagnostic formatting unavailable",
                            )
                        try:
                            handle.__dict__["_submission_error"] = details
                        except BaseException:  # noqa: S110
                            pass
                        # Return the already tracked handle to conn.py's source
                        # lifetime barrier. The error is surfaced after terminality.
                        return "PROC"
                    raise
            state = self._work_state(handle, release=False)
            if state == "ERR":
                # Current conn.py callers raise immediately on a post failure
                # and never issue the later release call. Terminal failure is
                # already authoritative, so release it here; a failed cleanup
                # remains tracked and is retried by the retained-work sweeper.
                try:
                    self._release_work(handle)
                except Exception as exc:
                    logger.warning("PyTorch failed Work cleanup failed: %s", exc)
                    with self._lock:
                        self._retained_work_ids.add(id(handle))
                    self._start_retained_work_sweeper()
            return state

    def check_xfer_state(self, handle: _WorkHandle) -> str:
        with self._admitted_call():
            return self._work_state(handle, release=True)

    def begin_handle_batch(self, handles: list[_WorkHandle]) -> None:
        """Publish a worker-owned sink before that worker can create a Work."""
        with self._lock:
            self._require_endpoint()
            if getattr(self._batch_handles, "value", None) is not None:
                raise RuntimeError("PyTorch transfer handle batch is already active")
            batch_id = id(handles)
            existing = self._active_handle_batches.get(batch_id)
            if existing is not None and existing is not handles:
                raise RuntimeError("PyTorch transfer handle batch identity collision")
            # Publish the list itself before any Core call. Every later handle
            # append is therefore already reachable if its worker is interrupted.
            try:
                self._active_handle_batches[batch_id] = handles
                self._batch_handles.value = handles
            except BaseException:
                # conn.py has not observed a successful begin and therefore
                # cannot own cleanup yet. Roll back either store even when the
                # store itself committed before raising.
                if getattr(self._batch_handles, "value", None) is handles:
                    del self._batch_handles.value
                if self._active_handle_batches.get(batch_id) is handles:
                    self._active_handle_batches.pop(batch_id, None)
                raise

    def end_handle_batch(self, handles: list[_WorkHandle]) -> None:
        with self._lock:
            self._require_endpoint()
            current = getattr(self._batch_handles, "value", None)
            if current is not handles:
                raise RuntimeError("PyTorch transfer handle batch ownership mismatch")
            del self._batch_handles.value
            self._active_handle_batches.pop(id(handles), None)
            self._prune_retained_handle_batches_unlocked()
            if id(handles) in self._retained_handle_batches:
                self._start_retained_work_sweeper()

    def retain_handle_batch(self, handles: list[_WorkHandle]) -> None:
        """Durably retain a whole worker batch with one identity publication."""

        with self._lock:
            self._require_endpoint()
            active = self._active_handle_batches.get(id(handles))
            if active is not None and active is not handles:
                raise RuntimeError("PyTorch transfer handle batch identity collision")
            self._retained_handle_batches[id(handles)] = handles
            self._prune_retained_handle_batches_unlocked()
            if id(handles) in self._retained_handle_batches:
                self._start_retained_work_sweeper()

    def handle_batch_is_quiescent(self, handles: Sequence[_WorkHandle]) -> bool:
        with self._lock:
            return not any(self._works.get(id(handle)) is handle for handle in handles)

    @staticmethod
    def pop_xfer_error(handle: _WorkHandle) -> RuntimeError | None:
        """Return deferred lost-return diagnostics after terminal settlement."""

        details = handle.__dict__.pop("_submission_error", None)
        if details is None:
            return None
        error_type, message = details
        return RuntimeError(
            "PyTorch transfer submission return was interrupted by "
            f"{error_type}: {message}"
        )

    def progress(self) -> None:
        """Advance the provider and reap detached notification requests.

        SGLang calls this before checking one batch of transfer handles. Regular
        Works are intentionally not scanned here: the following per-handle loop
        is their single status pass. Detached notification Works have no caller
        to check them, so this method still owns their terminal cleanup.
        """

        # Optimistic empty fast path: a concurrent sender publishes the Work in
        # ``_works`` before appending here, so missing that append only defers
        # detached cleanup to the next poll/close and cannot lose ownership.
        if self._background_progress and not self._detached_works:
            return
        with self._admitted_call():
            self._advance_provider()
            self._reap_detached_works()

    def cancel_handles(self, handles: Sequence[_WorkHandle]) -> None:
        """Best-effort cancel siblings while retaining non-terminal Works."""

        with self._admitted_call():
            for handle in handles:
                with self._lock:
                    tracked = id(handle) in self._works
                if not tracked:
                    continue
                with handle._state_lock:
                    phase = handle.phase
                    if phase == "prepared":
                        handle.terminal_state = "ERR"
                        handle.phase = "terminal"
                if phase == "prepared":
                    try:
                        self._release_work(handle)
                    except Exception as exc:
                        logger.warning(
                            "PyTorch prepared transfer cleanup failed: %s", exc
                        )
                        with self._lock:
                            self._retained_work_ids.add(id(handle))
                        self._start_retained_work_sweeper()
                    continue
                if phase == "posting":
                    with self._lock:
                        self._retained_work_ids.add(id(handle))
                    continue
                if phase == "recovery_pending":
                    self._retry_interrupted_post_recovery(handle, schedule=True)
                    with handle._state_lock:
                        recovery_pending = handle.phase == "recovery_pending"
                    if recovery_pending:
                        continue
                if self._work_state(handle, release=False) in ("DONE", "ERR"):
                    continue
                try:
                    with handle._state_lock:
                        work = handle.work
                    assert work is not None
                    work.cancel()
                except Exception as exc:
                    logger.warning("PyTorch transfer cancellation failed: %s", exc)
            self._progress()
            with self._lock:
                tracked_handles = tuple(
                    handle
                    for handle in handles
                    if self._works.get(id(handle)) is handle
                )
            retained = set()
            for handle in tracked_handles:
                with handle._state_lock:
                    recovery_pending = handle.phase == "recovery_pending"
                if recovery_pending or self._work_state(handle, release=False) not in (
                    "DONE",
                    "ERR",
                ):
                    retained.add(id(handle))
            if retained:
                with self._lock:
                    self._retained_work_ids.update(retained)
                self._start_retained_work_sweeper()

    def cancel_handles_and_wait(self, handles: Sequence[_WorkHandle]) -> None:
        """Cancel a failed batch and return only when every Work is reusable.

        This deliberately unbounded routine is used only on failure. Publishing
        a room failure while an uncancellable Work is active would let SGLang
        recycle KV pages that DMA may still be reading.
        """

        if isinstance(handles, list):
            self.retain_handle_batch(handles)
        handles = tuple(handles)
        with self._admitted_call():
            self.cancel_handles(handles)
            while True:
                pending = False
                self._advance_provider()
                for handle in handles:
                    with self._lock:
                        tracked = self._works.get(id(handle)) is handle
                    if not tracked:
                        continue
                    with handle._state_lock:
                        phase = handle.phase
                    if phase == "recovery_pending":
                        self._retry_interrupted_post_recovery(
                            handle,
                            schedule=False,
                        )
                        with handle._state_lock:
                            phase = handle.phase
                    if phase in ("posting", "recovery_pending"):
                        pending = True
                        continue
                    if phase == "prepared":
                        with handle._state_lock:
                            if handle.phase == "prepared":
                                handle.terminal_state = "ERR"
                                handle.phase = "terminal"
                    if self._work_state(handle, release=False) not in (
                        "DONE",
                        "ERR",
                    ):
                        pending = True
                        continue
                    try:
                        self._release_work(handle)
                    except Exception as exc:
                        logger.warning(
                            "PyTorch failed-batch terminal cleanup failed: %s",
                            exc,
                        )
                        pending = True
                if not pending:
                    with self._lock:
                        self._prune_retained_handle_batches_unlocked()
                    return
                time.sleep(0)

    def release_xfer_handle(self, handle: _WorkHandle) -> None:
        with self._admitted_call():
            with self._lock:
                tracked = self._works.get(id(handle)) is handle
            if not tracked:
                return
            with handle._state_lock:
                phase = handle.phase
            if phase == "prepared":
                self._release_work(handle)
                return
            if phase == "posting":
                raise RuntimeError("Cannot release a posting PyTorch transfer")
            if phase == "recovery_pending":
                self._retry_interrupted_post_recovery(handle, schedule=True)
                with handle._state_lock:
                    recovery_pending = handle.phase == "recovery_pending"
                if recovery_pending:
                    raise RuntimeError(
                        "Cannot release a PyTorch transfer while Work recovery is pending"
                    )
            if self._work_state(handle, release=False) not in ("DONE", "ERR"):
                with handle._state_lock:
                    work = handle.work
                assert work is not None
                work.cancel()
                self._progress()
            if self._work_state(handle, release=False) not in ("DONE", "ERR"):
                raise RuntimeError("Cannot release an active PyTorch transfer")
            self._release_work(handle)

    def send_notif(self, peer_name: str, payload: bytes) -> _WorkHandle | None:
        with self._admitted_call() as endpoint:
            with self._lock:
                peer = self._require_peer(peer_name)
                send_notification_sync = self._send_notification_sync
            if type(payload) is not bytes:
                payload = bytes(payload)
            if send_notification_sync is not None:
                send_notification_sync(peer, payload)
                return None
            handle = _WorkHandle(phase="posting")
            try:
                # Tracking and provider entry are one recovery transaction.
                self._track_work(handle)
                work = endpoint.send_notification(
                    peer,
                    payload,
                    recovery_token=handle,
                )
                with handle._state_lock:
                    handle.work = work
                    handle.phase = "posted"
                with self._lock:
                    self._detached_works.append(handle)
                return handle
            except BaseException as error:
                self._handle_interrupted_submission(handle, error)
                with self._lock:
                    tracked = self._works.get(id(handle)) is handle
                if tracked:
                    self._ensure_detached(handle)
                raise

    def _notification_alias(
        self,
        source_endpoint_id: str | None,
        source_incarnation: str | None,
        source_aliases: Mapping[object, str] | None = None,
    ) -> str:
        if source_endpoint_id is None:
            if source_incarnation is not None:
                raise RuntimeError("notification source identity is incomplete")
            # One-way WRITE topologies need not import the sender. Notification
            # payloads remain self-routing in SGLang's current protocol.
            return _UNKNOWN_NOTIFICATION_SOURCE
        if source_incarnation is None:
            raise RuntimeError(
                "notification source is missing its endpoint incarnation"
            )
        source_key = (str(source_endpoint_id), str(source_incarnation))
        if source_aliases is None:
            with self._lock:
                source_aliases = self._source_aliases
        try:
            return source_aliases[source_key]
        except KeyError as exc:
            raise RuntimeError(
                "notification arrived from an unknown or stale endpoint "
                f"incarnation {source_key!r}"
            ) from exc

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        with self._admitted_call() as endpoint:
            poll_batches = getattr(endpoint, "poll_notification_batches", None)
            if callable(poll_batches):
                notifications = poll_batches()
                grouped_receive = True
            else:
                notifications = endpoint.poll_notifications()
                grouped_receive = False
            # Either polling operation already drives NIXL notification progress.
            # Reap detached notification Works without issuing a second
            # provider progress call or scanning caller-owned transfers.
            self._reap_detached_works()
            # Peer mutation publishes a replacement alias map under ``_lock``.
            # Snapshot only after native polling, then route the whole batch
            # without a lock per notification.
            with self._lock:
                source_aliases = self._source_aliases
            grouped: dict[str, list[bytes]] = defaultdict(list)
            if grouped_receive:
                for batch in notifications:
                    alias = self._notification_alias(
                        batch.source_endpoint_id,
                        batch.source_incarnation,
                        source_aliases,
                    )
                    grouped[alias].extend(batch.payloads)
                return dict(grouped)
            if isinstance(notifications, Mapping):
                for source, payloads in notifications.items():
                    if source is None:
                        alias = _UNKNOWN_NOTIFICATION_SOURCE
                    else:
                        source = str(source)
                        alias = source_aliases.get(source, source)
                    grouped[alias].extend(bytes(payload) for payload in payloads)
                return dict(grouped)
            for notification in notifications:
                alias = self._notification_alias(
                    notification.source_endpoint_id,
                    getattr(notification, "source_incarnation", None),
                    source_aliases,
                )
                grouped[alias].append(bytes(notification.payload))
            return dict(grouped)

    def close(self) -> None:
        endpoint = self._begin_close()
        if endpoint is None:
            return
        try:
            # Endpoint.closed is the authoritative commit receipt. Inspect it
            # before querying children because a previous close may have
            # committed even though its Python return was lost.
            if self._endpoint_is_closed(endpoint):
                self._finalize_closed_endpoint_unlocked(endpoint)
                return
            self._close_open_endpoint_exclusive(endpoint)
        finally:
            self._end_close()

    def _close_open_endpoint_exclusive(self, endpoint: object) -> None:
        """Close one live endpoint while the lifecycle gate is exclusive."""

        try:
            self._retry_endpoint_cleanup_unlocked()
            self._reconcile_construction_state_unlocked(force=True)
            for handle in tuple(self._works.values()):
                if handle.phase == "recovery_pending":
                    self._retry_interrupted_post_recovery(handle, schedule=False)
            for handle in tuple(self._works.values()):
                if handle.phase == "prepared":
                    self._release_work(handle)
            for handle in tuple(self._works.values()):
                if handle.phase == "posted" and self._work_state(
                    handle, release=False
                ) not in ("DONE", "ERR"):
                    assert handle.work is not None
                    handle.work.cancel()
            self._progress()
            active = [
                handle
                for handle in self._works.values()
                if handle.phase in ("posting", "recovery_pending")
                or (
                    handle.phase == "posted"
                    and self._work_state(handle, release=False) not in ("DONE", "ERR")
                )
            ]
            if active:
                raise RuntimeError(
                    f"Cannot close endpoint with {len(active)} active transfer(s)"
                )
            for handle in tuple(self._works.values()):
                self._release_work(handle)
            self._detached_works.clear()
            self._retained_work_ids.clear()
            self._active_handle_batches.clear()
            self._retained_handle_batches.clear()
            self._retiring_peers.update(self._peers)
            self._release_cached_plans(lambda cached: True)
            for peer in reversed(tuple(self._peers.values())):
                peer.close()
            self._peers.clear()
            self._peer_wire_metadata.clear()
            self._retiring_peers.clear()
            for record in reversed(self._registrations):
                record.registration.close()
            self._registrations.clear()
            self._registration_groups.clear()
            endpoint.close()
        except BaseException:
            if self._endpoint_is_closed(endpoint):
                self._finalize_closed_endpoint_unlocked(endpoint)
            raise
        self._finalize_closed_endpoint_unlocked(endpoint)

    @contextmanager
    def _admitted_call(self):
        """Admit native work without holding a Python lock across the call."""

        depth = getattr(self._admission_local, "depth", 0)
        if depth:
            self._admission_local.depth = depth + 1
            try:
                yield self._require_endpoint()
            finally:
                self._admission_local.depth = depth
            return
        thread_id = threading.get_ident()
        with self._lifecycle_cv:
            if self._closing and self._close_owner != thread_id:
                raise RuntimeError("PyTorch transfer endpoint close is in progress")
            endpoint = self._require_endpoint()
            self._active_calls += 1
            self._admission_local.depth = 1
        try:
            yield endpoint
        finally:
            with self._lifecycle_cv:
                self._admission_local.depth = 0
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._lifecycle_cv.notify_all()

    def _begin_close(self) -> object | None:
        thread_id = threading.get_ident()
        with self._lifecycle_cv:
            while self._closing and self._close_owner != thread_id:
                self._lifecycle_cv.wait()
            if self._close_owner == thread_id:
                raise RuntimeError("recursive PyTorch transfer endpoint close")
            if self._endpoint is None:
                return None
            self._closing = True
            self._close_owner = thread_id
            while self._active_calls:
                self._lifecycle_cv.wait()
            return self._endpoint

    def _end_close(self) -> None:
        with self._lifecycle_cv:
            self._closing = False
            self._close_owner = None
            self._lifecycle_cv.notify_all()

    @staticmethod
    def _endpoint_is_closed(endpoint: object) -> bool:
        try:
            return getattr(endpoint, "closed", False) is True
        except BaseException:
            return False

    def _finalize_closed_endpoint_unlocked(self, endpoint: object) -> None:
        """Adopt authoritative endpoint close without touching Core children."""

        if self._endpoint is not endpoint:
            return
        for handle in self._works.values():
            if handle.terminal_state is None:
                handle.terminal_state = "ERR"
            handle.phase = "released"
            handle.local_indices = None
            handle.remote_indices = None
            handle.notification = None
            handle.plan = None
        self._works.clear()
        self._detached_works.clear()
        self._retained_work_ids.clear()
        self._active_handle_batches.clear()
        self._retained_handle_batches.clear()
        self._plan_cache.clear()
        self._closing_plans.clear()
        self._retiring_peers.clear()
        self._peers.clear()
        self._peer_wire_metadata.clear()
        self._source_aliases.clear()
        self._registrations.clear()
        self._registration_groups.clear()
        self._send_notification_sync = None
        self._endpoint = None
        self._construction_dirty = False
        self._sweep_wakeup.set()

    @staticmethod
    def _supports_capability(endpoint: object, name: str) -> bool:
        capabilities = getattr(endpoint, "capabilities", None)
        if callable(capabilities):
            capabilities = capabilities()
        if isinstance(capabilities, Mapping):
            return bool(capabilities.get(name, False))
        return bool(getattr(capabilities, name, False))

    @staticmethod
    def _normalize_descriptors(
        descriptors: Sequence[Sequence[object]],
    ) -> tuple[_RawDescriptor, ...]:
        normalized = []
        for descriptor in descriptors:
            if len(descriptor) < 3:
                raise ValueError("Transfer descriptors require address, size, device")
            address, nbytes, device_id = map(int, descriptor[:3])
            if address <= 0 or nbytes <= 0 or device_id < 0:
                raise ValueError(
                    "Transfer descriptor address/size must be positive and device "
                    "must be non-negative"
                )
            stride: int | None = None
            count: int | None = None
            if len(descriptor) >= 5:
                stride, count = map(int, descriptor[3:5])
                if stride < nbytes or count <= 0:
                    raise ValueError(
                        "Strided descriptors require stride >= size and count > 0"
                    )
                if count == 1:
                    stride = count = None
            elif len(descriptor) == 4 and not isinstance(descriptor[3], str):
                raise ValueError("Strided descriptors require both stride and count")
            normalized.append(_RawDescriptor(address, nbytes, device_id, stride, count))
        return tuple(normalized)

    @staticmethod
    def _device(memory_type: str, device_id: int) -> str:
        if memory_type == "DRAM":
            return "cpu"
        if memory_type == "VRAM":
            return f"cuda:{device_id}"
        raise NotImplementedError(
            f"PyTorch transfer adapter does not support {memory_type!r} memory"
        )

    def _resolve_local(
        self, descriptors: Sequence[_RawDescriptor], memory_type: str
    ) -> tuple[object, ...]:
        regions = []
        for descriptor in descriptors:
            end = descriptor.address + descriptor.extent
            for record in self._registrations:
                if (
                    record.memory_type == memory_type
                    and record.device_id == descriptor.device_id
                    and descriptor.address >= record.address
                    and end <= record.address + record.nbytes
                ):
                    kwargs = (
                        {}
                        if descriptor.count is None
                        else {
                            "stride": descriptor.stride,
                            "count": descriptor.count,
                        }
                    )
                    regions.append(
                        record.registration.region(
                            descriptor.address - record.address,
                            descriptor.nbytes,
                            **kwargs,
                        )
                    )
                    break
            else:
                raise ValueError(
                    f"Local raw span 0x{descriptor.address:x}+{descriptor.nbytes} "
                    "is outside registered memory"
                )
        return tuple(regions)

    def _resolve_remote(
        self,
        peer_name: str,
        descriptors: Sequence[_RawDescriptor],
        memory_type: str,
    ) -> tuple[object, ...]:
        peer = self._require_peer(peer_name)
        return tuple(
            peer.unsafe_region(
                descriptor.address,
                descriptor.nbytes,
                device=self._device(memory_type, descriptor.device_id),
                memory_type=memory_type,
                **(
                    {}
                    if descriptor.count is None
                    else {
                        "stride": descriptor.stride,
                        "count": descriptor.count,
                    }
                ),
            )
            for descriptor in descriptors
        )

    def _orient_prepared(
        self,
        operation: str,
        source: _PreparedList,
        source_indices: Sequence[int],
        destination: _PreparedList,
        destination_indices: Sequence[int],
    ) -> tuple[_PreparedList, _PreparedList, Sequence[int], Sequence[int]]:
        if operation == "WRITE":
            local, remote = source, destination
            local_indices, remote_indices = source_indices, destination_indices
        else:
            local, remote = destination, source
            local_indices, remote_indices = destination_indices, source_indices
        if local.remote or not remote.remote:
            raise ValueError(
                f"Invalid {operation} prepared-list local/remote orientation"
            )
        return local, remote, local_indices, remote_indices

    def _post(self, handle: _WorkHandle, endpoint: object) -> None:
        if handle.operation is None or handle.plan is None:
            raise RuntimeError("PyTorch transfer handle has no prepared operation")
        with self._lock:
            if handle.peer_name in self._retiring_peers:
                raise RuntimeError(f"Peer {handle.peer_name!r} teardown is in progress")
            if handle.peer_name is not None:
                self._require_peer(handle.peer_name)
            if handle.cache_key is not None and handle.cache_key in self._closing_plans:
                raise RuntimeError("Prepared transfer plan teardown is in progress")
        with handle._state_lock:
            handle._native_submit_entered = True
        first_state = None
        if handle.local_indices is not None or handle.remote_indices is not None:
            if handle.local_indices is None or handle.remote_indices is None:
                raise RuntimeError("PyTorch indexed transfer has incomplete indices")
            kwargs = {
                "local_indices": handle.local_indices,
                "remote_indices": handle.remote_indices,
                "max_polls": 0,
                "timeout_ns": None,
                "recovery_token": handle,
            }
            if handle.notification is None:
                work, first_state = handle.plan.submit_indices_and_poll(
                    self._core_transfer_operations[handle.operation],
                    **kwargs,
                )
            else:
                work, first_state = handle.plan.submit_indices_and_poll(
                    self._core_transfer_operations[handle.operation],
                    **kwargs,
                    notification=handle.notification,
                )
        else:
            method = getattr(endpoint, handle.operation.lower())
            if handle.notification is None:
                work = method(
                    handle.plan,
                    recovery_token=handle,
                )
            else:
                work = method(
                    handle.plan,
                    notification=handle.notification,
                    recovery_token=handle,
                )
        with handle._state_lock:
            handle.work = work
            handle.first_state = first_state
            # Core Work now owns its authenticated private snapshot. Drop the
            # adapter's temporary immutable inputs on the successful boundary.
            handle.local_indices = None
            handle.remote_indices = None
            handle.phase = "posted"

    def _recover_interrupted_post_work(
        self,
        handle: _WorkHandle,
        submission_error: BaseException | None = None,
    ) -> Literal["absent", "recovered", "pending"]:
        """Recover one Core Work by an exact, caller-owned identity token."""

        # Claim recovery with a state-only critical section. The Core ownership
        # snapshot below must not run under either an agent or handle lock, and
        # a second recovery caller must not race adoption of the same Work.
        with handle._state_lock:
            if handle.work is not None:
                handle.phase = "posted"
                return "recovered"
            if handle.phase not in ("posting", "recovery_pending"):
                return "absent"
            if handle._recovery_in_progress:
                return "pending"
            handle.phase = "recovery_pending"
            handle._recovery_in_progress = True
        match = None
        matches = ()
        try:
            # This snapshot stays entirely off the successful path. Distractor
            # Works cannot be adopted: token comparison is strictly ``is``.
            matches = tuple(
                work
                for work in self._require_endpoint().open_works
                if work.recovery_token is handle
            )
        except BaseException as recovery_error:
            self._note_recovery_issue(submission_error, recovery_error)
            recovery = "pending"
        else:
            if not matches:
                recovery = "absent"
            elif len(matches) == 1:
                match = matches[0]
                recovery = "recovered"
            else:
                recovery = "pending"
        if len(matches) > 1:
            self._note_recovery_issue(
                submission_error,
                RuntimeError(
                    "multiple Core Works carry one SGLang recovery-token identity"
                ),
            )
        with handle._state_lock:
            try:
                if recovery == "recovered":
                    handle.work = match
                    handle.phase = "posted"
                elif recovery == "absent":
                    # Absence is authoritative only for the recovery owner. Mark
                    # terminal before releasing the claim so no retry can race
                    # between the empty snapshot and terminal publication.
                    handle.terminal_state = "ERR"
                    handle.phase = "terminal"
            finally:
                handle._recovery_in_progress = False
        return recovery

    def _handle_interrupted_submission(
        self,
        handle: _WorkHandle,
        submission_error: BaseException,
    ) -> None:
        with handle._state_lock:
            handle.submission_interrupted = (
                handle.phase
                in (
                    "posting",
                    "recovery_pending",
                    "posted",
                )
                or handle.work is not None
            )
        recovery = self._recover_interrupted_post_work(handle, submission_error)
        if recovery == "absent":
            self._close_failed_submission(
                handle,
                submission_error,
                schedule=True,
            )
            return
        if recovery == "recovered":
            self._quarantine_recovered_submission(handle, schedule=True)
            return
        self._retain_recovery_pending(handle, schedule=True)

    def _retry_interrupted_post_recovery(
        self,
        handle: _WorkHandle,
        *,
        schedule: bool,
    ) -> None:
        with handle._state_lock:
            if handle.phase != "recovery_pending":
                return
        recovery = self._recover_interrupted_post_work(handle)
        if recovery == "pending":
            self._retain_recovery_pending(handle, schedule=schedule)
            return
        if recovery == "absent":
            self._close_failed_submission(handle, schedule=schedule)
            return
        self._quarantine_recovered_submission(handle, schedule=schedule)

    def _quarantine_recovered_submission(
        self,
        handle: _WorkHandle,
        *,
        schedule: bool,
    ) -> None:
        with handle._state_lock:
            work = handle.work
        assert work is not None
        try:
            work.cancel()
        except Exception as exc:
            logger.warning("PyTorch interrupted Work cancellation failed: %s", exc)
        if self._work_state(handle, release=False) in ("DONE", "ERR"):
            self._close_failed_submission(handle, schedule=schedule)
            return
        self._retain_recovery_pending(handle, schedule=schedule)

    def _close_failed_submission(
        self,
        handle: _WorkHandle,
        submission_error: BaseException | None = None,
        *,
        schedule: bool,
    ) -> None:
        try:
            self._release_work(handle)
        except BaseException as cleanup_error:
            self._note_recovery_issue(submission_error, cleanup_error)
            logger.warning("PyTorch failed-post cleanup failed: %s", cleanup_error)
            self._retain_recovery_pending(handle, schedule=schedule)
            if not isinstance(cleanup_error, Exception):
                raise

    def _retain_recovery_pending(
        self,
        handle: _WorkHandle,
        *,
        schedule: bool,
    ) -> None:
        retained = False
        with self._lock:
            if self._works.get(id(handle)) is handle:
                self._retained_work_ids.add(id(handle))
                retained = True
        if schedule and retained:
            self._start_retained_work_sweeper()

    def _require_no_unsettled_interrupted_submission(self, action: str) -> None:
        """Fail closed using state only; callers may already own ``_lock``.

        Recovery and status observation are driven by admitted transfer or
        sweeper calls. A teardown path must never hide either Core operation
        beneath the agent RLock merely to make cleanup opportunistically pass.
        """

        unsettled = []
        for handle in tuple(self._works.values()):
            with handle._state_lock:
                phase = handle.phase
                submission_interrupted = handle.submission_interrupted
            if phase in ("posting", "recovery_pending"):
                unsettled.append(handle)
                continue
            if not submission_interrupted or phase not in (
                "posted",
                "terminal",
            ):
                continue
            unsettled.append(handle)
        if unsettled:
            raise RuntimeError(
                f"Cannot {action} with {len(unsettled)} interrupted PyTorch "
                "submission(s) unsettled"
            )

    def _ensure_detached(self, handle: _WorkHandle) -> None:
        with self._lock:
            if self._works.get(id(handle)) is not handle:
                return
            if not any(candidate is handle for candidate in self._detached_works):
                self._detached_works.append(handle)

    @staticmethod
    def _note_recovery_issue(
        submission_error: BaseException | None,
        recovery_error: BaseException,
    ) -> None:
        if submission_error is None:
            return
        add_note = getattr(submission_error, "add_note", None)
        if callable(add_note):
            try:
                add_note(
                    "PyTorch interrupted-post Work recovery reported "
                    f"{type(recovery_error).__qualname__}"
                )
            except BaseException:  # noqa: S110
                pass

    def _track_work(self, handle: _WorkHandle) -> _WorkHandle:
        batch_handles = getattr(self._batch_handles, "value", None)
        try:
            with self._lock:
                if batch_handles is not None and handle.operation is not None:
                    # This publication precedes provider entry. If a later helper
                    # frame is interrupted, the room-level barrier still owns it.
                    batch_handles.append(handle)
                # Publish to the graph only after a worker batch can reach it.
                self._works[id(handle)] = handle
            return handle
        except BaseException:
            with self._lock:
                self._works.pop(id(handle), None)
                if batch_handles is not None:
                    batch_handles[:] = [
                        candidate
                        for candidate in batch_handles
                        if candidate is not handle
                    ]
            raise

    def _progress(self) -> None:
        self._advance_provider()
        self._reap_works()

    def _advance_provider(self) -> None:
        endpoint = self._require_endpoint()
        if not self._background_progress:
            try:
                endpoint.progress()
            except Exception as exc:
                # Global progress is an observation boundary too: failure does
                # not prove that any request is terminal. Subsequent per-Work
                # probes retain every unknown request and recover independently.
                logger.warning(
                    "PyTorch transfer progress is temporarily unknown: %s", exc
                )

    def _reap_works(self) -> None:
        with self._lock:
            handles = tuple(self._works.values())
        self._reap_handles(handles)

    def _reap_detached_works(self) -> None:
        with self._lock:
            handles = tuple(self._detached_works)
        if not handles:
            return
        self._reap_handles(handles)

    def _reap_retained_works(self) -> None:
        retained = []
        seen = set()
        with self._lock:
            for handle_id in self._retained_work_ids:
                handle = self._works.get(handle_id)
                if handle is not None:
                    retained.append(handle)
                    seen.add(handle_id)
            for handles in self._retained_handle_batches.values():
                for handle in handles:
                    handle_id = id(handle)
                    if handle_id in seen or self._works.get(handle_id) is not handle:
                        continue
                    retained.append(handle)
                    seen.add(handle_id)
        for handle in retained:
            with handle._state_lock:
                if handle.phase == "prepared":
                    handle.terminal_state = "ERR"
                    handle.phase = "terminal"
        self._reap_handles(tuple(retained))
        with self._lock:
            self._prune_retained_handle_batches_unlocked()

    def _reap_handles(self, handles: Sequence[_WorkHandle]) -> None:
        for handle in handles:
            with self._lock:
                tracked = self._works.get(id(handle)) is handle
            if not tracked:
                continue
            with handle._state_lock:
                phase = handle.phase
            if phase == "recovery_pending":
                self._retry_interrupted_post_recovery(handle, schedule=False)
                with handle._state_lock:
                    phase = handle.phase
            if phase in ("prepared", "posting", "recovery_pending"):
                continue
            if self._work_state(handle, release=False) not in ("DONE", "ERR"):
                continue
            try:
                self._release_work(handle)
            except Exception as exc:
                # The Work remains tracked, and the next progress call retries
                # cleanup without losing its terminal state.
                logger.warning("PyTorch terminal Work cleanup failed: %s", exc)
        with self._lock:
            self._detached_works = [
                handle
                for handle in self._detached_works
                if self._works.get(id(handle)) is handle
            ]
            self._retained_work_ids.intersection_update(self._works)

    def _start_retained_work_sweeper(self) -> None:
        with self._lock:
            if self._sweep_thread is not None and self._sweep_thread.is_alive():
                self._sweep_wakeup.set()
                return
            self._sweep_wakeup.clear()
            self._sweep_thread = threading.Thread(
                target=self._sweep_retained_works,
                name=f"{self.name}-retained-work-sweeper",
                daemon=True,
            )
            self._sweep_thread.start()

    def _sweep_retained_works(self) -> None:
        """Periodically reap abandoned siblings without freeing active Works."""

        while True:
            self._sweep_wakeup.wait(_RETAINED_WORK_SWEEP_INTERVAL_S)
            self._sweep_wakeup.clear()
            with self._lock:
                if self._endpoint is None:
                    self._sweep_thread = None
                    return
                self._retained_work_ids.intersection_update(self._works)
                self._prune_retained_handle_batches_unlocked()
                if not self._retained_work_ids and not self._retained_handle_batches:
                    self._sweep_thread = None
                    return
            try:
                # Abandoned siblings need autonomous provider progress, but no
                # adapter lock spans provider progress or Work-state refresh.
                with self._admitted_call():
                    self._advance_provider()
                    self._reap_retained_works()
                    self._reap_detached_works()
            except RuntimeError:
                with self._lock:
                    if self._closing or self._endpoint is None:
                        self._sweep_thread = None
                        return
                raise

    def _prune_retained_handle_batches_unlocked(self) -> None:
        for batch_id, handles in tuple(self._retained_handle_batches.items()):
            if not any(self._works.get(id(handle)) is handle for handle in handles):
                self._retained_handle_batches.pop(batch_id, None)

    def _work_state(self, handle: _WorkHandle, *, release: bool) -> str:
        while True:
            with handle._state_lock:
                terminal_state = handle.terminal_state
                phase = handle.phase
                if terminal_state is not None:
                    break
                if phase == "prepared":
                    raise RuntimeError("PyTorch transfer has not been posted")
                if phase == "posting":
                    raise RuntimeError(
                        "PyTorch transfer submission is still in progress"
                    )
                if phase == "recovery_pending":
                    recover = True
                else:
                    recover = False
                    if handle.work is None:
                        raise RuntimeError("Posted PyTorch transfer has no Work")
                    if handle._status_in_progress:
                        return "PROC"
                    handle._status_in_progress = True
                    work = handle.work
                    state = handle.first_state
                    handle.first_state = None
            if not recover:
                break
            self._retry_interrupted_post_recovery(handle, schedule=True)
            with handle._state_lock:
                recovery_pending = handle.phase == "recovery_pending"
            if recovery_pending:
                return "PROC"

        if terminal_state is not None:
            if release:
                self._release_work(handle)
            return terminal_state
        try:
            # Work.state is the sole authoritative observation. In current
            # Core it refreshes once and already incorporates provider errors;
            # reading test(), error, and state separately would refresh the
            # same Work up to three times.
            if state is None:
                state = work.state
            outcomes = self._work_state_outcomes
            if outcomes is not None:
                try:
                    terminal = outcomes.get(state, _UNKNOWN_WORK_OUTCOME)
                except TypeError:
                    terminal = _UNKNOWN_WORK_OUTCOME
                if terminal is _UNKNOWN_WORK_OUTCOME:
                    raise ValueError(f"Unknown PyTorch WorkState: {state!r}")
                if terminal is None:
                    return "PROC"
            else:
                state_name = self._state_name(state)
                if state_name in _PENDING_STATES:
                    return "PROC"
                if state_name in _DONE_STATES:
                    terminal = "DONE"
                elif state_name in _FAILED_STATES:
                    terminal = "ERR"
                else:
                    raise ValueError(f"Unknown PyTorch WorkState: {state!r}")
        except Exception as exc:
            # A status observation failure does not prove the transport is
            # terminal. Returning PROC preserves all memory/plan lifetimes and
            # permits a later poll to recover or report a real terminal error.
            logger.warning("PyTorch transfer status is temporarily unknown: %s", exc)
            return "PROC"
        finally:
            with handle._state_lock:
                handle._status_in_progress = False
        with handle._state_lock:
            if handle.submission_interrupted:
                terminal = "ERR"
            if handle.terminal_state is None:
                handle.terminal_state = terminal
                handle.phase = "terminal"
            else:
                terminal = handle.terminal_state
        if release:
            self._release_work(handle)
        return terminal

    @staticmethod
    def _complete_work_state_outcomes(
        api: Any,
    ) -> dict[object, str | None] | None:
        work_state = getattr(api, "WorkState", None)
        try:
            outcomes = {
                work_state.PENDING: None,
                work_state.RUNNING: None,
                work_state.COMPLETED: "DONE",
                work_state.FAILED: "ERR",
                work_state.CANCELLED: "ERR",
            }
        except (AttributeError, TypeError):
            return None
        return outcomes if len(outcomes) == 5 else None

    def _release_work(self, handle: _WorkHandle) -> None:
        with self._lock:
            tracked = self._works.get(id(handle)) is handle
        if not tracked:
            return
        with handle._state_lock:
            if handle.phase == "released" or handle._release_in_progress:
                return
            if handle.phase in ("posting", "recovery_pending"):
                raise RuntimeError(
                    "Cannot release a Work with unresolved submission ownership"
                )
            handle._release_in_progress = True
            work = handle.work
            owned_plan = handle.owned_plan
        try:
            if work is not None:
                work.close()
            if owned_plan is not None:
                owned_plan.close()
        except BaseException:
            with handle._state_lock:
                handle._release_in_progress = False
            raise
        with self._lock:
            if self._works.get(id(handle)) is handle:
                self._works.pop(id(handle), None)
            if handle.operation is None:
                self._detached_works = [
                    candidate
                    for candidate in self._detached_works
                    if candidate is not handle
                ]
        with handle._state_lock:
            handle.phase = "released"
            handle.local_indices = None
            handle.remote_indices = None
            handle.notification = None
            handle.plan = None
            handle._release_in_progress = False

    def _release_cached_plans(self, predicate: Callable[[_CachedPlan], bool]) -> None:
        # Move every matching plan out of the submit-visible cache before the
        # first close attempt. A partially committed provider close remains
        # discoverable here and may be retried, but can never be submitted.
        for key, cached in tuple(self._plan_cache.items()):
            if not predicate(cached):
                continue
            del self._plan_cache[key]
            self._closing_plans[key] = cached
        for key, cached in reversed(tuple(self._closing_plans.items())):
            if not predicate(cached):
                continue
            if any(handle.cache_key == key for handle in self._works.values()):
                raise RuntimeError(
                    "Prepared transfer plan is still referenced by a transfer handle"
                )
            cached.plan.close()
            del self._closing_plans[key]

    @staticmethod
    def _contains_identity(values: Sequence[object], candidate: object) -> bool:
        return any(value is candidate for value in values)

    def _reconcile_construction_state_unlocked(self, *, force: bool = False) -> None:
        """Close exact Core children absent from this adapter's owner graph."""

        if not force and not self._construction_dirty:
            return
        endpoint = self._require_endpoint()
        entries = tuple(self._plan_cache.values()) + tuple(self._closing_plans.values())
        owned_plans = tuple(entry.plan for entry in entries) + tuple(
            handle.owned_plan
            for handle in self._works.values()
            if handle.owned_plan is not None
        )
        owned_peers = tuple(self._peers.values())
        owned_registrations = tuple(
            record.registration for record in self._registrations
        )
        # SGLang does not currently construct bindings or request slots. They
        # are still scanned first so any future interrupted child cannot pin a
        # lost plan while being silently mistaken for an owned object.
        for attribute, owned in (
            ("open_transfer_slots", ()),
            ("open_bound_transfers", ()),
            ("open_plans", owned_plans),
            ("open_peers", owned_peers),
            ("open_registrations", owned_registrations),
        ):
            for child in tuple(getattr(endpoint, attribute)):
                if not self._contains_identity(owned, child):
                    child.close()
        self._construction_dirty = False

    def _reconcile_after_construction_error_unlocked(
        self, error: BaseException, kind: str
    ) -> None:
        try:
            self._reconcile_construction_state_unlocked(force=True)
        except BaseException as cleanup_error:
            self._add_exception_note(
                error,
                f"{kind} construction cleanup also raised: "
                f"{type(cleanup_error).__qualname__}",
            )

    def _matches_recovery_endpoint(
        self,
        candidate: object,
        *,
        endpoint_id: str,
        progress_mode: object,
        thread_mode: object,
    ) -> bool:
        return (
            type(candidate) is self._api.Endpoint
            and getattr(candidate, "endpoint_id", None) == endpoint_id
            and getattr(candidate, "name", None) == self.name
            and getattr(candidate, "backend", None) == self._endpoint_backend
            and getattr(candidate, "progress_mode", None) is progress_mode
            and getattr(candidate, "thread_mode", None) is thread_mode
        )

    @staticmethod
    def _close_recovery_endpoint(endpoint: object) -> None:
        if not getattr(endpoint, "closed", False):
            endpoint.close()
        if not getattr(endpoint, "closed", False):
            raise RuntimeError("Endpoint cleanup returned before close committed")

    def _retry_endpoint_cleanup_unlocked(self) -> None:
        endpoint = self._endpoint_recovery
        if endpoint is not None:
            self._close_recovery_endpoint(endpoint)
            self._endpoint_recovery = None

    def _retry_endpoint_recovery_unlocked(self) -> None:
        """Compatibility alias for focused lifecycle tests."""

        self._retry_endpoint_cleanup_unlocked()

    @staticmethod
    def _add_exception_note(error: BaseException, message: str) -> None:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            try:
                add_note(message)
            except BaseException:  # noqa: S110
                pass

    @staticmethod
    def _state_name(state: object) -> str | None:
        state = getattr(state, "name", state)
        return state.upper() if isinstance(state, str) else None

    @staticmethod
    def _normalize_operation(operation: str) -> Literal["READ", "WRITE"]:
        operation = operation.upper()
        if operation not in ("READ", "WRITE"):
            raise ValueError(f"Unsupported transfer operation: {operation!r}")
        return operation  # type: ignore[return-value]

    @staticmethod
    def _peer_source_ids(peer: object) -> tuple[str, ...]:
        identities = []
        for attribute in ("endpoint_id", "name", "source_endpoint_id"):
            value = getattr(peer, attribute, None)
            if value is not None:
                identity = str(value)
                if identity not in identities:
                    identities.append(identity)
        if not identities:
            raise TypeError("Imported peer does not expose an endpoint identity")
        return tuple(identities)

    def _require_endpoint(self) -> Any:
        if self._closing and self._close_owner != threading.get_ident():
            raise RuntimeError("PyTorch transfer endpoint close is in progress")
        if self._endpoint is None:
            raise RuntimeError("PyTorch transfer endpoint is not initialized")
        return self._endpoint

    def _require_peer(self, peer_name: str) -> Any:
        self._require_endpoint()
        try:
            return self._peers[peer_name]
        except KeyError as exc:
            raise KeyError(f"Unknown PyTorch transfer peer {peer_name!r}") from exc
