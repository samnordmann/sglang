"""Basic CPU unit tests for NIXL disaggregation control paths."""

import struct
import sys
import threading
import types
import unittest
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_handler import (
    STAGING_REQ_WIRE_TAG,
    STAGING_RSP_WIRE_TAG,
    PrefillStagingContext,
    handle_staging_req,
    handle_staging_rsp,
)
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.nixl.conn import (
    KVArgsRegisterInfo,
    NixlKVManager,
    NixlKVReceiver,
    NixlKVSender,
    TransferInfo,
    TransferKVChunk,
    TransferStatus,
    repeat_indices_over_layers,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

TRANSFER_GENERATION = "11" * 16
OLD_TRANSFER_GENERATION = "22" * 16


def completion_notification(room, suffix, generation=TRANSFER_GENERATION):
    return f"nixlv2_{generation}_{room}_{suffix}"


class NotificationFakeAgent:
    def __init__(self, messages):
        self.messages = messages

    def get_new_notifs(self):
        return {"peer": [msg.encode("ascii") for msg in self.messages]}


class StagingFakeAgent:
    def __init__(self, register_result=None):
        self.register_result = (
            register_result if register_result is not None else ["desc"]
        )
        self.register_memory_calls = []
        self.get_xfer_descs_calls = []
        self.initialize_xfer_calls = []
        self.transfer_calls = []

    def register_memory(self, addrs, mem_type):
        self.register_memory_calls.append((addrs, mem_type))
        return self.register_result

    def get_xfer_descs(self, reqs, mem_type):
        self.get_xfer_descs_calls.append((reqs, mem_type))
        return f"{mem_type}_{len(self.get_xfer_descs_calls)}"

    def initialize_xfer(self, *args):
        self.initialize_xfer_calls.append(args)
        return "handle"

    def transfer(self, handle):
        self.transfer_calls.append(handle)
        return "DONE"


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class FakeTensor:
    shape = (1, 1, 8)

    def element_size(self):
        return 2


class FakeStagingBuffer:
    def __init__(self, ptr=0x9000, size=1 << 20):
        self.ptr = ptr
        self.size = size

    def fits(self, required_bytes):
        return required_bytes <= self.size

    def get_ptr(self):
        return self.ptr


class FakeStagingAllocator:
    ALLOC_OVERSIZED = -2


def _fake_staging_buffer_module(mock_gather=None):
    module = types.ModuleType("sglang.srt.disaggregation.common.staging_buffer")
    module.StagingAllocator = FakeStagingAllocator
    module.compute_head_slice_params = lambda *args: (0, 1, 0, 1)
    module.compute_staging_layout = lambda *args: (2, [256, 256], 512)
    module.resolve_total_kv_heads = lambda kv_args, attn_tp_size: 2
    module.gather_all_layers_to_staging = mock_gather or MagicMock()
    return module


class TestNixlTransferInfo(CustomTestCase):
    def test_from_zmq_parses_required_fields(self):
        kv_indices = np.array([3, 5, 8], dtype=np.int32)
        state_indices = [[1, 2], [], [9]]
        msg = [
            b"7",
            b"127.0.0.1",
            b"12345",
            b"decode_agent",
            kv_indices.tobytes(),
            b"4",
            b"2",
            pack_int_lists(state_indices, "i"),
            b"11",
            b"0",
            TRANSFER_GENERATION.encode("ascii"),
        ]

        info = TransferInfo.from_zmq(msg)

        self.assertEqual(info.room, 7)
        self.assertEqual(info.endpoint, "127.0.0.1")
        self.assertEqual(info.dst_port, 12345)
        self.assertEqual(info.agent_name, "decode_agent")
        np.testing.assert_array_equal(info.dst_kv_indices, kv_indices)
        self.assertEqual(info.dst_aux_index, 4)
        self.assertEqual(info.required_dst_info_num, 2)
        self.assertEqual(info.dst_state_indices, state_indices)
        self.assertEqual(info.decode_prefix_len, 11)
        self.assertEqual(info.transfer_generation, TRANSFER_GENERATION)

    def test_from_zmq_defaults_optional_fields(self):
        info = TransferInfo.from_zmq(
            [
                b"8",
                b"127.0.0.1",
                b"12346",
                b"agent",
                np.array([1], dtype=np.int32).tobytes(),
                b"0",
                b"1",
                b"",
                b"",
                b"",
                TRANSFER_GENERATION.encode("ascii"),
            ]
        )

        self.assertEqual(info.dst_state_indices, [])
        self.assertIsNone(info.decode_prefix_len)

    def test_decode_radix_full_hit_is_not_dummy(self):
        info = TransferInfo.from_zmq(
            [
                b"9",
                b"127.0.0.1",
                b"12347",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"128",
                b"",
                TRANSFER_GENERATION.encode("ascii"),
            ]
        )

        self.assertFalse(info.is_dummy())

    def test_empty_indices_without_decode_prefix_is_dummy(self):
        info = TransferInfo.from_zmq(
            [
                b"10",
                b"127.0.0.1",
                b"12348",
                b"agent",
                np.array([], dtype=np.int32).tobytes(),
                b"2",
                b"1",
                b"",
                b"0",
                b"",
                TRANSFER_GENERATION.encode("ascii"),
            ]
        )

        self.assertTrue(info.is_dummy())


class TestNixlKVArgsRegisterInfo(CustomTestCase):
    def test_from_zmq_preserves_unsigned_pointers_and_optional_fields(self):
        high_ptr = 0xFFFF_81AB_54E0_1000
        kv_ptrs = [high_ptr, high_ptr + 0x1000]
        aux_ptrs = [0x1000, 0x2000]
        state_ptrs = [[high_ptr + 0x2000], [high_ptr + 0x3000, high_ptr + 0x4000]]
        state_item_lens = [[64], [128, 256]]
        state_dims = [[16], [32, 64]]
        staging_ptr = high_ptr + 0x5000

        msg = [
            b"None",
            b"10.0.0.2",
            b"23456",
            b"agent_with_large_ptr",
            b"metadata",
            b"".join(struct.pack("Q", ptr) for ptr in kv_ptrs),
            b"".join(struct.pack("Q", ptr) for ptr in aux_ptrs),
            pack_int_lists(state_ptrs, "Q"),
            b"3",
            b"4",
            b"1",
            b"1024",
            pack_int_lists(state_item_lens, "I"),
            pack_int_lists(state_dims, "I"),
            struct.pack("Q", staging_ptr),
            b"1048576",
            b"64",
            b"DRAM,DRAM",
            b"".join(struct.pack("Q", item_len) for item_len in [1024, 2048]),
            pack_int_lists([[4], [4, 5]], "I"),
            b"".join(struct.pack("I", layer_id) for layer_id in [2, 7]),
            b"4",
            b"3",
        ]

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.room, "None")
        self.assertEqual(info.endpoint, "10.0.0.2")
        self.assertEqual(info.dst_port, 23456)
        self.assertEqual(info.agent_name, "agent_with_large_ptr")
        self.assertEqual(info.agent_metadata, b"metadata")
        self.assertEqual(info.dst_kv_ptrs, kv_ptrs)
        self.assertEqual(info.dst_aux_ptrs, aux_ptrs)
        self.assertEqual(info.dst_state_data_ptrs, state_ptrs)
        self.assertEqual(info.gpu_id, 3)
        self.assertEqual(info.decode_tp_size, 4)
        self.assertEqual(info.decode_tp_rank, 1)
        self.assertEqual(info.dst_kv_item_len, 1024)
        self.assertEqual(info.dst_kv_item_lens, [1024, 2048])
        self.assertEqual(info.dst_num_slots, 64)
        self.assertEqual(info.dst_kv_mem_kinds, ["DRAM", "DRAM"])
        self.assertEqual(info.dst_state_item_lens, state_item_lens)
        self.assertEqual(info.dst_state_dim_per_tensor, state_dims)
        self.assertEqual(info.dst_dcp_size, 4)
        self.assertEqual(info.dst_dcp_rank, 3)
        self.assertEqual(info.dst_state_layer_ids, [[4], [4, 5]])
        self.assertEqual(info.dst_kv_layer_ids, [2, 7])
        self.assertEqual(info.staging_base_ptr, staging_ptr)
        self.assertEqual(info.staging_total_size, 1048576)

    def test_from_zmq_allows_missing_state_and_staging_fields(self):
        msg = [
            b"None",
            b"10.0.0.3",
            b"23457",
            b"agent",
            b"metadata",
            struct.pack("Q", 0x1000),
            struct.pack("Q", 0x2000),
            b"",
            b"0",
            b"1",
            b"0",
            b"256",
        ]

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.dst_state_data_ptrs, [])
        self.assertEqual(info.dst_state_item_lens, [])
        self.assertEqual(info.dst_state_dim_per_tensor, [])
        self.assertEqual(info.dst_kv_item_lens, [256])
        self.assertEqual(info.dst_dcp_size, 1)
        self.assertEqual(info.dst_dcp_rank, 0)
        self.assertEqual(info.staging_base_ptr, 0)
        self.assertEqual(info.staging_total_size, 0)


class TestNixlTransferStatus(CustomTestCase):
    def test_completion_source_rank_set_is_exact_and_immutable(self):
        status = TransferStatus(transfer_generation=TRANSFER_GENERATION)
        status.bind_expected_source_ranks(frozenset({4, 9}))

        self.assertTrue(status.accepts_source_rank(4))
        for unexpected in (True, -1, 0, 10):
            self.assertFalse(status.accepts_source_rank(unexpected))
        self.assertEqual(status.num_pp_ranks_expected, 2)
        status.bind_expected_source_ranks(frozenset({4, 9}))
        with self.assertRaises(RuntimeError):
            status.bind_expected_source_ranks(frozenset({4}))

    def test_not_done_until_aux_and_expected_count_arrive(self):
        status = TransferStatus(transfer_generation=TRANSFER_GENERATION)

        self.assertFalse(status.is_done())

        status.received_aux = True
        self.assertFalse(status.is_done())

        status.num_pp_ranks_expected = 1
        self.assertFalse(status.is_done())

        status.expected_kvs_per_pp[0] = 1
        self.assertFalse(status.is_done())

        status.received_kvs_per_pp[0].add(0)
        self.assertTrue(status.is_done())

    def test_zero_kv_aux_only_completion(self):
        status = TransferStatus(transfer_generation=TRANSFER_GENERATION)
        status.received_aux = True
        status.num_pp_ranks_expected = 1
        status.expected_kvs_per_pp[0] = 0

        self.assertTrue(status.is_done())

    def test_multi_pp_requires_each_rank_expected_chunks(self):
        status = TransferStatus(transfer_generation=TRANSFER_GENERATION)
        status.received_aux = True
        status.num_pp_ranks_expected = 2
        status.expected_kvs_per_pp[0] = 1
        status.received_kvs_per_pp[0].add(0)

        self.assertFalse(status.is_done())

        status.expected_kvs_per_pp[1] = 2
        status.received_kvs_per_pp[1].update({0, 1})
        self.assertTrue(status.is_done())

    def test_state_required_completion_waits_for_all_pp_ranks(self):
        status = TransferStatus(transfer_generation=TRANSFER_GENERATION)
        status.received_aux = True
        status.num_pp_ranks_expected = 2
        status.expected_kvs_per_pp[0] = 0
        status.expected_kvs_per_pp[1] = 0
        status.expects_state = True

        self.assertFalse(status.is_done())

        status.received_state_per_pp.add(0)
        self.assertFalse(status.is_done())

        status.received_state_per_pp.add(1)
        self.assertTrue(status.is_done())


class TestNixlKVSenderChunkPolicy(CustomTestCase):
    def test_last_zero_page_chunk_is_sent_for_aux_only_completion(self):
        sender = object.__new__(NixlKVSender)

        self.assertTrue(sender.should_send_kv_chunk(0, last_chunk=True))
        self.assertFalse(sender.should_send_kv_chunk(0, last_chunk=False))
        self.assertTrue(sender.should_send_kv_chunk(3, last_chunk=False))

    def test_both_data_planes_support_overlap_early_send(self):
        sender = object.__new__(NixlKVSender)

        for use_torch_transfer in (False, True):
            sender.kv_mgr = SimpleNamespace(_use_torch_transfer=use_torch_transfer)
            self.assertTrue(sender.supports_overlap_early_send)

    def test_send_consumes_early_send_event_into_transfer_chunk(self):
        sender = object.__new__(NixlKVSender)
        wait_event = object()
        sender._early_send_wait_event = wait_event
        sender._send_failed = False
        sender._transfer_start_time = 1.0
        sender.bootstrap_room = 7
        sender.chunk_id = 2
        sender.aux_index = 3
        sender.has_sent = False
        sender.kv_mgr = SimpleNamespace(add_transfer_request=MagicMock())
        prepared_indices = np.array([5], dtype=np.int32)
        sender._prepare_send_indices = MagicMock(
            return_value=(
                prepared_indices,
                slice(0, 1),
                False,
                False,
            )
        )
        sender._record_transfer_indices = MagicMock()

        sender.send(np.array([5], dtype=np.int32), num_kv_tokens=4)

        self.assertIsNone(sender._early_send_wait_event)
        sender.kv_mgr.add_transfer_request.assert_called_once_with(
            7,
            prepared_indices,
            slice(0, 1),
            False,
            2,
            3,
            None,
            4,
            wait_event=wait_event,
        )


class TestNixlBootstrapErrors(CustomTestCase):
    def _make_manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr.enable_staging = False
        mgr._bootstrap_errors = {}
        mgr.exceptions = {}
        mgr.transfer_infos = {}
        mgr.req_to_decode_prefix_len = {}
        mgr.prep_handles = {}
        mgr._handle_abort_notification = MagicMock(return_value=False)
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock()
        return mgr

    def test_registration_error_is_propagated_to_later_room(self):
        mgr = self._make_manager()
        unsupported = NotImplementedError("unsupported PyTorch transfer topology")
        mgr._add_remote_peer = MagicMock(side_effect=unsupported)
        registration_message = [
            b"NixlMsgGuard",
            b"None",
            b"endpoint",
            b"1234",
            b"decode-agent",
        ]
        room_message = [
            b"NixlMsgGuard",
            b"41",
            b"endpoint",
            b"1234",
            b"decode-agent",
            b"",
            b"0",
            b"1",
            b"",
            b"",
            b"0",
            TRANSFER_GENERATION.encode("ascii"),
        ]

        with patch.object(KVArgsRegisterInfo, "from_zmq", return_value=object()):
            with self.assertRaises(NotImplementedError):
                mgr._handle_bootstrap_message(registration_message)
        self.assertIs(mgr._bootstrap_errors["decode-agent"], unsupported)

        with self.assertRaises(NotImplementedError):
            mgr._handle_bootstrap_message(room_message)

        self.assertIs(mgr.exceptions[41], unsupported)
        mgr.record_failure.assert_called_once_with(41, str(unsupported))
        mgr.update_status.assert_called_once_with(41, KVPoll.Failed)

    def test_listener_continues_after_message_error(self):
        mgr = self._make_manager()
        mgr.server_socket = SimpleNamespace(
            recv_multipart=MagicMock(side_effect=[[b"bad"], [b"good"], SystemExit()])
        )
        mgr._handle_bootstrap_message = MagicMock(
            side_effect=[ValueError("bad bootstrap"), None]
        )

        with patch(
            "sglang.srt.disaggregation.nixl.conn.threading.Thread"
        ) as thread_cls:
            mgr._start_bootstrap_thread()
        target = thread_cls.call_args.kwargs["target"]

        with self.assertRaises(SystemExit):
            target()

        self.assertEqual(mgr._handle_bootstrap_message.call_count, 2)
        self.assertTrue(thread_cls.call_args.kwargs["daemon"])

    def test_peer_import_failure_does_not_poison_registration_table(self):
        mgr = self._make_manager()
        mgr._use_torch_transfer = True
        mgr.attn_tp_size = 1
        mgr.disaggregation_mode = object()
        mgr.decode_kv_args_table = {}
        mgr.requires_dcp_relayout = MagicMock(return_value=False)
        mgr.agent = SimpleNamespace(
            add_remote_agent=MagicMock(side_effect=RuntimeError("metadata rejected"))
        )
        peer = SimpleNamespace(
            agent_name="decode-agent",
            agent_metadata=b"metadata",
            decode_tp_size=1,
            dst_dcp_size=1,
            dst_dcp_rank=0,
            dst_kv_mem_kinds=["VRAM"],
        )

        with self.assertRaisesRegex(RuntimeError, "metadata rejected"):
            mgr._add_remote_peer(peer)

        self.assertNotIn("decode-agent", mgr.decode_kv_args_table)


class TestNixlAbortHandling(CustomTestCase):
    def _make_manager(self, request_status=None):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = dict(request_status or {})
        mgr._connect = MagicMock()
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr._abort_state_lock = threading.Lock()
        mgr._deferred_ack_targets = {}
        mgr._active_transfer_generations = {}
        mgr._abort_ack_receipts = set()
        mgr._staging_outstanding = {room: 1 for room in mgr.request_status}
        mgr._send_abort_ack = MagicMock(return_value=True)
        mgr.enable_deferred_decode_kv_release = False
        for room in mgr.request_status:
            mgr.register_transfer_generation(room, TRANSFER_GENERATION)
        return mgr

    def test_given_known_incomplete_room_when_abort_arrives_then_room_fails_without_ack(
        self,
    ):
        mgr = self._make_manager({11: KVPoll.WaitingForInput})

        handled = mgr._handle_abort_notification(
            [
                b"ABORT",
                b"11",
                b"127.0.0.1",
                b"5555",
                TRANSFER_GENERATION.encode("ascii"),
            ]
        )

        self.assertTrue(handled)
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)
        self.assertEqual(
            mgr.failure_records[11],
            "Aborted by decode-side abort notification.",
        )
        mgr._connect.assert_not_called()

    def test_given_successful_room_when_abort_arrives_then_status_is_preserved(self):
        mgr = self._make_manager({12: KVPoll.Success})

        handled = mgr._handle_abort_notification(
            [
                b"ABORT",
                b"12",
                b"127.0.0.1",
                b"5556",
                TRANSFER_GENERATION.encode("ascii"),
            ]
        )

        self.assertTrue(handled)
        self.assertEqual(mgr.request_status[12], KVPoll.Success)
        self.assertEqual(mgr.failure_records, {})
        mgr._connect.assert_not_called()

    def test_given_unknown_room_when_abort_arrives_then_status_remains_absent(self):
        mgr = self._make_manager()

        handled = mgr._handle_abort_notification(
            [
                b"ABORT",
                b"14",
                b"127.0.0.1",
                b"5557",
                TRANSFER_GENERATION.encode("ascii"),
            ]
        )

        self.assertTrue(handled)
        self.assertNotIn(14, mgr.request_status)
        self.assertEqual(mgr.failure_records, {})
        mgr._connect.assert_not_called()

    def test_given_malformed_abort_when_handled_then_no_exception_or_ack(self):
        mgr = self._make_manager({13: KVPoll.WaitingForInput})

        handled = mgr._handle_abort_notification(
            [
                b"ABORT",
                b"invalid-room",
                b"127.0.0.1",
                b"5558",
                TRANSFER_GENERATION.encode("ascii"),
            ]
        )

        self.assertTrue(handled)
        self.assertEqual(mgr.request_status[13], KVPoll.WaitingForInput)
        self.assertEqual(mgr.failure_records, {})
        mgr._connect.assert_not_called()


class TestNixlUpdateStatus(CustomTestCase):
    def _make_manager(self, request_status):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = dict(request_status)
        return mgr

    def test_given_failed_room_when_status_is_promoted_then_failed_is_preserved(self):
        for status in (KVPoll.Transferring, KVPoll.Success):
            with self.subTest(status=status):
                mgr = self._make_manager({17: KVPoll.Failed})

                mgr.update_status(17, status)

                self.assertEqual(mgr.request_status[17], KVPoll.Failed)

    def test_given_missing_room_when_failed_update_arrives_then_room_is_not_resurrected(
        self,
    ):
        mgr = self._make_manager({})

        mgr.update_status(18, KVPoll.Failed)

        self.assertNotIn(18, mgr.request_status)


class TestNixlTransferWorker(CustomTestCase):
    def _make_manager(self, room):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = {room: KVPoll.WaitingForInput}
        mgr.transfer_infos = {
            room: {
                "agent": TransferInfo(
                    room=room,
                    endpoint="127.0.0.1",
                    dst_port=5555,
                    agent_name="agent",
                    dst_kv_indices=np.array([2], dtype=np.int32),
                    dst_aux_index=0,
                    required_dst_info_num=1,
                    dst_state_indices=[],
                    transfer_generation=TRANSFER_GENERATION,
                )
            }
        }
        mgr.decode_kv_args_table = {
            "agent": SimpleNamespace(
                decode_tp_size=1,
                dst_kv_ptrs=[0],
                dst_aux_ptrs=[0],
                gpu_id=0,
                staging_base_ptr=0,
                staging_total_size=0,
                kv_xfer_segments=None,
                dst_homogeneous_mem_kind="VRAM",
                # Non-DCP peer. Without this the worker raises AttributeError
                # and lands in the same Failed status the assertions expect,
                # so the transfer path would go unexercised.
                requires_dcp_relayout=False,
                dcp_dst_region_indices=None,
                dcp_token_item_lens=None,
            )
        }
        mgr.req_to_decode_prefix_len = {room: 4}
        mgr.enable_staging = False
        mgr.enable_deferred_decode_kv_release = False
        mgr._staging_ctx = None
        mgr._staging_outstanding = defaultdict(int)
        mgr._abort_state_lock = threading.Lock()
        mgr._deferred_ack_targets = {}
        mgr._active_transfer_generations = {room: TRANSFER_GENERATION}
        mgr._abort_ack_receipts = set()
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.transfer_source_rank = 0
        mgr.kv_args = SimpleNamespace(engine_rank=0, kv_data_ptrs=[0])
        mgr._use_torch_transfer = False
        mgr.exceptions = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}

        def check_xfer_state(_handle):
            mgr.update_status(room, KVPoll.Failed)
            return "DONE"

        mgr.agent = SimpleNamespace(check_xfer_state=check_xfer_state)
        return mgr

    def _make_chunk(self, room, prefill_kv_indices, is_last_chunk):
        return TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array(prefill_kv_indices, dtype=np.int32),
            index_slice=slice(0, len(prefill_kv_indices)),
            is_last_chunk=is_last_chunk,
            chunk_id=0,
            prefill_aux_index=0 if is_last_chunk else None,
            state_indices=None,
        )

    def _run_worker_once(self, mgr, chunk):
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

    def test_given_last_chunk_aborts_mid_transfer_when_worker_finishes_then_failed_status_is_preserved(
        self,
    ):
        room = 21
        mgr = self._make_manager(room)
        mgr.send_aux = MagicMock(return_value="aux_handle")
        chunk = self._make_chunk(room, [], is_last_chunk=True)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)
        mgr.send_aux.assert_called_once()
        self.assertEqual(
            mgr.send_aux.call_args.args[-1],
            completion_notification(21, "aux_nokv_0_0"),
        )

    def test_given_non_last_chunk_aborts_mid_transfer_then_failed_room_is_retired(
        self,
    ):
        room = 22
        mgr = self._make_manager(room)
        mgr.send_kvcache = MagicMock(return_value="kv_handle")
        chunk = self._make_chunk(room, [1], is_last_chunk=False)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)
        self.assertIn((room, TRANSFER_GENERATION), mgr._abort_ack_receipts)
        mgr.send_kvcache.assert_called_once()

    def test_waits_for_early_send_event_once_before_source_access(self):
        for room, use_torch_transfer in ((23, False), (24, True)):
            with self.subTest(use_torch_transfer=use_torch_transfer):
                mgr = self._make_manager(room)
                mgr._use_torch_transfer = use_torch_transfer
                mgr.agent.progress = MagicMock()
                actions = []
                captured_handles = []
                mgr.agent.begin_handle_batch = lambda handles: captured_handles.append(
                    handles
                )
                mgr.agent.end_handle_batch = lambda handles: self.assertIs(
                    captured_handles[-1], handles
                )
                wait_event = MagicMock()
                wait_event.synchronize.side_effect = lambda: actions.append("wait")

                def send_kvcache(*args, **kwargs):
                    actions.append("send")
                    if use_torch_transfer:
                        captured_handles[-1].append("handle")
                    return "handle"

                mgr.send_kvcache = MagicMock(side_effect=send_kvcache)
                chunk = self._make_chunk(room, [1], is_last_chunk=False)
                chunk.wait_event = wait_event

                self._run_worker_once(mgr, chunk)

                self.assertEqual(actions, ["wait", "send"])
                wait_event.synchronize.assert_called_once_with()
                self.assertIsNone(chunk.wait_event)
                if use_torch_transfer:
                    mgr.agent.progress.assert_called_once_with()
                else:
                    mgr.agent.progress.assert_not_called()

    def test_torch_failure_is_published_only_after_every_handle_is_drained(self):
        room = 26
        mgr = self._make_manager(room)
        mgr._use_torch_transfer = True
        second_req = replace(mgr.transfer_infos[room]["agent"], agent_name="agent-2")
        mgr.transfer_infos[room]["agent-2"] = second_req
        mgr.decode_kv_args_table["agent-2"] = mgr.decode_kv_args_table["agent"]
        release_kv_cache = MagicMock()
        events = []
        batch = []
        original_update_status = mgr.update_status

        def send_kvcache(*args, **kwargs):
            handle = "failed" if not batch[-1] else "active"
            batch[-1].append(handle)
            return handle

        def update_status(candidate_room, status):
            events.append(("status", status))
            original_update_status(candidate_room, status)
            if status == KVPoll.Failed:
                release_kv_cache(candidate_room)

        def cancel_handles_and_wait(handles):
            events.append(("drain", tuple(handles)))
            self.assertEqual(mgr.request_status[room], KVPoll.Transferring)
            release_kv_cache.assert_not_called()

        mgr.update_status = update_status
        mgr.send_kvcache = MagicMock(side_effect=send_kvcache)
        mgr.agent = SimpleNamespace(
            begin_handle_batch=lambda handles: batch.append(handles),
            end_handle_batch=lambda handles: self.assertIs(batch[-1], handles),
            progress=MagicMock(),
            check_xfer_state=MagicMock(return_value="ERR"),
            pop_xfer_error=MagicMock(return_value=None),
            cancel_handles_and_wait=cancel_handles_and_wait,
        )
        chunk = self._make_chunk(room, [1], is_last_chunk=False)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(events[-2], ("drain", ("failed", "active")))
        self.assertEqual(events[-1], ("status", KVPoll.Failed))
        release_kv_cache.assert_called_once_with(room)
        self.assertEqual(mgr._staging_outstanding[room], 0)
        self.assertFalse(chunk.staging_counted)

    def test_fatal_worker_cut_waits_without_timeout_until_batch_is_quiescent(self):
        class FatalWorkerCut(BaseException):
            pass

        room = 28
        mgr = self._make_manager(room)
        mgr._use_torch_transfer = True
        mgr.enable_deferred_decode_kv_release = True
        events = []
        batches = []
        cleanup_attempts = 0
        original_update_status = mgr.update_status

        def begin(handles):
            batches.append(handles)
            events.append("begin")

        def send_then_die(*args, **kwargs):
            batches[-1].append("live-handle")
            raise FatalWorkerCut("worker cancelled")

        def retain(handles):
            self.assertIs(handles, batches[-1])
            self.assertEqual(mgr.request_status[room], KVPoll.Transferring)
            events.append("retain")

        def drain(handles):
            nonlocal cleanup_attempts
            cleanup_attempts += 1
            events.append(f"drain-{cleanup_attempts}")
            self.assertIs(handles, batches[-1])
            self.assertEqual(mgr.request_status[room], KVPoll.Transferring)
            if cleanup_attempts == 1:
                raise KeyboardInterrupt("cleanup interrupted")

        def update_status(candidate_room, status):
            original_update_status(candidate_room, status)
            if status == KVPoll.Failed:
                events.append("failed")

        def ack(candidate_room, generation=None):
            self.assertEqual(candidate_room, room)
            self.assertEqual(generation, TRANSFER_GENERATION)
            self.assertEqual(mgr.request_status[room], KVPoll.Failed)
            self.assertEqual(mgr._staging_outstanding[room], 0)
            self.assertFalse(chunk.staging_counted)
            events.append("ack")

        def end(handles):
            self.assertIs(handles, batches[-1])
            events.append("end")

        mgr.update_status = update_status
        mgr.send_kvcache = MagicMock(side_effect=send_then_die)
        mgr._maybe_ack_drained_abort = ack
        mgr.agent = SimpleNamespace(
            begin_handle_batch=begin,
            end_handle_batch=end,
            retain_handle_batch=retain,
            cancel_handles_and_wait=drain,
            handle_batch_is_quiescent=lambda handles: False,
        )
        chunk = self._make_chunk(room, [1], is_last_chunk=False)
        queue = SimpleNamespace(get=MagicMock(return_value=chunk))

        with self.assertRaisesRegex(FatalWorkerCut, "worker cancelled"):
            mgr.transfer_worker(queue)

        self.assertEqual(
            events,
            ["begin", "retain", "drain-1", "drain-2", "failed", "ack", "end"],
        )
        self.assertEqual(mgr._staging_outstanding[room], 0)
        self.assertFalse(chunk.staging_counted)
        self.assertIsInstance(mgr.exceptions[room], RuntimeError)

    def test_torch_peer_bootstrap_base_exception_rolls_back_exact_state(self):
        class BootstrapCut(BaseException):
            pass

        class StoreThenInterrupt(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                raise BootstrapCut("manager commit cut")

        room = 29
        mgr = self._make_manager(room)
        mgr._use_torch_transfer = True
        mgr.attn_tp_size = 1
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.requires_dcp_relayout = MagicMock(return_value=False)
        mgr.prep_handles = {"": "existing-local"}
        mgr.decode_kv_args_table = StoreThenInterrupt()
        released = []
        rolled_back = []
        mgr.agent = SimpleNamespace(
            add_remote_agent=MagicMock(return_value="agent"),
            release_dlist_handle=lambda handle: released.append(handle),
            rollback_remote_agent=lambda name, metadata: rolled_back.append(
                (name, metadata)
            ),
        )
        mgr._prepare_payload_xfer = lambda peer: mgr.prep_handles.__setitem__(
            "agent", "new-remote"
        )
        peer = SimpleNamespace(
            agent_name="agent",
            agent_metadata=b"peer-metadata",
            decode_tp_size=1,
            dst_dcp_size=1,
            dst_dcp_rank=0,
            dst_kv_mem_kinds=["VRAM"],
        )

        with self.assertRaisesRegex(BootstrapCut, "manager commit cut"):
            mgr._add_remote_peer(peer)

        self.assertNotIn("agent", mgr.decode_kv_args_table)
        self.assertEqual(mgr.prep_handles, {"": "existing-local"})
        self.assertEqual(released, ["new-remote"])
        self.assertEqual(rolled_back, [("agent", b"peer-metadata")])

    def test_partial_multi_submit_uses_worker_owned_handle_sink(self):
        room = 27
        mgr = self._make_manager(room)
        mgr._use_torch_transfer = True
        mgr.decode_kv_args_table["agent"].kv_xfer_segments = [object(), object()]
        drained = []

        def submit_then_fail(_peer, _src, _dst, _notif, handles):
            handles.append("first-active-part")
            raise RuntimeError("second part failed before publication")

        mgr.send_kvcache_mixed = MagicMock(side_effect=submit_then_fail)
        mgr.agent = SimpleNamespace(
            begin_handle_batch=lambda handles: None,
            end_handle_batch=lambda handles: None,
            progress=MagicMock(),
            check_xfer_state=MagicMock(),
            pop_xfer_error=MagicMock(return_value=None),
            cancel_handles_and_wait=lambda handles: drained.append(tuple(handles)),
        )
        chunk = self._make_chunk(room, [1], is_last_chunk=False)

        self._run_worker_once(mgr, chunk)

        self.assertEqual(drained, [("first-active-part",)])
        self.assertEqual(mgr.request_status[room], KVPoll.Failed)

    def test_dcp_destinations_use_disjoint_pack_regions_before_chunk_barrier(self):
        room = 25
        mgr = self._make_manager(room)
        agents = ("agent0a", "agent0b", "agent1")
        dcp_ranks = (0, 0, 1)
        mgr.transfer_infos[room] = {
            agent: TransferInfo(
                room=room,
                endpoint="127.0.0.1",
                dst_port=5555 + i,
                agent_name=agent,
                dst_kv_indices=np.array([2 + i], dtype=np.int32),
                dst_aux_index=0,
                required_dst_info_num=len(agents),
                dst_state_indices=[],
                transfer_generation=TRANSFER_GENERATION,
            )
            for i, agent in enumerate(agents)
        }
        mgr.decode_kv_args_table = {
            agent: SimpleNamespace(
                decode_tp_size=len(agents),
                dst_kv_ptrs=[0x3000 + i * 0x100],
                dst_aux_ptrs=[0],
                gpu_id=0,
                staging_base_ptr=0,
                staging_total_size=0,
                kv_xfer_segments=None,
                dst_homogeneous_mem_kind="VRAM",
                requires_dcp_relayout=True,
                dst_dcp_size=2,
                dst_dcp_rank=dcp_rank,
                dcp_dst_region_indices=[0],
                dcp_token_item_lens=[4],
            )
            for i, (agent, dcp_rank) in enumerate(zip(agents, dcp_ranks))
        }
        mgr.kv_args = SimpleNamespace(
            engine_rank=0,
            kv_data_ptrs=[0x1000],
            page_size=4,
        )
        mgr._dcp_pack_buffers = [SimpleNamespace(get_size=lambda: 16)]

        packed_rank0 = ([0x9000], np.arange(2, dtype=np.int64))
        packed_rank1 = ([0x9008], np.arange(2, dtype=np.int64))
        try_pack = MagicMock(side_effect=[packed_rank0, packed_rank1])
        dcp_pack_module = types.ModuleType("sglang.srt.disaggregation.common.dcp_pack")
        dcp_pack_module.try_pack_dcp_src = try_pack
        submitted = []

        def send_kvcache_dcp(*args, **kwargs):
            submitted.append((args[0], args[-1]))
            return f"handle-{args[0]}"

        mgr.send_kvcache_dcp = MagicMock(side_effect=send_kvcache_dcp)
        submitted_counts_at_poll = []

        def check_xfer_state(_handle):
            submitted_counts_at_poll.append(len(submitted))
            return "DONE"

        mgr.agent = SimpleNamespace(check_xfer_state=check_xfer_state)
        chunk = self._make_chunk(room, [1], is_last_chunk=False)
        chunk.num_kv_tokens = 4

        with patch.dict(
            sys.modules,
            {"sglang.srt.disaggregation.common.dcp_pack": dcp_pack_module},
        ):
            self._run_worker_once(mgr, chunk)

        self.assertEqual(try_pack.call_count, 2)
        self.assertEqual(
            [call.kwargs["pack_offset_bytes"] for call in try_pack.call_args_list],
            [0, 8],
        )
        self.assertEqual(
            submitted,
            [
                ("agent0a", packed_rank0),
                ("agent0b", packed_rank0),
                ("agent1", packed_rank1),
            ],
        )
        self.assertEqual(submitted_counts_at_poll, [3, 3, 3])


class TestNixlNotifications(CustomTestCase):
    def _make_manager(
        self,
        messages,
        *,
        room=5,
        required=None,
        generation=TRANSFER_GENERATION,
        expected_ranks=frozenset({0}),
    ):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = NotificationFakeAgent(messages)
        mgr.transfer_statuses = {
            room: TransferStatus(
                transfer_generation=generation,
                expected_source_ranks=expected_ranks,
            )
        }
        mgr.required_prefill_response_num_table = required or {}
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr._chunk_writer_counts = defaultdict(lambda: defaultdict(list))
        return mgr

    def test_kv_last_notification_sets_expected_count(self):
        mgr = self._make_manager([completion_notification(5, "kv_2_1_0")])

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_pp[0], {2})
        self.assertEqual(status.expected_kvs_per_pp[0], 3)
        self.assertEqual(status.num_pp_ranks_expected, 1)

    def test_staging_notification_preserves_agent_name_with_underscores(self):
        mgr = self._make_manager(
            [completion_notification(5, "stg_0_1_0_2_4_8_agent_with_underscores")]
        )
        calls = []
        mgr._handle_staging_chunk_arrived = lambda *args: calls.append(args)

        mgr.update_transfer_status()

        self.assertEqual(calls, [(5, 2, 4, 8, 0)])
        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_pp[0], {0})
        self.assertEqual(status.expected_kvs_per_pp[0], 1)

    def test_aux_nokv_marks_zero_expected_chunks_for_pp_rank(self):
        mgr = self._make_manager(
            [completion_notification(6, "aux_nokv_3_0")],
            room=6,
            required={6: 4},
            expected_ranks=frozenset({0, 1, 2, 3}),
        )

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[6]
        self.assertTrue(status.received_aux)
        self.assertEqual(status.expected_kvs_per_pp[3], 0)
        self.assertEqual(status.num_pp_ranks_expected, 4)

    def test_state_notification_marks_pp_rank(self):
        mgr = self._make_manager(
            [completion_notification(7, "state_2_4")],
            room=7,
            expected_ranks=frozenset({2}),
        )

        mgr.update_transfer_status()

        self.assertEqual(mgr.transfer_statuses[7].received_state_per_pp, {2})

    def test_aux_nokv_allows_full_hit_completion(self):
        mgr = self._make_manager(
            [completion_notification(8, "aux_nokv_0_0")],
            room=8,
            required={8: 1},
        )

        mgr.update_transfer_status()

        self.assertTrue(mgr.transfer_statuses[8].is_done())

    def test_mixed_parts_complete_only_after_all_generation_bound_parts(self):
        mgr = self._make_manager(
            [
                completion_notification(9, "kv_0_1_0_part_0_2"),
                completion_notification(9, "kv_0_1_0_part_1_2"),
            ],
            room=9,
        )

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[9]
        self.assertEqual(status.received_kvs_per_pp[0], {0})
        self.assertEqual(status.expected_kvs_per_pp[0], 1)

    def test_arbitrary_and_duplicate_source_ranks_cannot_satisfy_completion(self):
        mgr = self._make_manager(
            [
                completion_notification(10, "kv_0_1_99"),
                completion_notification(10, "aux_99"),
                completion_notification(10, "kv_0_1_0"),
                completion_notification(10, "kv_0_1_0"),
                completion_notification(10, "aux_0"),
            ],
            room=10,
            expected_ranks=frozenset({0, 1}),
        )

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[10]
        self.assertEqual(status.received_kvs_per_pp, {0: {0}})
        self.assertEqual(status.expected_kvs_per_pp, {0: 1})
        self.assertTrue(status.received_aux)
        self.assertFalse(status.is_done())

    def test_old_generation_cannot_mutate_any_completion_family_after_reuse(self):
        mgr = self._make_manager(
            [
                completion_notification(5, "kv_0_1_0", OLD_TRANSFER_GENERATION),
                completion_notification(
                    5, "kv_0_1_0_part_0_1", OLD_TRANSFER_GENERATION
                ),
                completion_notification(5, "state_0_0", OLD_TRANSFER_GENERATION),
                completion_notification(5, "aux_nokv_0_0", OLD_TRANSFER_GENERATION),
                completion_notification(
                    5,
                    "stg_0_1_0_0_0_1_old_agent",
                    OLD_TRANSFER_GENERATION,
                ),
            ]
        )
        staging_calls = []
        mgr._handle_staging_chunk_arrived = lambda *args: staging_calls.append(args)

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_pp, {})
        self.assertEqual(status.expected_kvs_per_pp, {})
        self.assertEqual(status.received_state_per_pp, set())
        self.assertFalse(status.received_aux)
        self.assertIsNone(status.received_kv_parts_per_pp)
        self.assertEqual(staging_calls, [])

    def test_legacy_and_malformed_notifications_never_create_or_mutate_status(self):
        mgr = self._make_manager(
            [
                "5_kv_0_1_0",
                f"nixlv1_{TRANSFER_GENERATION}_5_aux",
                completion_notification(5, "kv_-1_1_0"),
                completion_notification(5, "kv_0_2_0"),
                completion_notification(5, "aux_extra"),
                completion_notification(999, "aux_0"),
            ]
        )

        mgr.update_transfer_status()

        self.assertEqual(set(mgr.transfer_statuses), {5})
        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_pp, {})
        self.assertEqual(status.expected_kvs_per_pp, {})
        self.assertFalse(status.received_aux)


class TestGenerationBoundStagingControl(CustomTestCase):
    @staticmethod
    def _request(generation, *, legacy=False):
        if legacy:
            return [b"STAGING_REQ", b"7", b"0", b"1", b"peer", b"3"]
        return [
            STAGING_REQ_WIRE_TAG,
            generation.encode("ascii"),
            b"7",
            b"0",
            b"1",
            b"peer",
            b"3",
        ]

    def test_stale_and_legacy_staging_requests_cannot_allocate_reused_room(self):
        sock = MagicMock()
        receiver = SimpleNamespace(
            transfer_generation=TRANSFER_GENERATION,
            chunk_staging_infos=[],
            _connect_to_bootstrap_server=MagicMock(
                return_value=(sock, threading.Lock())
            ),
        )
        allocator = SimpleNamespace(
            assign=MagicMock(return_value=(3, 128, 0)), total_size=1 << 20
        )
        kwargs = dict(
            staging_allocator=allocator,
            kv_args=SimpleNamespace(
                page_size=64,
                kv_item_lens=[4096, 4096],
                total_kv_head_num=4,
                engine_rank=0,
            ),
            attn_tp_size=16,
            prefill_attn_tp_size=1,
            kv_buffer_tensors=None,
            room_receivers={7: receiver},
            room_bootstrap={7: [{"pp_rank": 3}]},
        )

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": _fake_staging_buffer_module()
            },
        ):
            self.assertFalse(
                handle_staging_req(self._request(OLD_TRANSFER_GENERATION), **kwargs)
            )
            self.assertFalse(
                handle_staging_req(
                    self._request(TRANSFER_GENERATION, legacy=True), **kwargs
                )
            )

        allocator.assign.assert_not_called()
        receiver._connect_to_bootstrap_server.assert_not_called()
        self.assertEqual(receiver.chunk_staging_infos, [])

    def test_staging_request_and_response_bind_exact_generation(self):
        sock = MagicMock()
        receiver = SimpleNamespace(
            transfer_generation=TRANSFER_GENERATION,
            chunk_staging_infos=[],
            _connect_to_bootstrap_server=MagicMock(
                return_value=(sock, threading.Lock())
            ),
        )
        allocator = SimpleNamespace(
            assign=MagicMock(return_value=(3, 128, 0)), total_size=1 << 20
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": _fake_staging_buffer_module()
            },
        ):
            self.assertTrue(
                handle_staging_req(
                    self._request(TRANSFER_GENERATION),
                    allocator,
                    SimpleNamespace(
                        page_size=64,
                        kv_item_lens=[4096, 4096],
                        total_kv_head_num=4,
                        engine_rank=0,
                    ),
                    attn_tp_size=16,
                    prefill_attn_tp_size=1,
                    kv_buffer_tensors=None,
                    room_receivers={7: receiver},
                    room_bootstrap={7: [{"pp_rank": 3}]},
                )
            )

        expected_rsp = [
            STAGING_RSP_WIRE_TAG,
            TRANSFER_GENERATION.encode("ascii"),
            b"7",
            b"0",
            b"128",
            b"0",
            b"640",
            b"peer",
        ]
        sock.send_multipart.assert_called_once_with(expected_rsp)

        tinfo = SimpleNamespace(
            transfer_generation=TRANSFER_GENERATION,
            staging=None,
        )
        transfer_infos = {7: {"peer": tinfo}}
        stale_rsp = list(expected_rsp)
        stale_rsp[1] = OLD_TRANSFER_GENERATION.encode("ascii")
        self.assertFalse(handle_staging_rsp(stale_rsp, transfer_infos))
        self.assertIsNone(tinfo.staging)
        self.assertFalse(
            handle_staging_rsp(
                [b"STAGING_RSP", b"7", b"0", b"128", b"0", b"640", b"peer"],
                transfer_infos,
            )
        )
        self.assertIsNone(tinfo.staging)

        self.assertTrue(handle_staging_rsp(expected_rsp, transfer_infos))
        self.assertEqual(tinfo.staging.offsets, [128])
        self.assertEqual(tinfo.staging.rounds, [0])
        self.assertEqual(tinfo.staging.ends, [640])


class TestNixlReceiverPoll(CustomTestCase):
    def test_supplied_generation_is_shared_and_read_only_across_decode_ranks(self):
        mgr = SimpleNamespace(
            transfer_statuses={},
            addr_to_rooms_tracker=defaultdict(set),
            update_status=MagicMock(),
        )

        first = NixlKVReceiver(
            mgr, "prefill:8998", 11, transfer_generation=TRANSFER_GENERATION
        )
        second = NixlKVReceiver(
            mgr, "prefill:8998", 11, transfer_generation=TRANSFER_GENERATION
        )

        self.assertEqual(first.transfer_generation, TRANSFER_GENERATION)
        self.assertEqual(second.transfer_generation, TRANSFER_GENERATION)
        self.assertEqual(
            mgr.transfer_statuses[11].transfer_generation, TRANSFER_GENERATION
        )
        with self.assertRaises(AttributeError):
            first.transfer_generation = OLD_TRANSFER_GENERATION

    def _make_receiver(self, status=KVPoll.WaitingForInput):
        mgr = MagicMock()
        mgr.waiting_timeout = 5
        mgr.check_status.return_value = status
        mgr.check_transfer_done.return_value = False
        mgr.transfer_statuses = {}
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker["prefill:8998"].add(11)

        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 11
        receiver.bootstrap_addr = "prefill:8998"
        receiver.started_transfer = False
        receiver.init_time = None
        receiver.conclude_state = None
        receiver.abort_notified = False
        receiver._transfer_generation = TRANSFER_GENERATION
        receiver._connection_pool_entries = {}
        return receiver, mgr

    def test_returns_existing_conclude_state_without_polling_manager(self):
        receiver, mgr = self._make_receiver()
        receiver.conclude_state = KVPoll.Success

        self.assertEqual(receiver.poll(), KVPoll.Success)
        mgr.check_status.assert_not_called()

    def test_returns_bootstrap_status_before_transfer_starts(self):
        receiver, mgr = self._make_receiver(status=KVPoll.Bootstrapping)

        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        mgr.update_transfer_status.assert_not_called()

    def test_manager_success_or_failed_status_is_terminal(self):
        for terminal_status in (KVPoll.Success, KVPoll.Failed):
            receiver, _ = self._make_receiver(status=terminal_status)

            self.assertEqual(receiver.poll(), terminal_status)
            self.assertEqual(receiver.conclude_state, terminal_status)

    @patch("sglang.srt.disaggregation.nixl.conn.time.time")
    def test_waiting_timeout_records_failure(self, mock_time):
        mock_time.return_value = 20.0
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        mgr.record_failure.assert_called_once()
        self.assertIn("timed out", mgr.record_failure.call_args[0][1])
        mgr.update_status.assert_called_once_with(11, KVPoll.Failed)

    @patch("sglang.srt.disaggregation.nixl.conn.time.time")
    def test_queued_completion_wins_over_waiting_timeout(self, mock_time):
        # Past the deadline, but the completion is already queued/observed:
        # draining before the timeout check must yield Success, not a false
        # timeout, and must not send an abort.
        mock_time.return_value = 20.0
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0
        mgr.transfer_statuses = {
            11: TransferStatus(transfer_generation=TRANSFER_GENERATION)
        }
        mgr.check_transfer_done.return_value = True

        self.assertEqual(receiver.poll(), KVPoll.Success)
        mgr.update_transfer_status.assert_called_once_with()
        mgr.record_failure.assert_not_called()
        mgr.update_status.assert_not_called()
        self.assertNotIn(11, mgr.transfer_statuses)

    @patch("sglang.srt.disaggregation.nixl.conn.time.time")
    def test_transfer_done_returns_success_and_cleans_room_state(self, mock_time):
        mock_time.return_value = 12.0
        receiver, mgr = self._make_receiver(status=KVPoll.WaitingForInput)
        receiver.started_transfer = True
        receiver.init_time = 10.0
        status = TransferStatus(transfer_generation=TRANSFER_GENERATION)
        status.received_aux = True
        status.num_pp_ranks_expected = 1
        status.expected_kvs_per_pp[0] = 0
        mgr.transfer_statuses = {11: status}
        mgr.check_transfer_done.return_value = True

        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertNotIn(11, mgr.transfer_statuses)
        self.assertNotIn(11, mgr.addr_to_rooms_tracker["prefill:8998"])
        self.assertEqual(receiver.conclude_state, KVPoll.Success)


class TestNixlNodeFailure(CustomTestCase):
    def _make_manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr.connection_lock = threading.Lock()
        # Connection keys are "{addr}_{dp_rank}_{cp_rank}_{tp_rank}".
        mgr.connection_pool = {
            "10.0.0.1:8998_0_0_0": [{"rank_ip": "10.0.0.1"}],
            "10.0.0.1:8998_0_0_1": [{"rank_ip": "10.0.0.1"}],
            "10.0.0.2:8998_0_0_0": [{"rank_ip": "10.0.0.2"}],
        }
        mgr.prefill_info_table = {
            "10.0.0.1:8998": object(),
            "10.0.0.2:8998": object(),
        }
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker["10.0.0.1:8998"] = {3, 4, 5}
        mgr.request_status = {
            3: KVPoll.WaitingForInput,
            4: KVPoll.Transferring,
            5: KVPoll.Success,
        }
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.update_status = CommonKVManager.update_status.__get__(mgr, CommonKVManager)
        mgr.check_status = CommonKVManager.check_status.__get__(mgr, CommonKVManager)
        mgr.record_failure = CommonKVManager.record_failure.__get__(
            mgr, CommonKVManager
        )
        return mgr

    def test_handle_node_failure_removes_connections_and_marks_pending_rooms(self):
        mgr = self._make_manager()

        mgr._handle_node_failure("10.0.0.1:8998")

        self.assertNotIn("10.0.0.1:8998_0_0_0", mgr.connection_pool)
        self.assertNotIn("10.0.0.1:8998_0_0_1", mgr.connection_pool)
        self.assertIn("10.0.0.2:8998_0_0_0", mgr.connection_pool)
        self.assertNotIn("10.0.0.1:8998", mgr.prefill_info_table)
        self.assertNotIn("10.0.0.1:8998", mgr.addr_to_rooms_tracker)
        self.assertEqual(mgr.request_status[3], KVPoll.Failed)
        self.assertEqual(mgr.request_status[4], KVPoll.Failed)
        self.assertEqual(mgr.request_status[5], KVPoll.Success)
        self.assertIn(3, mgr.failure_records)
        self.assertIn(4, mgr.failure_records)
        self.assertNotIn(5, mgr.failure_records)

    def test_late_failed_update_does_not_resurrect_cleared_room(self):
        mgr = object.__new__(CommonKVManager)
        mgr.request_status = {}

        CommonKVManager.update_status(mgr, 9, KVPoll.Failed)

        self.assertNotIn(9, mgr.request_status)


class TestNixlStaging(CustomTestCase):
    def _make_manager(self, agent=None):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = agent or StagingFakeAgent()
        mgr.attn_tp_size = 2
        mgr.is_mla_backend = False
        mgr.transfer_source_rank = 1
        mgr.kv_args = SimpleNamespace(
            gpu_id=1,
            engine_rank=1,
            page_size=2,
            total_kv_head_num=2,
            kv_head_num=1,
        )
        mgr.server_args = SimpleNamespace(chunked_prefill_size=4)
        return mgr

    def test_register_buffer_to_engine_groups_kv_memory_kinds_in_one_pass(self):
        agent = StagingFakeAgent(register_result=["desc"])
        mgr = self._make_manager(agent)
        mgr.kv_args.kv_data_ptrs = [0x1000, 0x2000, 0x3000]
        mgr.kv_args.kv_data_lens = [64, 128, 256]
        mgr.kv_args.kv_data_mem_kinds = ["VRAM", "DRAM", "VRAM"]
        mgr.kv_args.aux_data_ptrs = [0x4000]
        mgr.kv_args.aux_data_lens = [32]
        mgr.kv_args.state_data_ptrs = []
        mgr.kv_args.state_data_lens = []

        mgr.register_buffer_to_engine()

        self.assertEqual(
            agent.register_memory_calls,
            [
                (
                    [(0x1000, 64, 1, ""), (0x3000, 256, 1, "")],
                    "VRAM",
                ),
                ([(0x2000, 128, 0, "")], "DRAM"),
                ([(0x4000, 32, 0, "")], "DRAM"),
            ],
        )
        self.assertEqual(mgr.kv_descs, [["desc"], ["desc"]])
        self.assertEqual(mgr.aux_descs, ["desc"])

    def test_register_staging_memory_uses_vram_and_fails_on_empty_descs(self):
        agent = StagingFakeAgent(register_result=["staging"])
        mgr = self._make_manager(agent)

        mgr._register_staging_memory(0x1000, 4096)

        self.assertEqual(
            agent.register_memory_calls,
            [([(0x1000, 4096, 1, "")], "VRAM")],
        )

        mgr = self._make_manager(StagingFakeAgent(register_result=[]))
        with self.assertRaisesRegex(RuntimeError, "staging buffer"):
            mgr._register_staging_memory(0x1000, 4096)

    def test_prefetch_staging_reqs_noops_when_disabled_or_missing_kv_buffers(self):
        mgr = self._make_manager()
        mgr.enable_staging = False
        mgr.kv_buffer_tensors = {"k_buffers": [], "v_buffers": [], "page_size": 1}

        mgr._prefetch_staging_reqs(3)

        mgr.enable_staging = True
        mgr.kv_buffer_tensors = None
        mgr._prefetch_staging_reqs(3)

    def test_prefetch_staging_reqs_marks_room_when_no_peer_needs_staging(self):
        mgr = self._make_manager()
        mgr.enable_staging = True
        mgr.kv_buffer_tensors = {"k_buffers": [], "v_buffers": [], "page_size": 1}
        mgr._staging_ctx = PrefillStagingContext()
        mgr.transfer_infos = {
            3: {
                "agent": TransferInfo(
                    room=3,
                    endpoint="127.0.0.1",
                    dst_port=1000,
                    agent_name="agent",
                    dst_kv_indices=np.array([1], dtype=np.int32),
                    dst_aux_index=0,
                    required_dst_info_num=1,
                    dst_state_indices=[],
                    transfer_generation=TRANSFER_GENERATION,
                )
            }
        }
        mgr.decode_kv_args_table = {
            "agent": SimpleNamespace(decode_tp_size=2),
        }

        mgr._prefetch_staging_reqs(3)

        self.assertIn(3, mgr._staging_ctx.prefetched_rooms)

    def test_do_staging_transfer_requeues_when_allocation_not_ready(self):
        mgr = self._make_manager()
        mgr._staging_ctx = PrefillStagingContext()
        strategy = MagicMock()
        strategy.check_ready.return_value = (False, 0, -1, 0, -1)
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10, 11], dtype=np.int32),
            index_slice=slice(0, 2),
            is_last_chunk=False,
            chunk_id=0,
            prefill_aux_index=None,
            state_indices=None,
        )
        req = SimpleNamespace(
            room=3,
            agent_name="decode_agent",
            completion_notification_prefix=f"nixlv2_{TRANSFER_GENERATION}_3",
        )
        queue = FakeQueue()

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module()
                )
            },
        ):
            handle, deferred = mgr._do_staging_transfer(
                strategy,
                kv_chunk,
                kv_chunk.prefill_kv_indices,
                req,
                SimpleNamespace(),
                queue,
            )

        self.assertIsNone(handle)
        self.assertTrue(deferred)
        self.assertEqual(queue.items, [kv_chunk])

    def test_do_staging_transfer_raises_for_oversized_allocation(self):
        mgr = self._make_manager()
        strategy = MagicMock()
        strategy.check_ready.return_value = (
            False,
            0,
            FakeStagingAllocator.ALLOC_OVERSIZED,
            0,
            -1,
        )
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            chunk_id=0,
            prefill_aux_index=None,
            state_indices=None,
        )

        with self.assertRaisesRegex(RuntimeError, "ring buffer total size"):
            with patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.common.staging_buffer": (
                        _fake_staging_buffer_module()
                    )
                },
            ):
                mgr._do_staging_transfer(
                    strategy,
                    kv_chunk,
                    kv_chunk.prefill_kv_indices,
                    SimpleNamespace(
                        room=3,
                        agent_name="decode_agent",
                        completion_notification_prefix=(
                            f"nixlv2_{TRANSFER_GENERATION}_3"
                        ),
                    ),
                    SimpleNamespace(),
                    FakeQueue(),
                )

    def test_do_staging_transfer_builds_staging_notification(self):
        mgr = self._make_manager()
        strategy = MagicMock()
        strategy.check_ready.return_value = (True, 2, 128, 0, 512)
        strategy.staging_buffer = FakeStagingBuffer()
        kv_chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=np.array([10, 11], dtype=np.int32),
            index_slice=slice(4, 6),
            is_last_chunk=True,
            chunk_id=7,
            prefill_aux_index=0,
            state_indices=None,
        )
        dst_info = KVArgsRegisterInfo(
            room="None",
            endpoint="127.0.0.1",
            dst_port=1000,
            agent_name="decode_agent",
            agent_metadata=b"",
            dst_kv_ptrs=[],
            dst_kv_mem_kinds=[],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[],
            gpu_id=5,
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_kv_item_len=128,
            dst_kv_item_lens=[],
            staging_base_ptr=0x8000,
            staging_total_size=4096,
        )
        calls = []
        mgr.send_kvcache_staged = lambda *args, **kwargs: (
            calls.append((args, kwargs)) or "handle"
        )

        handle, deferred = mgr._do_staging_transfer(
            strategy,
            kv_chunk,
            kv_chunk.prefill_kv_indices,
            SimpleNamespace(
                room=3,
                agent_name="decode_agent",
                completion_notification_prefix=f"nixlv2_{TRANSFER_GENERATION}_3",
            ),
            dst_info,
            FakeQueue(),
        )

        self.assertEqual(handle, "handle")
        self.assertFalse(deferred)
        self.assertEqual(
            calls[0][0][8],
            completion_notification(3, "stg_7_1_1_2_4_2_decode_agent"),
        )

    def test_send_kvcache_staged_uses_one_bulk_vram_write(self):
        mock_gather = MagicMock()
        agent = StagingFakeAgent()
        mgr = self._make_manager(agent)
        mgr.kv_buffer_tensors = {
            "k_buffers": [FakeTensor(), FakeTensor()],
            "v_buffers": [FakeTensor(), FakeTensor()],
            "page_size": 2,
        }

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module(mock_gather)
                )
            },
        ):
            handle = mgr.send_kvcache_staged(
                "peer",
                np.array([1, 2], dtype=np.int32),
                dst_staging_ptr=0x100000,
                dst_staging_size=1 << 20,
                dst_gpu_id=4,
                dst_tp_rank=0,
                dst_attn_tp_size=1,
                dst_kv_item_len=128,
                notif=completion_notification(3, "stg_0_1_1_0_0_2_decode_agent"),
                staging_buffer=FakeStagingBuffer(ptr=0x9000, size=1 << 20),
            )

        self.assertEqual(handle, "handle")
        mock_gather.assert_called_once()
        src_reqs, src_mem = agent.get_xfer_descs_calls[0]
        dst_reqs, dst_mem = agent.get_xfer_descs_calls[1]
        self.assertEqual(src_mem, "VRAM")
        self.assertEqual(dst_mem, "VRAM")
        self.assertEqual(src_reqs.shape, (1, 3))
        self.assertEqual(dst_reqs.shape, (1, 3))
        self.assertTrue(np.issubdtype(src_reqs.dtype, np.integer))
        self.assertTrue(np.issubdtype(dst_reqs.dtype, np.integer))
        self.assertEqual(int(src_reqs[0, 0]), 0x9000)
        self.assertGreaterEqual(int(dst_reqs[0, 0]), 0x100000)
        self.assertEqual(agent.initialize_xfer_calls[0][0], "WRITE")
        self.assertEqual(
            agent.initialize_xfer_calls[0][-1],
            completion_notification(3, "stg_0_1_1_0_0_2_decode_agent").encode("ascii"),
        )

    def test_send_kvcache_staged_falls_back_when_prefill_buffer_too_small(self):
        mgr = self._make_manager()
        mgr.kv_buffer_tensors = {
            "k_buffers": [FakeTensor(), FakeTensor()],
            "v_buffers": [FakeTensor(), FakeTensor()],
            "page_size": 2,
        }

        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_buffer": (
                    _fake_staging_buffer_module()
                )
            },
        ):
            handle = mgr.send_kvcache_staged(
                "peer",
                np.array([1, 2], dtype=np.int32),
                dst_staging_ptr=0xA000,
                dst_staging_size=1 << 20,
                dst_gpu_id=4,
                dst_tp_rank=0,
                dst_attn_tp_size=1,
                dst_kv_item_len=128,
                notif="notif",
                staging_buffer=FakeStagingBuffer(size=1),
            )

        self.assertIsNone(handle)


class DlistCaptureAgent:
    """Records prep_xfer_dlist descriptor arrays so tests can inspect them."""

    def __init__(self):
        self.calls = []  # (peer_name, np.ndarray, mem_kind)

    def prep_xfer_dlist(self, peer_name, array, mem_kind):
        self.calls.append((peer_name, np.asarray(array), mem_kind))
        return f"handle_{len(self.calls)}"


class TestNixlTorchTransferCompactDlist(CustomTestCase):
    def test_equal_tp_dlist_uses_one_strided_region_per_allocation(self):
        mgr = object.__new__(NixlKVManager)
        mgr._use_torch_transfer = True
        mgr.agent = DlistCaptureAgent()

        handle = mgr._prep_equal_tp_dlist(
            "decode-agent",
            kv_ptrs=[0x10000, 0x20000],
            kv_item_lens=[64, 128],
            kv_data_lens=[640, 1280],
            gpu_id=3,
        )

        self.assertEqual(handle, "handle_1")
        peer_name, descriptors, memory_type = mgr.agent.calls[0]
        self.assertEqual(peer_name, "decode-agent")
        self.assertEqual(memory_type, "VRAM")
        np.testing.assert_array_equal(
            descriptors,
            np.asarray(
                [
                    [0x10000, 64, 3, 64, 10],
                    [0x20000, 128, 3, 128, 10],
                ],
                dtype=np.uint64,
            ),
        )
        np.testing.assert_array_equal(
            repeat_indices_over_layers(
                np.asarray([1, 9], dtype=np.int32),
                num_layers=2,
                layer_length=10,
            ),
            np.asarray([1, 9, 11, 19], dtype=np.int32),
        )


class TestNixlHeteroTpReplicatedKV(CustomTestCase):
    """Regression guard for #31295.

    Prefill attention-TP1 -> decode TP4 on a model with only 2 KV heads forces
    GQA replication: decode ranks 0,1 share KV head 0 and ranks 2,3 share KV
    head 1. The shared source dlist must interleave one group per *unique*
    source head-slice (2), and each peer's head_group_idx must map replicated
    decode ranks via integer division (0,0,1,1). The pre-fix code used
    ``num_groups = decode_tp // prefill_tp`` (=4) -- addressing 2x past the
    registered source region, which NIXL rejects with NIXL_ERR_NOT_FOUND -- and
    a modulo head map (0,1,0,1).
    """

    TOTAL_KV_HEADS = 2
    DECODE_TP = 4
    PAGE_SIZE = 1
    BYTES_PER_HEAD = 128  # per token, per head slice
    SRC_KV_ITEM_LEN = TOTAL_KV_HEADS * BYTES_PER_HEAD  # both heads on one prefill rank
    DST_KV_ITEM_LEN = BYTES_PER_HEAD  # one replicated head per decode rank
    NUM_SLOTS = 4
    SRC_PTRS = [0x10000, 0x20000]  # K, V for the single local layer
    REGION_LEN = NUM_SLOTS * SRC_KV_ITEM_LEN

    def _make_manager(self):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = DlistCaptureAgent()
        mgr.attn_tp_size = 1  # prefill attention TP = 1 (DP attention)
        mgr.prep_handle_slice_src = None
        mgr.prep_handles_slice_dst = {}
        mgr.kv_args = SimpleNamespace(
            gpu_id=0,
            engine_rank=0,
            page_size=self.PAGE_SIZE,
            prefill_start_layer=0,
            total_kv_head_num=self.TOTAL_KV_HEADS,
            kv_head_num=self.TOTAL_KV_HEADS,
            kv_item_lens=[self.SRC_KV_ITEM_LEN, self.SRC_KV_ITEM_LEN],
            kv_data_ptrs=list(self.SRC_PTRS),
            kv_data_lens=[self.REGION_LEN, self.REGION_LEN],
        )
        return mgr

    def _decode_args(self, decode_tp_rank):
        return SimpleNamespace(
            agent_name=f"decode_{decode_tp_rank}",
            decode_tp_size=self.DECODE_TP,
            decode_tp_rank=decode_tp_rank,
            dst_kv_item_len=self.DST_KV_ITEM_LEN,
            dst_kv_ptrs=[0x30000, 0x40000],
            dst_num_slots=self.NUM_SLOTS,
            gpu_id=0,
        )

    def test_src_dlist_stays_within_registered_region_and_num_groups(self):
        # Src dlist is built once (shared across peers) on the first call.
        mgr = self._make_manager()
        mgr._init_hetero_tp_prep_handle(
            peer_name="decode_0", decode_kv_args=self._decode_args(0)
        )

        # num_groups must be 2 (one per unique KV head), not decode_tp//prefill_tp=4.
        src_handle, num_groups, _num_ptr_pairs, _num_slots = mgr.prep_handle_slice_src
        self.assertEqual(num_groups, 2)

        # Every source descriptor [addr, addr+len) must lie inside a registered
        # base region [ptr, ptr+REGION_LEN). Pre-fix, num_groups=4 pushed the
        # top group's addresses past the region -> NIXL_ERR_NOT_FOUND.
        src_call = next(c for c in mgr.agent.calls if c[0] == "")
        src_array = src_call[1]
        regions = [(p, p + self.REGION_LEN) for p in self.SRC_PTRS]
        for addr, length, _dev in src_array:
            addr = int(addr)
            length = int(length)
            self.assertTrue(
                any(lo <= addr and addr + length <= hi for lo, hi in regions),
                f"descriptor [{addr:#x}, {addr + length:#x}) escapes all "
                f"registered source regions {[(hex(lo), hex(hi)) for lo, hi in regions]}",
            )

    def test_head_group_idx_maps_replicated_ranks_by_integer_division(self):
        # Each decode rank's per-peer dst handle records its head_group_idx.
        # Expected replicated-KV mapping: ranks 0,1 -> group 0; ranks 2,3 -> group 1.
        expected = {0: 0, 1: 0, 2: 1, 3: 1}
        for rank in range(self.DECODE_TP):
            mgr = self._make_manager()
            mgr._init_hetero_tp_prep_handle(
                peer_name=f"decode_{rank}", decode_kv_args=self._decode_args(rank)
            )
            _dst_handle, _num_slots_dst, head_group_idx = mgr.prep_handles_slice_dst[
                f"decode_{rank}"
            ]
            self.assertEqual(
                head_group_idx,
                expected[rank],
                f"decode rank {rank} mapped to group {head_group_idx}, "
                f"expected {expected[rank]} (modulo bug gives 0,1,0,1)",
            )


if __name__ == "__main__":
    unittest.main()
