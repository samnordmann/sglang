"""Adversarial Mooncake control-plane lease identity tests."""

import struct
import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler
from sglang.srt.disaggregation.mooncake.conn import (
    MOONCAKE_AUX_DATA_WIRE_TAG,
    MOONCAKE_CHUNK_READY_WIRE_TAG,
    MOONCAKE_TRANSFER_STATUS_WIRE_TAG,
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

GEN_A = "a1" * 16
GEN_B = "b2" * 16
ROOM = 7
SESSION = "decode-session"


def _manager(generation=GEN_B, expected=frozenset({3, 5})):
    manager = object.__new__(MooncakeKVManager)
    manager._abort_state_lock = threading.Lock()
    manager._receive_transfer_leases = {}
    manager.prefill_response_tracker = defaultdict(set)
    manager.request_status = {ROOM: KVPoll.WaitingForInput}
    manager.required_prefill_response_num_table = {ROOM: len(expected)}
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.enable_staging = True
    manager._staging_handler = MagicMock()
    manager.kv_args = SimpleNamespace()
    manager._register_receive_transfer_lease(ROOM, generation, SESSION, expected)
    return manager


def _chunk(generation, rank=3, session=SESSION):
    return [
        MOONCAKE_CHUNK_READY_WIRE_TAG,
        generation.encode("ascii"),
        str(ROOM).encode("ascii"),
        b"0",
        b"4",
        b"2",
        session.encode("ascii"),
        str(rank).encode("ascii"),
    ]


def _aux(generation, rank=3, payload=b"data"):
    return [
        MOONCAKE_AUX_DATA_WIRE_TAG,
        generation.encode("ascii"),
        str(ROOM).encode("ascii"),
        str(rank).encode("ascii"),
        b"0",
        b"1",
        struct.pack(">I", len(payload)),
        payload,
    ]


def _status(generation, rank, status=KVPoll.Success):
    return [
        MOONCAKE_TRANSFER_STATUS_WIRE_TAG,
        generation.encode("ascii"),
        str(ROOM).encode("ascii"),
        str(status).encode("ascii"),
        str(rank).encode("ascii"),
    ]


class TestMooncakeGenerationControl(CustomTestCase):
    def test_receiver_init_binds_immutable_exact_rank_set(self):
        manager = _manager()
        manager._clear_receive_transfer_lease(ROOM, GEN_B)
        manager.addr_to_rooms_tracker = defaultdict(set)
        manager.get_session_id = MagicMock(return_value=SESSION)

        receiver = MooncakeKVReceiver(
            manager,
            "prefill:8998",
            ROOM,
            transfer_generation=GEN_B,
        )
        receiver.bootstrap_infos = [
            {"prefill_unique_rank": 3, "is_dummy": False},
            {"prefill_unique_rank": 4, "is_dummy": True},
            {"prefill_unique_rank": 5, "is_dummy": False},
        ]
        with patch.object(CommonKVReceiver, "init", return_value=None):
            receiver.init(prefill_dp_rank=0)

        lease = manager._receive_transfer_leases[ROOM]
        self.assertEqual(lease.generation, GEN_B)
        self.assertEqual(lease.session_id, SESSION)
        self.assertEqual(lease.expected_prefill_ranks, frozenset({3, 5}))
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            manager._register_receive_transfer_lease(
                ROOM, GEN_B, SESSION, frozenset({3, 6})
            )

    def test_old_chunk_cannot_scatter_reused_room(self):
        manager = _manager()

        self.assertFalse(manager._handle_chunk_ready(_chunk(GEN_A)))
        self.assertFalse(manager._handle_chunk_ready(_chunk(GEN_B, rank=99)))
        self.assertFalse(manager._handle_chunk_ready(_chunk(GEN_B, session="old")))
        self.assertFalse(
            manager._handle_chunk_ready(
                [b"CHUNK_READY", b"7", b"0", b"4", b"2", SESSION.encode()]
            )
        )
        manager._staging_handler.handle_chunk_arrived.assert_not_called()

        self.assertTrue(manager._handle_chunk_ready(_chunk(GEN_B, rank=3)))
        manager._staging_handler.handle_chunk_arrived.assert_called_once_with(
            ROOM, 0, 4, 2, 3
        )

    def test_old_aux_cannot_write_reused_room(self):
        manager = _manager()
        target = "sglang.srt.disaggregation.mooncake.conn.AuxDataCodec.deserialize_data_to_buffer"
        with patch(target) as deserialize:
            self.assertFalse(manager._handle_aux_data(_aux(GEN_A)))
            self.assertFalse(manager._handle_aux_data(_aux(GEN_B, rank=99)))
            self.assertFalse(
                manager._handle_aux_data(
                    [b"AUX_DATA", b"7", b"0", b"1", struct.pack(">I", 4), b"data"]
                )
            )
            deserialize.assert_not_called()

            self.assertTrue(manager._handle_aux_data(_aux(GEN_B, rank=3)))
            deserialize.assert_called_once_with(manager.kv_args, 0, 1, b"data")

    def test_status_requires_exact_generation_and_all_distinct_expected_ranks(self):
        manager = _manager()

        self.assertFalse(manager._handle_transfer_status(_status(GEN_A, 3)))
        self.assertFalse(manager._handle_transfer_status(_status(GEN_B, 99)))
        self.assertFalse(manager._handle_transfer_status([b"7", b"4", b"3"]))
        self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)

        self.assertTrue(manager._handle_transfer_status(_status(GEN_B, 3)))
        self.assertTrue(manager._handle_transfer_status(_status(GEN_B, 3)))
        self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)
        self.assertEqual(manager.prefill_response_tracker[ROOM], {3})

        self.assertTrue(manager._handle_transfer_status(_status(GEN_B, 5)))
        self.assertEqual(manager.prefill_response_tracker[ROOM], {3, 5})
        self.assertEqual(manager.request_status[ROOM], KVPoll.Success)

    def test_delayed_old_receiver_clear_cannot_erase_new_lease(self):
        manager = _manager(generation=GEN_A)
        self.assertTrue(manager._clear_receive_transfer_lease(ROOM, GEN_A))
        manager.request_status[ROOM] = KVPoll.WaitingForInput
        manager.required_prefill_response_num_table[ROOM] = 2
        # The old receiver has already cleared its identity. A duplicate clear
        # must not treat that absence as authority over state B is publishing.
        self.assertFalse(manager._clear_receive_transfer_lease(ROOM, GEN_A))
        self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)
        manager._register_receive_transfer_lease(
            ROOM, GEN_B, SESSION, frozenset({3, 5})
        )

        self.assertFalse(manager._clear_receive_transfer_lease(ROOM, GEN_A))
        self.assertEqual(manager._receive_transfer_leases[ROOM].generation, GEN_B)
        self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)

    def test_staging_fan_in_deduplicates_exact_writer_rank(self):
        handler = object.__new__(DecodeStagingHandler)
        handler._room_to_receiver = {ROOM: object()}
        handler._writer_counts = {}
        handler.num_writers_for = MagicMock(return_value=2)
        handler.submit_chunk_scatter = MagicMock(return_value=True)

        self.assertFalse(handler.handle_chunk_arrived(ROOM, 0, 4, 2, 3))
        self.assertFalse(handler.handle_chunk_arrived(ROOM, 0, 4, 2, 3))
        handler.submit_chunk_scatter.assert_not_called()

        self.assertTrue(handler.handle_chunk_arrived(ROOM, 0, 4, 2, 5))
        handler.submit_chunk_scatter.assert_called_once_with(ROOM, 0, 4, 2)
