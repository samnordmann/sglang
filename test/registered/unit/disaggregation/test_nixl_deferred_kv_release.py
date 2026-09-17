"""Deferred decode-side KV release on the NIXL backend.

When a decode request is aborted while its prefill->decode transfer may still be
in flight, the decode holds its KV pages until every prefill rank acks that its
transfer drained. NIXL transfers are asynchronous (agent.transfer() posts, the
worker polls check_xfer_state), so the ack must come from the transfer worker
after its DONE barrier -- never from the bootstrap thread for an active room.
"""

import threading
import unittest
from unittest.mock import MagicMock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.nixl.conn import GUARD, NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

GEN_A = "11" * 16
GEN_B = "22" * 16


def _prefill_mgr(cls=CommonKVManager, enabled=True):
    """Bare manager carrying only the prefill-side deferred-ack state."""
    mgr = cls.__new__(cls)
    mgr.enable_deferred_decode_kv_release = enabled
    mgr._abort_state_lock = threading.Lock()
    mgr._deferred_ack_targets = {}
    mgr._active_transfer_generations = {}
    mgr._abort_ack_receipts = set()
    mgr._staging_outstanding = {}
    mgr.request_status = {}
    mgr._sent = []
    # Capture acks instead of opening a socket.
    mgr._send_abort_ack = lambda ip, port, room, generation: (
        mgr._sent.append((ip, port, room, generation)) or True
    )
    return mgr


def _activate_failed(mgr, room, generation=GEN_A):
    mgr.register_transfer_generation(room, generation)
    mgr.request_status[room] = KVPoll.Failed
    return mgr.register_deferred_ack_target(room, generation, "10.0.0.1", 5000)


class TestDeferredAckTargets(CustomTestCase):
    def test_ack_held_until_outstanding_drains(self):
        mgr = _prefill_mgr()
        self.assertTrue(_activate_failed(mgr, 7))

        mgr._staging_outstanding[7] = 1
        mgr._maybe_ack_drained_abort(7, GEN_A)
        self.assertEqual(mgr._sent, [])  # still writing -> no ack

        mgr._staging_outstanding[7] = 0
        mgr._maybe_ack_drained_abort(7, GEN_A)
        self.assertEqual(mgr._sent, [("10.0.0.1", 5000, 7, GEN_A)])

    def test_one_skipped_writer_cannot_erase_an_active_sibling(self):
        mgr = _prefill_mgr()
        self.assertTrue(_activate_failed(mgr, 70))
        self.assertEqual(mgr._count_transfer_writer(70), 1)
        self.assertEqual(mgr._count_transfer_writer(70), 2)

        self.assertEqual(mgr._uncount_transfer_writer(70), 1)
        mgr._maybe_ack_drained_abort(70, GEN_A)
        self.assertEqual(mgr._sent, [])
        self.assertFalse(mgr.mark_transfer_quiescent(70, GEN_A))
        self.assertEqual(mgr.active_transfer_generation(70), GEN_A)

        self.assertEqual(mgr._uncount_transfer_writer(70), 0)
        self.assertTrue(mgr.mark_transfer_quiescent(70, GEN_A))
        self.assertEqual(mgr._sent, [("10.0.0.1", 5000, 70, GEN_A)])
        self.assertIsNone(mgr.active_transfer_generation(70))

    def test_ack_fires_at_most_once(self):
        mgr = _prefill_mgr()
        mgr.register_transfer_generation(8, GEN_A)
        mgr.request_status[8] = KVPoll.Failed
        mgr.register_deferred_ack_target(8, GEN_A, "10.0.0.2", 5001)
        mgr._maybe_ack_drained_abort(8, GEN_A)
        mgr._maybe_ack_drained_abort(8, GEN_A)
        self.assertEqual(len(mgr._sent), 1)
        self.assertNotIn((8, GEN_A), mgr._deferred_ack_targets)

    def test_unregistered_room_is_noop(self):
        mgr = _prefill_mgr()
        mgr._maybe_ack_drained_abort(999, GEN_A)
        self.assertEqual(mgr._sent, [])

    def test_normal_send_failure_retains_target_and_receipt_for_retry(self):
        mgr = _prefill_mgr()
        self.assertTrue(_activate_failed(mgr, 9))
        mgr._send_abort_ack = MagicMock(side_effect=[False, True])

        mgr._maybe_ack_drained_abort(9, GEN_A)
        self.assertIn((9, GEN_A), mgr._deferred_ack_targets)
        self.assertIn((9, GEN_A), mgr._abort_ack_receipts)
        mgr._maybe_ack_drained_abort(9, GEN_A)
        self.assertNotIn((9, GEN_A), mgr._deferred_ack_targets)
        self.assertEqual(mgr._send_abort_ack.call_count, 2)

    def test_base_exception_after_acceptance_retains_durable_retry_owner(self):
        mgr = _prefill_mgr()
        self.assertTrue(_activate_failed(mgr, 10))
        accepted = []

        def accepted_then_interrupted(ip, port, room, generation):
            accepted.append((ip, port, room, generation))
            raise KeyboardInterrupt("lost return")

        mgr._send_abort_ack = accepted_then_interrupted
        with self.assertRaises(KeyboardInterrupt):
            mgr._maybe_ack_drained_abort(10, GEN_A)
        self.assertIn((10, GEN_A), mgr._deferred_ack_targets)
        self.assertIn((10, GEN_A), mgr._abort_ack_receipts)

        mgr._send_abort_ack = lambda ip, port, room, generation: (
            accepted.append((ip, port, room, generation)) or True
        )
        mgr._maybe_ack_drained_abort(10, GEN_A)
        self.assertNotIn((10, GEN_A), mgr._deferred_ack_targets)
        self.assertEqual(len(accepted), 2)

    def test_receipt_tombstones_are_not_evicted(self):
        mgr = _prefill_mgr()
        for room in range(2048):
            mgr._remember_abort_receipt_unlocked((room, GEN_A))
        self.assertEqual(len(mgr._abort_ack_receipts), 2048)
        self.assertIn((0, GEN_A), mgr._abort_ack_receipts)

    def test_prefill_unique_rank_matches_success_sync_formula(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.attn_tp_rank, mgr.pp_size, mgr.attn_cp_size = 2, 3, 4
        mgr.pp_rank, mgr.attn_cp_rank = 1, 3
        self.assertEqual(mgr._prefill_unique_rank(), 2 * (3 * 4) + 1 * 4 + 3)

    def test_ack_target_is_immutable_for_one_generation(self):
        mgr = _prefill_mgr()
        mgr.register_transfer_generation(12, GEN_A)
        self.assertTrue(mgr.register_deferred_ack_target(12, GEN_A, "10.0.0.1", 5000))
        self.assertFalse(mgr.register_deferred_ack_target(12, GEN_A, "10.0.0.2", 5001))
        target = mgr._deferred_ack_targets[(12, GEN_A)]
        self.assertEqual((target.decode_ip, target.decode_port), ("10.0.0.1", 5000))


class TestNixlAbortNotification(CustomTestCase):
    """_handle_abort_notification is the prefill bootstrap-thread entry point."""

    @staticmethod
    def _abort_msg(room=11, ip="10.0.0.3", port=6000, generation=GEN_A):
        return [
            b"ABORT",
            str(room).encode("ascii"),
            ip.encode("ascii"),
            str(port).encode("ascii"),
            generation.encode("ascii"),
        ]

    def _mgr(self, enabled=True, room=11, status=KVPoll.WaitingForInput):
        mgr = _prefill_mgr(NixlKVManager, enabled=enabled)
        mgr.register_transfer_generation(room, GEN_A)
        if status is not None:
            mgr.request_status[room] = status
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock(
            side_effect=lambda r, s: mgr.request_status.__setitem__(r, s)
        )
        mgr.check_status = lambda r: mgr.request_status[r]
        return mgr

    def test_in_flight_room_registers_target_and_does_not_ack_yet(self):
        # A counted chunk holds the ack: only the worker knows when it landed.
        mgr = self._mgr()
        mgr._staging_outstanding[11] = 1
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        target = mgr._deferred_ack_targets[(11, GEN_A)]
        self.assertEqual((target.decode_ip, target.decode_port), ("10.0.0.3", 6000))
        self.assertEqual(mgr._sent, [])
        # Marked Failed first, so no new chunk can be enqueued for the room.
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)

    def test_quiescent_active_room_acks_without_waiting_for_a_worker_visit(self):
        # Window 2: chunks already drained with none left to come, so the worker
        # never revisits the room -- acking here keeps it off the timeout path.
        mgr = self._mgr()
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11, GEN_A)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_worker_skip_before_registration_still_acks(self):
        # Window 1: the worker can pass its skip point between the Failed flip
        # and registration; the ack attempt at registration covers that.
        mgr = self._mgr()
        mgr._staging_outstanding[11] = 1

        real_update = mgr.update_status.side_effect

        def failed_then_worker_skips(room, status):
            real_update(room, status)
            # Worker dequeues, sees Failed, uncounts, and finds no target yet.
            mgr._staging_outstanding.pop(room, None)
            mgr.mark_transfer_quiescent(room, GEN_A)

        mgr.update_status = MagicMock(side_effect=failed_then_worker_skips)
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11, GEN_A)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_concluded_room_acks_immediately(self):
        # Concluded and quiescent: ack straight away.
        mgr = self._mgr(status=None)
        mgr.mark_transfer_quiescent(11, GEN_A)
        mgr.check_status = lambda r: KVPoll.Success
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11, GEN_A)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_cleared_room_with_outstanding_chunk_does_not_ack(self):
        # The ERR path abandons sibling handles that may still be writing and
        # leaves the chunk counted; clear() then drops the room. Acking on
        # "unknown room" alone would release decode pages under those writes.
        mgr = self._mgr(status=None)  # room absent == cleared/unknown
        mgr._staging_outstanding[11] = 1
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [])
        self.assertIn((11, GEN_A), mgr._deferred_ack_targets)

    def test_explicit_feature_false_cannot_disable_safe_ack_protocol(self):
        mgr = self._mgr(enabled=False)
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._deferred_ack_targets, {})
        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11, GEN_A)])
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)

    def test_old_generation_abort_cannot_mutate_reused_room(self):
        mgr = self._mgr()
        self.assertTrue(
            mgr._handle_abort_notification(self._abort_msg(generation=GEN_B))
        )
        self.assertEqual(mgr.request_status[11], KVPoll.WaitingForInput)
        self.assertEqual(mgr.active_transfer_generation(11), GEN_A)
        self.assertEqual(mgr._deferred_ack_targets, {})
        self.assertEqual(mgr._sent, [])

    def test_legacy_two_frame_abort_is_tolerated(self):
        # Older peers send [ABORT, room] with no return address.
        mgr = self._mgr()
        self.assertTrue(mgr._handle_abort_notification([b"ABORT", b"11"]))
        self.assertEqual(mgr._deferred_ack_targets, {})
        self.assertEqual(mgr._sent, [])

    def test_non_abort_message_is_not_claimed(self):
        mgr = self._mgr()
        self.assertFalse(mgr._handle_abort_notification([b"STAGING_REQ", b"11"]))


class TestNixlDecodeAckIngest(CustomTestCase):
    def test_abort_ack_is_aggregated_per_rank(self):
        mgr = _prefill_mgr()
        mgr._deferred_abort_ack_tracker = {}
        mgr.register_deferred_abort_room(21, GEN_A, frozenset({0, 1}))

        for rank in (0, 1, 1):
            mgr.note_abort_ack(21, GEN_A, rank)

        self.assertTrue(mgr.is_abort_release_safe(21, GEN_A))

    def test_late_metadata_does_not_poison_new_generation(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._bootstrap_errors = {}
        mgr.exceptions = {}
        mgr.transfer_infos = {21: {}}
        mgr.record_failure = MagicMock()
        mgr.register_transfer_generation(21, GEN_B)
        late_payload = [
            b"21",
            b"endpoint",
            b"1234",
            b"agent",
            b"",
            b"0",
            b"1",
            b"",
            b"",
            b"0",
            GEN_A.encode("ascii"),
        ]

        mgr._handle_bootstrap_message([GUARD, *late_payload])

        self.assertEqual(mgr.active_transfer_generation(21), GEN_B)
        self.assertEqual(mgr.transfer_infos[21], {})
        self.assertNotIn(21, mgr.exceptions)
        mgr.record_failure.assert_not_called()

    def test_malformed_or_generationless_metadata_cannot_poison_reused_room(self):
        for generation_frame in (b"not-a-generation", None):
            mgr = _prefill_mgr(NixlKVManager)
            mgr._bootstrap_errors = {}
            mgr.exceptions = {}
            mgr.transfer_infos = {23: {}}
            mgr.record_failure = MagicMock()
            mgr.register_transfer_generation(23, GEN_B)
            payload = [
                b"23",
                b"endpoint",
                b"1234",
                b"agent",
                b"",
                b"0",
                b"1",
                b"",
                b"",
                b"0",
            ]
            if generation_frame is not None:
                payload.append(generation_frame)

            mgr._handle_bootstrap_message([GUARD, *payload])

            self.assertEqual(mgr.active_transfer_generation(23), GEN_B)
            self.assertEqual(mgr.transfer_infos[23], {})
            self.assertNotIn(23, mgr.exceptions)
            self.assertNotIn("agent", mgr._bootstrap_errors)
            mgr.record_failure.assert_not_called()

    def test_retired_metadata_cannot_resurrect_room_before_reuse(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._bootstrap_errors = {}
        mgr.exceptions = {}
        mgr.transfer_infos = {}
        mgr.record_failure = MagicMock()
        mgr.register_transfer_generation(22, GEN_A)
        mgr.mark_transfer_quiescent(22, GEN_A)
        late_payload = [
            b"22",
            b"endpoint",
            b"1234",
            b"agent",
            b"",
            b"0",
            b"1",
            b"",
            b"",
            b"0",
            GEN_A.encode("ascii"),
        ]

        mgr._handle_bootstrap_message([GUARD, *late_payload])

        self.assertIsNone(mgr.active_transfer_generation(22))
        self.assertNotIn(22, mgr.transfer_infos)
        self.assertIn((22, GEN_A), mgr._abort_ack_receipts)
        self.assertNotIn(22, mgr.exceptions)
        mgr.record_failure.assert_not_called()

    def test_retirement_and_duplicate_registration_are_atomic(self):
        mgr = _prefill_mgr(NixlKVManager)
        rooms = tuple(range(1000, 1256))
        for room in rooms:
            self.assertTrue(mgr.register_transfer_generation(room, GEN_A))

        operation_start = threading.Barrier(2)
        operation_done = threading.Barrier(2)
        replay_results = []

        def retire_generations():
            for room in rooms:
                operation_start.wait()
                mgr.mark_transfer_quiescent(room, GEN_A)
                operation_done.wait()

        def replay_metadata_registration():
            for room in rooms:
                operation_start.wait()
                replay_results.append(mgr.register_transfer_generation(room, GEN_A))
                operation_done.wait()

        retire_thread = threading.Thread(target=retire_generations)
        replay_thread = threading.Thread(target=replay_metadata_registration)
        retire_thread.start()
        replay_thread.start()
        retire_thread.join(timeout=5)
        replay_thread.join(timeout=5)

        self.assertFalse(retire_thread.is_alive())
        self.assertFalse(replay_thread.is_alive())
        self.assertEqual(len(replay_results), len(rooms))
        for room in rooms:
            self.assertIsNone(mgr.active_transfer_generation(room))
            self.assertIn((room, GEN_A), mgr._abort_ack_receipts)


if __name__ == "__main__":
    unittest.main()
