"""Unit tests for the deferred decode-side KV release mechanism.

When a decode request is aborted while its prefill->decode KV transfer may still
be in flight, the decode side holds its KV pages / req-slot instead of freeing
them immediately (which could let the still-in-flight write land on pages already
reused by another request). The pages are released only once every exact prefill
rank for the immutable room generation acks that its transfer drained. A timeout
is diagnostic and never authorizes reuse.
"""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    next_transfer_generation,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_manager():
    """A bare CommonKVManager carrying only the deferred-ack state the helpers
    touch (avoids the heavy real __init__)."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr._deferred_abort_ack_tracker = {}
    mgr._abort_state_lock = threading.Lock()
    return mgr


GEN_A = "11" * 16
GEN_B = "22" * 16


class TestAbortAckAggregation(CustomTestCase):
    def test_release_safe_only_after_all_required_ranks_ack(self):
        mgr = _make_manager()
        room = 100
        mgr.register_deferred_abort_room(room, GEN_A, frozenset({4, 9}))
        self.assertFalse(mgr.is_abort_release_safe(room, GEN_A))

        self.assertTrue(mgr.note_abort_ack(room, GEN_A, 4))
        self.assertFalse(mgr.is_abort_release_safe(room, GEN_A))

        self.assertTrue(mgr.note_abort_ack(room, GEN_A, 9))
        self.assertTrue(mgr.is_abort_release_safe(room, GEN_A))

    def test_duplicate_rank_ack_does_not_over_count(self):
        mgr = _make_manager()
        room = 101
        mgr.register_deferred_abort_room(room, GEN_A, frozenset({0, 1}))
        mgr.note_abort_ack(room, GEN_A, 0)
        mgr.note_abort_ack(room, GEN_A, 0)
        self.assertFalse(mgr.is_abort_release_safe(room, GEN_A))

    def test_invalid_and_unexpected_rank_identities_are_rejected(self):
        mgr = _make_manager()
        room = 102
        mgr.register_deferred_abort_room(room, GEN_A, frozenset({7}))
        for bad_rank in (True, -1, 8, 2**63):
            self.assertFalse(mgr.note_abort_ack(room, GEN_A, bad_rank))
        self.assertFalse(mgr.is_abort_release_safe(room, GEN_A))
        self.assertTrue(mgr.note_abort_ack(room, GEN_A, 7))
        self.assertTrue(mgr.is_abort_release_safe(room, GEN_A))

    def test_clear_deferred_abort_state(self):
        mgr = _make_manager()
        room = 103
        mgr.register_deferred_abort_room(room, GEN_A, frozenset({0}))
        mgr.note_abort_ack(room, GEN_A, 0)
        mgr.clear_deferred_abort_state(room, GEN_A)
        self.assertNotIn((room, GEN_A), mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, GEN_A))

    def test_ack_before_register_is_dropped(self):
        mgr = _make_manager()
        room = 104
        self.assertFalse(mgr.note_abort_ack(room, GEN_A, 0))
        self.assertNotIn((room, GEN_A), mgr._deferred_abort_ack_tracker)

    def test_room_reuse_is_generation_isolated(self):
        mgr = _make_manager()
        room = 105
        expected = frozenset({0, 1})
        mgr.register_deferred_abort_room(room, GEN_A, expected)
        mgr.note_abort_ack(room, GEN_A, 0)
        mgr.clear_deferred_abort_state(room, GEN_A)
        mgr.register_deferred_abort_room(room, GEN_B, expected)

        self.assertFalse(mgr.note_abort_ack(room, GEN_A, 1))
        mgr.note_abort_ack(room, GEN_B, 0)
        self.assertFalse(mgr.is_abort_release_safe(room, GEN_B))
        mgr.note_abort_ack(room, GEN_B, 1)
        self.assertTrue(mgr.is_abort_release_safe(room, GEN_B))

    def test_rearming_is_idempotent_but_expected_set_is_immutable(self):
        mgr = _make_manager()
        room = 106
        expected = frozenset({0, 1})
        mgr.register_deferred_abort_room(room, GEN_A, expected)
        mgr.note_abort_ack(room, GEN_A, 0)
        mgr.register_deferred_abort_room(room, GEN_A, expected)
        self.assertFalse(mgr.is_abort_release_safe(room, GEN_A))
        with self.assertRaises(RuntimeError):
            mgr.register_deferred_abort_room(room, GEN_A, frozenset({0}))

    def test_expected_rank_set_cannot_be_empty_or_malformed(self):
        mgr = _make_manager()
        for bad in (frozenset(), frozenset({-1}), frozenset({True})):
            with self.assertRaises(ValueError):
                mgr.register_deferred_abort_room(107, GEN_A, bad)


class _FakeIdxAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def _make_queue(timeout=30.0):
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q._deferred_releases = []
    q.deferred_kv_release_timeout = timeout
    q.enable_staging = False
    q.staging_handler = None
    q.tree_cache = object()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={})
    q.req_to_metadata_buffer_idx_allocator = _FakeIdxAllocator()
    return q


def _make_decode_req(room, idx, mgr, ranks=(0,), generation=GEN_A):
    receiver = SimpleNamespace(
        kv_mgr=mgr,
        bootstrap_infos=[{"prefill_unique_rank": rank} for rank in ranks],
        transfer_generation=generation,
        abort_notified=False,
        clear=MagicMock(),
        retry_abort_notification=MagicMock(),
        failure_exception=MagicMock(side_effect=RuntimeError("transfer failed")),
    )

    def arm_abort_intent():
        if receiver.abort_notified:
            return
        mgr.register_deferred_abort_room(room, generation, frozenset(ranks))
        receiver.abort_notified = True

    receiver.arm_abort_intent = arm_abort_intent
    return SimpleNamespace(
        req=SimpleNamespace(
            bootstrap_room=room,
            rid=f"request-{room}",
            return_logprob=False,
        ),
        kv_receiver=receiver,
        metadata_buffer_index=idx,
    )


class TestResolveDeferredReleases(CustomTestCase):
    def test_noop_when_nothing_deferred(self):
        q = _make_queue()
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_not_called()

    def test_holds_until_drained_then_releases(self):
        mgr = _make_manager()
        room, idx = 200, 7
        q = _make_queue()
        dreq = _make_decode_req(room, idx, mgr, ranks=(4, 9))
        q._defer_release(dreq)

        with patch.object(decode_mod, "release_kv_cache") as rel:
            # Not yet acked -> held, not released.
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # One of two ranks acked -> still held.
            mgr.note_abort_ack(room, GEN_A, 4)
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # Both ranks acked -> released exactly once.
            mgr.note_abort_ack(room, GEN_A, 9)
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

        # Held state fully cleaned up.
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertEqual(q.metadata_buffers.bootstrap_room[idx], 0)
        self.assertNotIn((room, GEN_A), mgr._deferred_abort_ack_tracker)
        self.assertIsNone(dreq.kv_receiver)

    def test_timeout_only_diagnoses_retries_and_retains_quarantine(self):
        mgr = _make_manager()
        room, idx = 300, 3
        q = _make_queue(timeout=30.0)
        dreq = _make_decode_req(room, idx, mgr)
        q._defer_release(dreq)
        q._deferred_releases[0].next_diagnostic_at = float("-inf")

        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
            rel.assert_not_called()

        self.assertEqual(len(q._deferred_releases), 1)
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [])
        self.assertIs(dreq.kv_receiver, q._deferred_releases[0].kv_receiver)
        dreq.kv_receiver.retry_abort_notification.assert_called_once_with()
        self.assertEqual(q._deferred_releases[0].diagnostic_count, 1)

    def test_failed_release_is_fail_stop_while_other_owner_completes(self):
        mgr = _make_manager()
        q = _make_queue()
        bad = _make_decode_req(701, 2, mgr, generation=GEN_A)
        good = _make_decode_req(700, 1, mgr, generation=GEN_B)
        q._defer_release(bad)
        q._defer_release(good)
        mgr.note_abort_ack(701, GEN_A, 0)
        mgr.note_abort_ack(700, GEN_B, 0)

        calls = []

        def fake_release(req, tree_cache, is_insert):
            calls.append(req)
            if req is bad.req:
                raise RuntimeError("boom")

        with patch.object(decode_mod, "release_kv_cache", side_effect=fake_release):
            q.resolve_deferred_releases()
            self.assertIn(good.req, calls)
            self.assertEqual(len(q._deferred_releases), 1)
            failed = q._deferred_releases[0]
            self.assertIs(failed.decode_req, bad)
            self.assertEqual(failed.release_phase, "KV_RELEASE_ENTERED")
            self.assertIsNotNone(failed.release_error)
            prior_calls = list(calls)
            q.resolve_deferred_releases()
            self.assertEqual(calls, prior_calls)

    def test_lost_return_from_non_idempotent_free_is_never_retried(self):
        mgr = _make_manager()
        q = _make_queue()
        dreq = _make_decode_req(702, 5, mgr)
        q._defer_release(dreq)
        mgr.note_abort_ack(702, GEN_A, 0)

        real_free = q.req_to_metadata_buffer_idx_allocator.free

        def committed_then_interrupted(idx):
            real_free(idx)
            raise KeyboardInterrupt("lost return")

        q.req_to_metadata_buffer_idx_allocator.free = MagicMock(
            side_effect=committed_then_interrupted
        )
        with self.assertRaises(KeyboardInterrupt):
            q.resolve_deferred_releases()
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [5])
        self.assertEqual(
            q._deferred_releases[0].release_phase,
            "METADATA_INDEX_FREE_ENTERED",
        )
        q.resolve_deferred_releases()
        q.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(5)

    def test_explicit_false_environment_cannot_authorize_timeout_release(self):
        mgr = _make_manager()
        with patch.dict(
            os.environ,
            {"SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE": "false"},
        ):
            q = DecodeTransferQueue(
                gloo_group=None,
                req_to_metadata_buffer_idx_allocator=_FakeIdxAllocator(),
                tp_rank=0,
                metadata_buffers=SimpleNamespace(bootstrap_room={}),
                scheduler=SimpleNamespace(spec_algorithm=None),
                tree_cache=object(),
            )
        self.assertTrue(q.enable_deferred_kv_release)
        dreq = _make_decode_req(703, 6, mgr)
        q._defer_release(dreq)
        q._deferred_releases[0].next_diagnostic_at = float("-inf")
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_not_called()
        self.assertEqual(len(q._deferred_releases), 1)

    def test_false_compatibility_fields_cannot_bypass_failure_quarantine(self):
        mgr = _make_manager()
        mgr.enable_deferred_decode_kv_release = False
        q = _make_queue()
        q.enable_deferred_kv_release = False
        q.enable_staging = False
        q.gloo_group = None
        q.tp_rank = 0
        q.queue = [_make_decode_req(704, 8, mgr)]
        q.queue[0].hicache_restore_status = None
        # This is specifically a decode-initiated failure with a possible
        # remote writer. A propagated prefill failure has already stopped its
        # writer and intentionally uses the immediate-safe release path.
        q.queue[0].kv_receiver.arm_abort_intent()
        q._poll_with_metadata_gate = MagicMock(return_value=[decode_mod.KVPoll.Failed])
        q._clean_hicache_prefetch_resources = MagicMock()
        q.scheduler = SimpleNamespace(
            enable_decode_hicache=False,
            enable_hisparse=False,
            output_streamer=SimpleNamespace(stream_output=MagicMock()),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )

        with patch.object(decode_mod, "release_kv_cache") as release:
            q.pop_transferred()

        release.assert_not_called()
        self.assertEqual(q.queue, [])
        self.assertEqual(len(q._deferred_releases), 1)
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [])
        self.assertIsNotNone(q._deferred_releases[0].decode_req.kv_receiver)

    def test_defer_release_records_deadline_and_idx(self):
        mgr = _make_manager()
        q = _make_queue(timeout=12.5)
        dreq = _make_decode_req(room=400, idx=9, mgr=mgr)
        q._defer_release(dreq)
        self.assertEqual(len(q._deferred_releases), 1)
        held = q._deferred_releases[0]
        self.assertIs(held.decode_req, dreq)
        self.assertEqual(held.metadata_buffer_index, 9)
        self.assertEqual(held.transfer_generation, GEN_A)
        self.assertIsInstance(held.next_diagnostic_at, float)


class TestTransferGenerationPlumbing(CustomTestCase):
    @staticmethod
    def _make_prealloc_queue():
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.transfer_backend = TransferBackend.NIXL
        queue.scheduler = SimpleNamespace(
            output_streamer=SimpleNamespace(stream_output=MagicMock())
        )
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
        queue._resolve_prefill_dp_rank = MagicMock(return_value=0)
        receiver = SimpleNamespace(init=MagicMock())
        queue._create_receiver_and_enqueue = MagicMock(
            return_value=SimpleNamespace(kv_receiver=receiver)
        )
        return queue, receiver

    def test_rebootstrap_advances_identically_on_each_decode_rank(self):
        advanced = []
        for _ in range(2):
            queue, receiver = self._make_prealloc_queue()
            req = SimpleNamespace(
                bootstrap_host="prefill.internal",
                transfer_generation=GEN_A,
                return_logprob=False,
            )

            queue.add(req, is_rebootstrap=True)

            advanced.append(req.transfer_generation)
            queue._create_receiver_and_enqueue.assert_called_once_with(
                req, is_rebootstrap=True
            )
            receiver.init.assert_called_once_with(0)

        self.assertEqual(advanced, [next_transfer_generation(GEN_A)] * 2)
        self.assertNotIn(advanced[0], (GEN_A, GEN_B))
        self.assertEqual(len(advanced[0]), 32)

    def test_missing_generation_fails_request_before_receiver_construction(self):
        queue, _ = self._make_prealloc_queue()
        req = SimpleNamespace(
            bootstrap_host="prefill.internal",
            transfer_generation=None,
            return_logprob=False,
        )

        with patch.object(decode_mod, "prepare_abort") as prepare_abort:
            queue.add(req)

        queue._create_receiver_and_enqueue.assert_not_called()
        prepare_abort.assert_called_once()
        self.assertEqual(
            prepare_abort.call_args.kwargs["status_code"],
            decode_mod.HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], False
        )


if __name__ == "__main__":
    unittest.main()
