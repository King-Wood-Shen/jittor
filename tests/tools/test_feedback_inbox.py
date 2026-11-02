import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tools.coordination import feedback_inbox as inbox


FIXTURE = Path(__file__).resolve().parents[1] / "_fixtures/coordination/legacy-inbox-missing-ack.json"


class FeedbackInboxTests(unittest.TestCase):
    def test_legacy_missing_ack_is_pending_and_lossless(self):
        legacy = json.loads(FIXTURE.read_text(encoding="utf-8"))
        original_payload = legacy["events"]["legacy-event-1"]["content"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback-inbox.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            pending = inbox.merge_pending(path, [])
            self.assertEqual(len(pending), 1)
            self.assertIs(pending[0]["acknowledged"], False)
            self.assertEqual(pending[0]["content"], original_payload)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(saved["events"]), {"legacy-event-1"})
            self.assertEqual(saved["events"]["legacy-event-1"]["content"], original_payload)

    def test_malformed_envelope_fails_without_rewrite(self):
        malformed = {"schema": 1, "events": []}
        with self.assertRaisesRegex(ValueError, "original file must be preserved"):
            inbox.normalize_inbox(malformed)
        self.assertEqual(malformed, {"schema": 1, "events": []})

    def test_pending_is_idempotent_and_acknowledgement_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "inbox.json"
            decision = root / "decision.json"
            decision.write_text('{"decision":"ok"}', encoding="utf-8")
            event = {"event_id": "event-a", "kind": "completed"}
            self.assertEqual(inbox.merge_pending(path, [event]), inbox.merge_pending(path, [event]))
            inbox.acknowledge(path, ["event-a"], decision)
            self.assertEqual(inbox.merge_pending(path, [event]), [])
            row = json.loads(path.read_text(encoding="utf-8"))["events"]["event-a"]
            self.assertTrue(row["acknowledged"])
            self.assertEqual(row["decision_sha256"], __import__("hashlib").sha256(decision.read_bytes()).hexdigest())

    def test_transient_wait_fault_is_reported_and_retried_without_ack(self):
        operation = Mock(side_effect=[KeyError("acknowledged"), ["pending"]])
        errors, sleeps = [], []
        result = inbox.retry_call(operation, on_error=errors.append, sleep=sleeps.append)
        self.assertEqual(result, ["pending"])
        self.assertEqual(len(errors), 1)
        self.assertEqual(sleeps, [10])

    def test_immediate_pending_result_does_not_wait(self):
        operation = Mock(return_value=["event"])
        sleep = Mock()
        self.assertEqual(inbox.retry_call(operation, on_error=Mock(), sleep=sleep), ["event"])
        sleep.assert_not_called()

    def test_dispatch_is_idempotent_and_live_sid_is_not_launched(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "dispatch.json"
            decision = {"cycle": "cycle-a", "task": "bounded"}
            apply = Mock(return_value={"status": "applied"})
            first = inbox.apply_once(ledger, decision, apply)
            replay = inbox.apply_once(ledger, decision, apply)
            self.assertEqual(first, replay)
            apply.assert_called_once()
            dispatch = Mock()
            did_dispatch, result = inbox.dispatch_if_idle(lambda: True, lambda: True, dispatch)
            self.assertFalse(did_dispatch)
            self.assertIsNone(result)
            dispatch.assert_not_called()
            did_dispatch, result = inbox.dispatch_if_idle(lambda: False, lambda: False, dispatch)
            self.assertFalse(did_dispatch)
            dispatch.assert_not_called()

    def test_interrupted_intent_requires_reconciliation_not_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "dispatch.json"
            decision = {"cycle": "cycle-interrupted"}
            with self.assertRaisesRegex(RuntimeError, "crash"):
                inbox.apply_once(ledger, decision, lambda: (_ for _ in ()).throw(RuntimeError("crash")))
            replay = Mock()
            with self.assertRaisesRegex(RuntimeError, "requires reconciliation"):
                inbox.apply_once(ledger, decision, replay)
            replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
