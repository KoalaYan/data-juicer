import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path

from demos.portrait_quality_gate.run_direct_portrait_humanaesexpert import (
    atomic_write_json,
    download_producer,
    jsonl_summary,
    valid_micro_success,
    write_logical_success,
)


class DirectPortraitPipelineTest(unittest.TestCase):

    def test_download_producer_is_bounded_and_dispatches_every_record(self):
        records = [{"id": index} for index in range(30)]
        quality_queue = queue.Queue(maxsize=30)
        result_queue = queue.Queue()
        stop_event = threading.Event()
        state = {"done": False, "dispatched": 0, "fatal": None}
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def download_one(sequence, record):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.002)
                return sequence, dict(record), {"download_bytes": 10}
            finally:
                with lock:
                    active -= 1

        download_producer(
            records=records,
            download_one=download_one,
            workers=4,
            max_pending=6,
            quality_queue=quality_queue,
            result_queue=result_queue,
            state=state,
            stop_event=stop_event,
            cleanup_record=lambda record: None,
        )

        observed = sorted(
            quality_queue.get_nowait()[1]
            for _ in range(len(records))
        )
        self.assertEqual(observed, list(range(len(records))))
        self.assertEqual(maximum_active, 4)
        self.assertEqual(state["dispatched"], len(records))
        self.assertTrue(state["done"])
        self.assertIsNone(state["fatal"])
        self.assertTrue(result_queue.empty())

    def test_download_failure_becomes_one_ordered_error(self):
        records = [{"id": index} for index in range(5)]
        quality_queue = queue.Queue()
        result_queue = queue.Queue()
        state = {"done": False, "dispatched": 0, "fatal": None}

        def download_one(sequence, record):
            if sequence == 2:
                raise OSError("synthetic download failure")
            return sequence, dict(record), {}

        download_producer(
            records=records,
            download_one=download_one,
            workers=2,
            max_pending=3,
            quality_queue=quality_queue,
            result_queue=result_queue,
            state=state,
            stop_event=threading.Event(),
            cleanup_record=lambda record: None,
        )

        self.assertEqual(quality_queue.qsize(), 4)
        status, sequence, payload, _ = result_queue.get_nowait()
        self.assertEqual(status, "error")
        self.assertEqual(sequence, 2)
        self.assertEqual(payload["stage"], "download")
        self.assertIn("synthetic download failure", payload["message"])
        self.assertTrue(result_queue.empty())

    def test_micro_and_logical_success_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            shards = []
            for index in range(2):
                shard = {
                    "index": index,
                    "logical_index": 0,
                    "micro_index": index,
                    "rows": 1,
                    "sha256": f"input-{index}",
                }
                shards.append(shard)
                micro_dir = (
                    output_root
                    / "shard-000000"
                    / f"micro-{index:04d}"
                )
                micro_dir.mkdir(parents=True)
                data_path = micro_dir / "data.jsonl"
                data_path.write_text(
                    json.dumps({"id": index}) + "\n",
                    encoding="utf-8",
                )
                output_rows, output_sha256 = jsonl_summary(data_path)
                atomic_write_json(
                    micro_dir / "SUCCESS",
                    {
                        "version": 1,
                        "mode": "direct-fused",
                        "logical_shard_index": 0,
                        "micro_shard_index": index,
                        "input_rows": 1,
                        "input_sha256": f"input-{index}",
                        "output_rows": output_rows,
                        "output_sha256": output_sha256,
                        "scored_rows": index,
                        "quality_only_rows": 1 - index,
                        "cache": {
                            "files": 0,
                            "bytes": 0,
                            "references": 0,
                            "partial_files": 0,
                        },
                    },
                )
                self.assertTrue(valid_micro_success(micro_dir, shard))

            self.assertTrue(
                write_logical_success(output_root, 0, shards)
            )
            logical = json.loads(
                (
                    output_root / "shard-000000/SUCCESS"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(logical["input_rows"], 2)
            self.assertEqual(logical["output_rows"], 2)
            self.assertEqual(logical["scored_rows"], 1)
            self.assertEqual(logical["quality_only_rows"], 1)

            (
                output_root
                / "shard-000000/micro-0001/data.jsonl"
            ).write_text('{"changed":true}\n', encoding="utf-8")
            self.assertFalse(
                valid_micro_success(
                    output_root / "shard-000000/micro-0001",
                    shards[1],
                )
            )
            self.assertFalse(
                write_logical_success(output_root, 0, shards)
            )
            self.assertFalse(
                (output_root / "shard-000000/SUCCESS").exists()
            )


if __name__ == "__main__":
    unittest.main()
