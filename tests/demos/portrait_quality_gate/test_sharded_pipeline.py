import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from demos.portrait_quality_gate.run_sharded_pipeline import (
    adapt_input_record,
    expected_output_rows,
    stage2_record_is_selected,
)
from demos.portrait_quality_gate.show_sharded_progress import snapshot


REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER = (
    REPOSITORY
    / "demos/portrait_quality_gate/run_sharded_pipeline.py"
)


class ShardedPortraitPipelineTest(unittest.TestCase):

    def write_fake_pipeline(self, root: Path) -> Path:
        script = root / "fake_pipeline.py"
        script.write_text(
            """
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--cache-root", required=True, type=Path)
parser.add_argument("--leave-cache-file", action="store_true")
parser.add_argument("--drop-last-row", action="store_true")
parser.add_argument("--fail-if-contains-id", type=int)
parser.add_argument("--fail-once-marker", type=Path)
args, _ = parser.parse_known_args()
args.output.mkdir(parents=True)
with args.input.open(encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
if (
    args.fail_if_contains_id is not None
    and any(row.get("id") == args.fail_if_contains_id for row in rows)
    and args.fail_once_marker is not None
    and not args.fail_once_marker.exists()
):
    args.fail_once_marker.write_text("failed-once\\n", encoding="utf-8")
    sys.exit(9)
if args.drop_last_row:
    rows = rows[:-1]
with (args.output / "part-000000.json").open("w", encoding="utf-8") as target:
    for row in rows:
        target.write(json.dumps(row) + "\\n")
if args.leave_cache_file:
    args.cache_root.mkdir(parents=True, exist_ok=True)
    (args.cache_root / "orphan.jpg").write_bytes(b"not-empty")
""".lstrip(),
            encoding="utf-8",
        )
        return script

    def base_command(
        self,
        input_path: Path,
        output_root: Path,
        cache_root: Path,
        work_root: Path,
        fake_pipeline: Path,
    ):
        return [
            sys.executable,
            str(RUNNER),
            "--mode",
            "stage1",
            "--pipeline-script",
            str(fake_pipeline),
            "--input",
            str(input_path),
            "--output-root",
            str(output_root),
            "--cache-root",
            str(cache_root),
            "--work-root",
            str(work_root),
            "--shard-size",
            "2",
            "--logical-shard-size",
            "4",
            "--cache-zero-timeout",
            "0",
        ]

    def test_shards_success_resume_and_incomplete_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps({"id": index}) + "\n"
                    for index in range(5)
                ),
                encoding="utf-8",
            )
            output_root = root / "output"
            cache_root = root / "cache"
            work_root = root / "work"
            fake_pipeline = self.write_fake_pipeline(root)
            command = self.base_command(
                input_path,
                output_root,
                cache_root,
                work_root,
                fake_pipeline,
            )

            first = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("completed=3, skipped=0", first.stdout)
            self.assertTrue(
                (output_root / "shard-000000/SUCCESS").is_file()
            )
            self.assertTrue(
                (output_root / "shard-000001/SUCCESS").is_file()
            )
            progress = snapshot(
                output_root,
                work_root,
                cache_root,
                None,
            )
            self.assertIn("logical shards: 2/2", progress)
            self.assertIn("micro shards: 3/3", progress)
            locations = (
                (0, 0, 2),
                (0, 1, 2),
                (1, 0, 1),
            )
            for logical_index, micro_index, expected_rows in locations:
                shard_dir = (
                    output_root
                    / f"shard-{logical_index:06d}"
                    / f"micro-{micro_index:04d}"
                )
                marker = json.loads(
                    (shard_dir / "SUCCESS").read_text(encoding="utf-8")
                )
                self.assertEqual(marker["input_rows"], expected_rows)
                self.assertEqual(marker["output_rows"], expected_rows)
                self.assertEqual(marker["cache"], {
                    "bytes": 0,
                    "files": 0,
                    "references": 0,
                })
                self.assertTrue((shard_dir / "data.jsonl").is_file())

            data_path = (
                output_root
                / "shard-000000/micro-0001/data.jsonl"
            )
            initial_mtime = data_path.stat().st_mtime_ns
            second = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("completed=0, skipped=3", second.stdout)
            self.assertEqual(data_path.stat().st_mtime_ns, initial_mtime)

            (
                output_root
                / "shard-000000/micro-0001/SUCCESS"
            ).unlink()
            data_path.write_text('{"incomplete":true}\n', encoding="utf-8")
            rerun = subprocess.run(
                [*command, "--logical-shard-index", "0"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("completed=1, skipped=1", rerun.stdout)
            self.assertEqual(
                len(data_path.read_text(encoding="utf-8").splitlines()),
                2,
            )

    def test_nonzero_cache_prevents_success_and_cleans_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            input_path.write_text('{"id":1}\n', encoding="utf-8")
            output_root = root / "output"
            cache_root = root / "cache"
            work_root = root / "work"
            fake_pipeline = self.write_fake_pipeline(root)
            command = self.base_command(
                input_path,
                output_root,
                cache_root,
                work_root,
                fake_pipeline,
            )

            result = subprocess.run(
                [*command, "--leave-cache-file"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(
                (
                    output_root
                    / "shard-000000/micro-0000"
                ).exists()
            )
            attempts = output_root / ".attempts"
            self.assertEqual(list(attempts.iterdir()), [])
            self.assertTrue(
                (
                    work_root
                    / "failures/shard-000000-micro-0000"
                ).is_file()
            )
            self.assertTrue(
                (
                    cache_root
                    / "shard-000000/micro-0000/orphan.jpg"
                ).is_file()
            )

    def test_row_mismatch_prevents_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            input_path.write_text(
                '{"id":1}\n{"id":2}\n',
                encoding="utf-8",
            )
            output_root = root / "output"
            command = self.base_command(
                input_path,
                output_root,
                root / "cache",
                root / "work",
                self.write_fake_pipeline(root),
            )
            result = subprocess.run(
                [*command, "--drop-last-row"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output row mismatch", result.stderr)
            self.assertFalse(
                (
                    output_root
                    / "shard-000000/micro-0000"
                ).exists()
            )

    def test_fused_execution_windows_resume_and_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps({"id": index}) + "\n"
                    for index in range(5)
                ),
                encoding="utf-8",
            )
            output_root = root / "output"
            cache_root = root / "cache"
            work_root = root / "work"
            failure_marker = root / "fail-once"
            command = [
                sys.executable,
                str(RUNNER),
                "--mode",
                "fused",
                "--pipeline-script",
                str(self.write_fake_pipeline(root)),
                "--input",
                str(input_path),
                "--output-root",
                str(output_root),
                "--cache-root",
                str(cache_root),
                "--work-root",
                str(work_root),
                "--micro-shard-size",
                "5",
                "--logical-shard-size",
                "5",
                "--execution-window-size",
                "2",
                "--cache-zero-timeout",
                "0",
                "--fail-if-contains-id",
                "3",
                "--fail-once-marker",
                str(failure_marker),
            ]

            failed = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            checkpoint_root = (
                work_root
                / "window_checkpoints/shard-000000/micro-0000"
            )
            self.assertTrue(
                (
                    checkpoint_root
                    / "window-0000/WINDOW_SUCCESS"
                ).is_file()
            )
            self.assertFalse(
                (
                    checkpoint_root
                    / "window-0001/WINDOW_SUCCESS"
                ).is_file()
            )
            progress = snapshot(
                output_root,
                work_root,
                cache_root,
                0,
            )
            self.assertIn("execution windows: 1/3", progress)

            resumed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(
                "[window-skip] shard-000000/micro-0000/window-0000",
                resumed.stdout,
            )
            final_dir = (
                output_root
                / "shard-000000/micro-0000"
            )
            marker = json.loads(
                (final_dir / "SUCCESS").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["execution_window_size"], 2)
            self.assertEqual(marker["execution_windows"], 3)
            self.assertEqual(marker["output_rows"], 5)
            self.assertEqual(
                len(
                    (
                        final_dir / "data.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                ),
                5,
            )
            self.assertFalse(checkpoint_root.exists())

    def test_stage2_expected_rows_matches_filter_rules(self):
        selected = {
            "__dj__meta__": {
                "portrait_quality": [
                    {
                        "status": "reject",
                        "human_status": "portrait_clear",
                    }
                ]
            }
        }
        rejected = {
            "__dj__meta__": {
                "portrait_quality": [
                    {
                        "status": "pass",
                        "human_status": "human_uncertain",
                    }
                ]
            }
        }
        self.assertTrue(stage2_record_is_selected(selected, False))
        self.assertFalse(stage2_record_is_selected(selected, True))
        self.assertFalse(stage2_record_is_selected(rejected, False))

        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "stage1.jsonl"
            input_path.write_text(
                json.dumps(selected) + "\n" + json.dumps(rejected) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                expected_output_rows("stage2", input_path, 2, []),
                1,
            )
            self.assertEqual(
                expected_output_rows(
                    "stage2",
                    input_path,
                    2,
                    ["--exclude-hard-rejects"],
                ),
                0,
            )

    def test_purchased_selection_adapter_adds_data_juicer_fields(self):
        source = {
            "id": "sample-1",
            "conversations": [
                {"from": "human", "value": "portrait caption"},
                {"from": "gpt", "value": "<image>"},
            ],
            "_sample": {
                "image_uri": "s3://bucket/path/image.jpg",
                "image_root": "s3://bucket/",
                "source_meta": "s3://bucket/meta/source.jsonl",
                "offset": 123,
            },
        }
        adapted = json.loads(
            adapt_input_record(
                json.dumps(source).encode("utf-8"),
                "purchased-selection",
                Path("/input.jsonl"),
                1,
            )
        )
        self.assertEqual(
            adapted["images"],
            ["s3://bucket/path/image.jpg"],
        )
        self.assertEqual(adapted["text"], "portrait caption")
        self.assertEqual(adapted["source_offset"], 123)


if __name__ == "__main__":
    unittest.main()
