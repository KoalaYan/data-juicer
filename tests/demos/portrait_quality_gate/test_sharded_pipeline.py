import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from demos.portrait_quality_gate.run_sharded_pipeline import (
    expected_output_rows,
    stage2_record_is_selected,
)


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
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--cache-root", required=True, type=Path)
parser.add_argument("--leave-cache-file", action="store_true")
parser.add_argument("--drop-last-row", action="store_true")
args, _ = parser.parse_known_args()
args.output.mkdir(parents=True)
with args.input.open(encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
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
            for index, expected_rows in enumerate((2, 2, 1)):
                shard_dir = output_root / f"shard-{index:06d}"
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

            data_path = output_root / "shard-000001/data.jsonl"
            initial_mtime = data_path.stat().st_mtime_ns
            second = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("completed=0, skipped=3", second.stdout)
            self.assertEqual(data_path.stat().st_mtime_ns, initial_mtime)

            (output_root / "shard-000001/SUCCESS").unlink()
            data_path.write_text('{"incomplete":true}\n', encoding="utf-8")
            rerun = subprocess.run(
                [*command, "--shard-index", "1"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("completed=1, skipped=0", rerun.stdout)
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
            self.assertFalse((output_root / "shard-000000").exists())
            attempts = output_root / ".attempts"
            self.assertEqual(list(attempts.iterdir()), [])
            self.assertTrue(
                (work_root / "failures/shard-000000").is_file()
            )
            self.assertTrue(
                (cache_root / "shard-000000/orphan.jpg").is_file()
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
            self.assertFalse((output_root / "shard-000000").exists())

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


if __name__ == "__main__":
    unittest.main()
