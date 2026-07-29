import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


class LazyOperatorImportTest(unittest.TestCase):

    def test_imports_only_requested_mapper_modules(self):
        code = """
import json
import sys
from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.load import ensure_operator_registered

names = [
    "s3_download_file_mapper",
    "image_portrait_quality_mapper",
    "image_portrait_cache_router_mapper",
    "image_humanaesexpert_mapper",
]
for name in names:
    ensure_operator_registered(name)
print(json.dumps({
    "registered": sorted(name for name in names if name in OPERATORS.modules),
    "eager_loaded": "data_juicer.ops.mapper._eager" in sys.modules,
    "video_loaded": any(
        module.startswith("data_juicer.ops.mapper.video_")
        for module in sys.modules
    ),
}))
"""
        environment = os.environ.copy()
        environment["DATA_JUICER_LAZY_OP_IMPORT"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [
                    str(REPOSITORY),
                    environment.get("PYTHONPATH"),
                ],
            )
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(
            payload["registered"],
            [
                "image_humanaesexpert_mapper",
                "image_portrait_cache_router_mapper",
                "image_portrait_quality_mapper",
                "s3_download_file_mapper",
            ],
        )
        self.assertFalse(payload["eager_loaded"])
        self.assertFalse(payload["video_loaded"])


if __name__ == "__main__":
    unittest.main()
