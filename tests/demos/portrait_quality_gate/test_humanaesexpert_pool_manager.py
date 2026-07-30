import importlib.metadata
import unittest
from unittest.mock import patch

from demos.portrait_quality_gate.manage_humanaesexpert_pool import (
    validate_dependencies,
)


class HumanAesExpertPoolManagerTest(unittest.TestCase):

    def test_dependency_preflight_reports_all_failures(self):
        versions = {
            "present": "1.0",
            "mismatch": "2.0",
        }

        def fake_version(distribution):
            if distribution not in versions:
                raise importlib.metadata.PackageNotFoundError(distribution)
            return versions[distribution]

        with (
            patch(
                "importlib.metadata.version",
                side_effect=fake_version,
            ),
            self.assertRaises(RuntimeError) as context,
        ):
            validate_dependencies(
                {
                    "present": "1.0",
                    "mismatch": "1.0",
                    "missing": "3.0",
                }
            )

        message = str(context.exception)
        self.assertIn("mismatch==1.0 (found 2.0)", message)
        self.assertIn("missing==3.0 (missing)", message)


if __name__ == "__main__":
    unittest.main()
