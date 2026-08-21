from __future__ import annotations

import json
import unittest
from pathlib import Path


class CompatibilityKnowledgeTests(unittest.TestCase):
    def test_rules_have_unique_ids_and_verification_steps(self) -> None:
        path = Path(__file__).resolve().parents[1] / "compatibility" / "rules.json"
        knowledge = json.loads(path.read_text(encoding="utf-8"))
        rules = knowledge["rules"]
        identifiers = [rule["id"] for rule in rules]

        self.assertEqual(knowledge["schema_version"], 1)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(all(rule["verification"] for rule in rules))
        self.assertTrue(all("automatic" in rule["repair"] for rule in rules))


if __name__ == "__main__":
    unittest.main()
