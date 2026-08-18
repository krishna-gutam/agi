import os
import tempfile
import pytest
from agent.skills import SkillRegistry

def test_skills_progressive_disclosure_and_triggers():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_content = """---
name: data_cleaner
version: "1.2.0"
description: Cleans messy CSV data.
triggers:
  - clean csv
  - missing values
---
# Data Cleaner Skill Body
Instructions on how to clean data...
"""
        skill_path = os.path.join(tmpdir, "data_cleaner.md")
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(skill_content)

        registry = SkillRegistry(skills_dir=tmpdir)

        catalog = registry.get_level1_catalog()
        assert len(catalog) == 1
        assert catalog[0]["name"] == "data_cleaner"
        assert catalog[0]["version"] == "1.2.0"
        assert "clean csv" in catalog[0]["triggers"]

        body = registry.load_skill_body("data_cleaner", version="1.2.0")
        assert body is not None
        assert "Instructions on how to clean data" in body

        body_default = registry.load_skill_body("data_cleaner")
        assert body_default is not None

        matched_skills = registry.evaluate_triggers("Please help me clean csv files.")
        assert "data_cleaner" in matched_skills

        no_match = registry.evaluate_triggers("Write a poem.")
        assert "data_cleaner" not in no_match

