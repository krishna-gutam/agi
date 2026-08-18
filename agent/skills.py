import os
import yaml
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class SkillMetadata(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str
    triggers: List[str] = Field(default_factory=list)

class Skill(BaseModel):
    metadata: SkillMetadata
    body: str
    path: str

class SkillRegistry:
    """
    Skill Registry implementing progressive disclosure:
    - Level 1: Metadata loading (lightweight catalog of names, versions, descriptions, triggers)
    - Level 2: Full skill body loading on-demand
    """
    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = os.path.abspath(skills_dir or "skills")
        self._skills_cache: Dict[str, Skill] = {}
        self._load_catalog()

    def _load_catalog(self):
        """Discovers and loads Level 1 metadata for all skills in skills_dir."""
        if not os.path.exists(self.skills_dir):
            return

        for root, _, files in os.walk(self.skills_dir):
            for file in files:
                if file.endswith((".yaml", ".yml", ".md")):
                    path = os.path.join(root, file)
                    skill = self._parse_skill_file(path)
                    if skill:
                        key = f"{skill.metadata.name}@{skill.metadata.version}"
                        self._skills_cache[key] = skill
                        # Also register latest/unversioned alias if not present
                        base_key = skill.metadata.name
                        if base_key not in self._skills_cache:
                            self._skills_cache[base_key] = skill

    def _parse_skill_file(self, path: str) -> Optional[Skill]:
        """Parses frontmatter metadata and body from skill file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    body = parts[2].strip()
                    metadata = SkillMetadata(
                        name=frontmatter.get("name", os.path.basename(path)),
                        version=frontmatter.get("version", "1.0.0"),
                        description=frontmatter.get("description", ""),
                        triggers=frontmatter.get("triggers", [])
                    )
                    return Skill(metadata=metadata, body=body, path=path)
            
            # Default fallback if no frontmatter
            name = os.path.splitext(os.path.basename(path))[0]
            return Skill(
                metadata=SkillMetadata(name=name, description=content[:100]),
                body=content,
                path=path
            )
        except Exception:
            return None

    def get_level1_catalog(self) -> List[Dict[str, Any]]:
        """Level 1 progressive disclosure: Returns lightweight metadata catalog."""
        catalog = []
        seen = set()
        for key, skill in self._skills_cache.items():
            if "@" in key:  # prefer unique versioned keys for catalog listing
                if skill.metadata.name not in seen:
                    seen.add(skill.metadata.name)
                    catalog.append({
                        "name": skill.metadata.name,
                        "version": skill.metadata.version,
                        "description": skill.metadata.description,
                        "triggers": skill.metadata.triggers
                    })
        return catalog

    def load_skill_body(self, name: str, version: Optional[str] = None) -> Optional[str]:
        """Level 2 progressive disclosure: Loads full skill body on demand."""
        key = f"{name}@{version}" if version else name
        skill = self._skills_cache.get(key)
        if not skill and not version:
            skill = self._skills_cache.get(name)
        return skill.body if skill else None

    def evaluate_triggers(self, context_text: str) -> List[str]:
        """Evaluates triggers against context text and returns matching skill names."""
        matched = []
        seen = set()
        for skill in self._skills_cache.values():
            if skill.metadata.name in seen:
                continue
            for trigger in skill.metadata.triggers:
                if trigger.lower() in context_text.lower():
                    seen.add(skill.metadata.name)
                    matched.append(skill.metadata.name)
                    break
        return matched
