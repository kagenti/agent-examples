import logging
import yaml

logger = logging.getLogger(__name__)

class FileRule:
    def __init__(self, pattern: str, target: str, action: str, priority: int = 0):
        self.pattern = pattern
        self.target = target
        self.action = action
        self.priority = priority

class RulesEngine:
    def __init__(self, rules: list[FileRule]):
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)

    def get_rules_summary(self) -> str:
        if not self.rules:
            return "No organization rules defined."
        summary = "File Organization Rules (sorted by priority):\n"
        for i, rule in enumerate(self.rules, 1):
            summary += f"{i}. Pattern: '{rule.pattern}' → {rule.action.upper()} to '{rule.target}' (Priority: {rule.priority})\n"
        return summary

def load_rules_from_string(yaml_content: str) -> RulesEngine:
    try:
        data = yaml.safe_load(yaml_content)
        rules = []
        if data and 'rules' in data:
            for rule_data in data['rules']:
                rules.append(FileRule(
                    pattern=rule_data.get('pattern', '*'),
                    target=rule_data.get('target', ''),
                    action=rule_data.get('action', 'copy'),
                    priority=rule_data.get('priority', 0)
                ))
        logger.info(f"Loaded {len(rules)} rules from YAML string")
        return RulesEngine(rules)
    except Exception as e:
        logger.error(f"Error loading rules from YAML string: {e}")
        return RulesEngine([])
