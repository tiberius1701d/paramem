"""Tool registry — merges HA auto-discovery with manual tools.yaml.

Default-deny: auto-discovery produces nothing without an explicit allowlist.
Sensitive domains are blocked even if listed unless explicitly overridden.
"""

import fnmatch
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Domains that require explicit override to expose
SENSITIVE_DOMAINS = {"alarm_control_panel", "lock", "person", "device_tracker"}


class ToolRegistry:
    """Manages tool definitions from HA and manual config."""

    def __init__(self):
        self._tools: dict[str, dict] = {}  # name → standard tool definition
        self._ha_proxied: set[str] = set()  # tools that execute via HA
        self._cloud_native: set[str] = set()  # tools that execute via cloud
        # tool_name → set of parameter names that are entity references
        self._entity_params: dict[str, set[str]] = {}
        # Extended execution metadata (separate from tool defs sent to LLM)
        self._execution_map: dict[str, dict] = {}

    @property
    def tools(self) -> list[dict]:
        """All tool definitions in standard format."""
        return list(self._tools.values())

    @property
    def ha_proxied_tools(self) -> set[str]:
        return self._ha_proxied

    @property
    def cloud_native_tools(self) -> set[str]:
        return self._cloud_native

    def get_tool(self, name: str) -> dict | None:
        return self._tools.get(name)

    def is_ha_proxied(self, name: str) -> bool:
        return name in self._ha_proxied

    def entity_params(self, name: str) -> set[str]:
        """Return parameter names that are entity references for this tool."""
        return self._entity_params.get(name, set())

    def get_execution_info(self, name: str) -> dict | None:
        """Return extended execution metadata for a tool, or None."""
        return self._execution_map.get(name)

    def load_from_ha(
        self,
        ha_services: list[dict],
        allowlist: list[str] | None = None,
        sensitive_override: bool = False,
    ):
        """Load tool definitions from HA service discovery.

        Default-deny: if allowlist is None or empty, no tools are loaded.
        Sensitive domains are blocked unless sensitive_override is True.
        """
        if not allowlist:
            logger.info("HA auto-discovery: no allowlist configured, skipping")
            return

        count = 0
        for service_domain in ha_services:
            domain = service_domain.get("domain", "")
            services = service_domain.get("services", {})

            for service_name, service_info in services.items():
                full_name = f"{domain}.{service_name}"

                if not _matches_allowlist(full_name, allowlist):
                    continue

                if domain in SENSITIVE_DOMAINS and not sensitive_override:
                    logger.warning(
                        "Blocked sensitive service: %s (use sensitive_override)",
                        full_name,
                    )
                    continue

                tool_def, entity_fields = _ha_service_to_tool(domain, service_name, service_info)
                self._tools[full_name] = tool_def
                self._ha_proxied.add(full_name)
                if entity_fields:
                    self._entity_params[full_name] = entity_fields
                count += 1

        logger.info("HA auto-discovery: loaded %d tools", count)

    def load_from_yaml(self, path: str | Path):
        """Load tool definitions from a YAML file.

        Auto-detects format:
        - Extended OpenAI Conversation format (list of {spec, function} dicts)
        - Legacy format ({tools: [{name, description, parameters, execution}]})

        Tools defined here override auto-discovered tools with the same name.
        """
        path = Path(path)
        if not path.exists():
            logger.debug("tools.yaml not found at %s, skipping", path)
            return

        with open(path) as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return

        # Auto-detect format
        if isinstance(raw, list) and raw and "spec" in raw[0]:
            self._load_extended_tools(raw, path)
        elif isinstance(raw, dict) and "tools" in raw:
            self._load_legacy_tools(raw["tools"], path)
        else:
            logger.warning("Unrecognized tools.yaml format at %s", path)

    def _load_legacy_tools(self, tools: list[dict], path: Path):
        """Load tools in the legacy {name, description, parameters, execution} format."""
        count = 0
        for tool in tools:
            name = tool.get("name")
            if not name:
                continue

            domain = name.split(".")[0] if "." in name else ""
            if domain in SENSITIVE_DOMAINS:
                logger.warning("Blocked sensitive tool from tools.yaml: %s", name)
                continue

            execution = tool.get("execution", "ha")
            tool_def = {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            }

            self._tools[name] = tool_def
            if execution == "cloud_native":
                self._cloud_native.add(name)
                self._ha_proxied.discard(name)
            else:
                self._ha_proxied.add(name)
                self._cloud_native.discard(name)
            count += 1

        logger.info("tools.yaml (legacy): loaded %d tools from %s", count, path)

    def _load_extended_tools(self, entries: list[dict], path: Path):
        """Load tools in Extended OpenAI Conversation format.

        Each entry has:
          spec: {name, description, parameters}
          function: {type: template|script|native, ...}
        """
        count = 0
        for entry in entries:
            spec = entry.get("spec", {})
            function = entry.get("function", {})

            name = spec.get("name")
            if not name:
                continue

            tool_def = {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
            }

            func_type = function.get("type", "native")
            execution_info: dict = {"type": func_type}

            if func_type == "template":
                execution_info["value_template"] = function.get("value_template", "")
            elif func_type == "script":
                execution_info["sequence"] = function.get("sequence", [])
            elif func_type == "native":
                execution_info["native_name"] = function.get("name", "")

            self._tools[name] = tool_def
            self._execution_map[name] = execution_info
            self._ha_proxied.add(name)
            self._cloud_native.discard(name)
            count += 1

        logger.info("tools.yaml (extended): loaded %d tools from %s", count, path)


def _matches_allowlist(full_name: str, allowlist: list[str]) -> bool:
    """Check if a service name matches any allowlist pattern.

    Supports glob patterns: 'light.*', 'script.music_*', etc.
    """
    return any(fnmatch.fnmatch(full_name, pattern) for pattern in allowlist)


def _ha_service_to_tool(
    domain: str, service_name: str, service_info: dict
) -> tuple[dict, set[str]]:
    """Convert HA service schema to standard tool definition.

    Returns (tool_def, entity_fields) where entity_fields is the set of
    parameter names that use an entity selector (need name→ID resolution).
    """
    description = service_info.get("description", f"{domain}.{service_name}")
    fields = service_info.get("fields", {})

    properties = {}
    required = []
    entity_fields: set[str] = set()
    for field_name, field_info in fields.items():
        prop: dict = {"type": "string"}
        if "description" in field_info:
            prop["description"] = field_info["description"]
        selector = field_info.get("selector", {})
        if "number" in selector:
            prop["type"] = "number"
        elif "boolean" in selector:
            prop["type"] = "boolean"
        elif "select" in selector:
            options = selector["select"].get("options", [])
            if options:
                prop["enum"] = options
        elif "entity" in selector or "target" in selector:
            entity_fields.add(field_name)
        if field_info.get("required"):
            required.append(field_name)
        properties[field_name] = prop

    # Also flag common entity parameter names (HA convention)
    for field_name in properties:
        if field_name == "entity_id" or field_name.endswith("_entity"):
            entity_fields.add(field_name)

    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    return (
        {
            "name": f"{domain}.{service_name}",
            "description": description,
            "parameters": parameters,
        },
        entity_fields,
    )
