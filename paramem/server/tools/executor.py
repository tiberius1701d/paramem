"""Tool execution infrastructure for the direct cloud escalation path.

Primary escalation routes through HA's conversation.process WebSocket,
which handles tool execution internally. This module provides the
fallback path: direct cloud model calls with server-side tool execution.

Also provides the Extended OpenAI Conversation format execution types
(template, script, native) via HA WebSocket execute_script.

Latency budget: 8s total, 3s per-tool timeout (configured in HAClient).
"""

import json
import logging
import time

from paramem.server.cloud.base import CloudAgent, CloudResponse, ToolCall
from paramem.server.tools.ha_client import HAClient
from paramem.server.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5
TOTAL_TIMEOUT_SECONDS = 8.0
MAX_TOOL_RESPONSE_LENGTH = 2000  # truncate large tool results


def execute_tool_loop(
    initial_response: CloudResponse,
    cloud_agent: CloudAgent,
    query: str,
    system_prompt: str,
    ha_client: HAClient | None,
    registry: ToolRegistry,
    max_rounds: int = MAX_TOOL_ROUNDS,
    total_timeout: float = TOTAL_TIMEOUT_SECONDS,
) -> str:
    """Run the agentic loop until the cloud model returns final text.

    Returns the final text response from the cloud model.
    """
    response = initial_response
    tool_results: list[dict] = []
    start_time = time.monotonic()

    for round_num in range(max_rounds):
        if not response.requires_tool_execution:
            return response.text or "I couldn't generate a response."

        elapsed = time.monotonic() - start_time
        if elapsed >= total_timeout:
            logger.warning(
                "Agentic loop timeout after %.1fs (%d rounds)",
                elapsed,
                round_num,
            )
            return "I ran out of time processing your request."

        # Execute each tool call
        round_results = []
        for tool_call in response.tool_calls:
            result = _execute_single_tool(tool_call, ha_client, registry)
            round_results.append(
                {
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": result,
                }
            )

        tool_results.extend(round_results)

        logger.info(
            "Agentic loop round %d: %d tool calls executed",
            round_num + 1,
            len(round_results),
        )

        # Send results back to cloud model
        response = cloud_agent.call(
            query=query,
            system_prompt=system_prompt,
            tool_results=tool_results,
        )

    # Max rounds exhausted
    if response.text:
        return response.text
    logger.warning("Agentic loop exhausted %d rounds", max_rounds)
    return "I needed too many steps to complete your request."


def _execute_single_tool(
    tool_call: ToolCall,
    ha_client: HAClient | None,
    registry: ToolRegistry,
) -> str:
    """Execute a single tool call and return the result as a string."""
    name = tool_call.name
    arguments = tool_call.arguments

    logger.info("Tool call: %s(%s)", name, arguments)

    # Extended execution (template / script / native from Extended OpenAI format)
    execution = registry.get_execution_info(name)
    if execution and ha_client is not None:
        return _execute_extended_tool(name, arguments, execution, ha_client, registry)

    # Standard HA-proxied service call
    if registry.is_ha_proxied(name) and ha_client is not None:
        return _execute_ha_tool(name, arguments, ha_client, registry)

    if name in registry.cloud_native_tools:
        return f"Tool {name} is cloud-native and should be handled by the model."

    logger.warning("Unknown tool: %s", name)
    return f"Error: tool '{name}' is not available."


def _substitute_template_args(template_str: str, arguments: dict) -> str:
    """Replace {{ param }} placeholders with actual argument values."""
    result = template_str
    for key, value in arguments.items():
        result = result.replace("{{ " + key + " }}", str(value))
        result = result.replace("{{" + key + "}}", str(value))
    return result


def _execute_extended_tool(
    name: str,
    arguments: dict,
    execution: dict,
    ha_client: HAClient,
    registry: ToolRegistry,
) -> str:
    """Execute a tool defined in Extended OpenAI Conversation format.

    All tool types are routed through HA WebSocket execute_script.
    Each type is converted to a sequence:
    - template → single-step template rendering
    - script → the defined sequence with response capture
    - native → service call sequence from arguments
    """
    func_type = execution.get("type", "native")

    # Resolve entity names in arguments before building the sequence
    resolved_args = ha_client.resolve_arguments(arguments) if arguments else {}

    if func_type == "template":
        sequence = _template_to_sequence(execution, resolved_args)
    elif func_type == "script":
        sequence = _script_to_sequence(execution, resolved_args)
    elif func_type == "native":
        sequence = _native_to_sequence(arguments, ha_client)
    else:
        logger.warning("Unknown execution type '%s' for tool %s", func_type, name)
        return f"Error: unknown execution type '{func_type}'"

    if not sequence:
        return f"Error: could not build sequence for tool {name}"

    result = ha_client.execute_script_ws(sequence, timeout=10.0)
    if result is None:
        return f"Error: execution failed for {name}"

    # Extract the result string from the response
    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, str):
                return _truncate(value)
        return _truncate(json.dumps(result, ensure_ascii=False, default=str))
    if isinstance(result, str):
        return _truncate(result)

    return _truncate(json.dumps(result, ensure_ascii=False, default=str))


def _template_to_sequence(execution: dict, arguments: dict) -> list[dict]:
    """Convert a template tool to a sequence that renders and returns it."""
    template = execution.get("value_template", "")
    if arguments:
        template = _substitute_template_args(template, arguments)

    return [
        {"variables": {"_function_result": {"result": template}}},
        {"stop": "Done", "response_variable": "_function_result"},
    ]


def _script_to_sequence(execution: dict, arguments: dict) -> list[dict]:
    """Convert a script tool to a resolved sequence with return step."""
    import copy

    sequence = copy.deepcopy(execution.get("sequence", []))

    # Substitute tool arguments into data blocks
    for step in sequence:
        data = step.get("data", {})
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = _substitute_template_args(value, arguments)

    # Append stop step if not already present
    for step in sequence:
        if "stop" in step:
            return sequence

    result = list(sequence)
    for i, step in enumerate(result):
        variables = step.get("variables")
        if variables and "_function_result" in variables:
            original = variables["_function_result"]
            if isinstance(original, str):
                result[i] = {"variables": {"_function_result": {"result": original}}}
            result.append({"stop": "Done", "response_variable": "_function_result"})
            return result

    return result


def _native_to_sequence(arguments: dict, ha_client: HAClient) -> list[dict]:
    """Convert a native tool call to a service call sequence."""
    sequence: list[dict] = []
    results_template_parts: list[str] = []

    if "list" in arguments:
        # execute_services: {list: [{domain, service, service_data}]}
        for idx, item in enumerate(arguments["list"]):
            domain = item.get("domain", "")
            service = item.get("service", "")
            service_data = dict(item.get("service_data", {}))

            entity_id = service_data.get("entity_id", "")
            if entity_id and not ("." in entity_id and entity_id.split(".")[0].isalpha()):
                service_data["entity_id"] = ha_client.resolve_entity_name(entity_id)

            sequence.append({"action": f"{domain}.{service}", "data": service_data})
            results_template_parts.append(f"{domain}.{service}: ok")
    else:
        domain = arguments.get("domain", "")
        service = arguments.get("service", "")
        data = {k: v for k, v in arguments.items() if k not in ("domain", "service")}
        sequence.append({"action": f"{domain}.{service}", "data": data})
        results_template_parts.append(f"{domain}.{service}: ok")

    # Add a return step with confirmation
    confirmation = "; ".join(results_template_parts)
    sequence.append({"variables": {"_function_result": {"result": confirmation}}})
    sequence.append({"stop": "Done", "response_variable": "_function_result"})
    return sequence


def _truncate(text: str) -> str:
    """Truncate tool result to max length."""
    if len(text) > MAX_TOOL_RESPONSE_LENGTH:
        return text[:MAX_TOOL_RESPONSE_LENGTH] + "... (truncated)"
    return text


def _execute_ha_tool(
    name: str,
    arguments: dict,
    ha_client: HAClient,
    registry: ToolRegistry,
) -> str:
    """Execute a tool via HA REST API."""
    parts = name.split(".", 1)
    if len(parts) != 2:
        return f"Error: invalid tool name format '{name}'"

    domain, service = parts

    # Resolve friendly names to entity IDs (only for entity-type params)
    entity_params = registry.entity_params(name)
    if entity_params:
        for param_name in entity_params:
            if param_name in arguments and isinstance(arguments[param_name], str):
                arguments[param_name] = ha_client.resolve_entity_name(arguments[param_name])

    # Scripts may return data via response_variable — request it
    return_response = domain == "script"
    result = ha_client.call_service(
        domain, service, data=arguments, return_response=return_response
    )

    if result is None:
        return f"Error: HA service {name} failed or timed out."

    try:
        result_str = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        result_str = str(result)
    if len(result_str) > MAX_TOOL_RESPONSE_LENGTH:
        result_str = result_str[:MAX_TOOL_RESPONSE_LENGTH] + "... (truncated)"
    return result_str
