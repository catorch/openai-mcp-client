#!/usr/bin/env python3
"""chat_cli_mcp.py – Chat CLI with Model Context Protocol (MCP) tool support.

Usage examples
--------------
# Start with a local filesystem MCP server (needs Node.js + @modelcontextprotocol/server-filesystem)
$ python chat_cli_mcp.py new "FS assistant" --mcp-stdio npx -y @modelcontextprotocol/server-filesystem .

# Start with multiple MCP servers defined in a config file
$ python chat_cli_mcp.py new "Multi-tool assistant" --mcp-config ~/.mcp_servers.json

# Resume an existing chat
$ python chat_cli_mcp.py chat 20250417-123456-abcdef
"""

import argparse, asyncio, json, sys, uuid, datetime as _dt
from pathlib import Path
# ---------------------------------------------------------------------------
# New imports
# ---------------------------------------------------------------------------
from typing import Dict, Any, Tuple, List, Optional, Union
from collections import defaultdict

# --- Colors ------------------------------------------------------------------
try:
    from colorama import init as _colorama_init, Fore, Style
    _colorama_init()          # enables ANSI on Windows
except ImportError:           # colorama not installed – fall back to no‑ops
    class _Dummy:             # pylint: disable=too-few-public-methods
        def __getattr__(self, name): return ''
    Fore = Style = _Dummy()

def clr(text: str, *colors: str) -> str:
    """Wrap text in ANSI color codes."""
    return "".join(colors) + text + Style.RESET_ALL
# ---------------------------------------------------------------------------

from openai import OpenAI
from openai._exceptions import APIError, RateLimitError, APITimeoutError

# Agents‑SDK MCP helpers
# ---------------------------------------------------------------------------
# Updated imports
# ---------------------------------------------------------------------------
from agents.mcp.server import MCPServerStdio, MCPServerSse
from agents.mcp.util import MCPUtil

CHAT_DIR = Path.home() / ".openai_chats"
CHAT_DIR.mkdir(exist_ok=True)
CONFIG_FILE = Path(__file__).parent / "config.json" # Default config file path

def _now() -> str: return _dt.datetime.now().isoformat(timespec="seconds")
def _chat_path(cid: str) -> Path: return CHAT_DIR / f"{cid}.json"

# ---------- Storage helpers ----------
def create_chat(title: str, model: str) -> str:
    cid = _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    (_chat_path(cid)).write_text(json.dumps({
        "id": cid, "title": title, "model": model, "created_at": _now(), "messages": []
    }, indent=2))
    print(f"[✓] Created chat {cid}: {title}")
    return cid

def _load(cid: str) -> Tuple[Dict[str, Any], Path]: 
    p=_chat_path(cid)
    if not p.exists(): sys.exit(f"[✗] Chat {cid} not found"); 
    return json.loads(p.read_text()), p
def _save(d: Dict[str, Any], p: Path) -> None: p.write_text(json.dumps(d,indent=2))

# ---------- MCP Handling ----------
# ---------------------------------------------------------------------------
# New helper: load JSON and spin up *all* MCP servers in parallel
# ---------------------------------------------------------------------------
async def prepare_mcp_servers(config_path: Path) -> tuple[
    dict[str, Any],              # tool_name  -> MCPServer
    list[dict[str, Any]],        # tools[]    ready for Chat API
    list[MCPServerStdio | MCPServerSse],  # every live server (for cleanup)
]:
    if not config_path.exists():
        print(f"[MCP] Config {config_path} not found – skipping MCP")
        return {}, [], []

    cfg = json.loads(config_path.read_text()).get("mcpServers", {})
    live_servers: list[MCPServerStdio | MCPServerSse] = []
    tool_to_server: dict[str, Any] = {}
    tools_json: list[dict[str, Any]] = []

    async def boot(alias: str, spec: dict[str, Any]):
        # We only implement stdio here; add SSE if you need it
        srv = MCPServerStdio(
            params={
                "command": spec["command"],
                "args": spec.get("args", []),
                "env": spec.get("env"),        # may be None
                "cwd": spec.get("cwd"),
            },
            cache_tools_list=True,
            name=alias,
        )
        await srv.connect()

        # ----- tee stdout / stderr ------------------------------------------------
        async def pipe_stream(stream, label):
            async for chunk in stream:          # anyio streams yield bytes
                line = chunk.decode().rstrip()
                print(clr(f"[{alias} {label}] {line}", Fore.YELLOW))

        proc = getattr(srv, "_process", None)   # <— use the private attribute
        if proc:
            if proc.stdout:
                asyncio.create_task(pipe_stream(proc.stdout, "STDOUT"))
            if proc.stderr:
                asyncio.create_task(pipe_stream(proc.stderr, "STDERR"))
        # -------------------------------------------------------------------------

        srv_tools = await MCPUtil.get_function_tools(
            srv, convert_schemas_to_strict=True
        )
        for ft in srv_tools:
            # Detect collisions early
            if ft.name in tool_to_server:
                print(f"[⚠] Tool name collision: {ft.name}")
            tool_to_server[ft.name] = srv
            tools_json.append(
                {
                    "type": "function",
                    "function": {
                        "name": ft.name,
                        "description": ft.description,
                        "parameters": ft.params_json_schema,
                    },
                }
            )
        live_servers.append(srv)

    # Launch every server concurrently
    await asyncio.gather(*(boot(alias, spec) for alias, spec in cfg.items()))
    print(f"[MCP] Started {len(live_servers)} server(s) – {len(tools_json)} tool(s)")
    return tool_to_server, tools_json, live_servers


# ---------- Chat loop ----------
# Updated chat_loop signature and MCP setup
# ---------------------------------------------------------------------------
async def chat_loop(cid: str, mcp_cfg_path: Optional[str]):
    data, path = _load(cid)
    model = data.get("model", "gpt-4o")
    client = OpenAI()

    # --- Load System Prompt --- 
    system_prompt_content = None
    system_prompt_added_this_session = False # Flag to track if we print it
    if CONFIG_FILE.exists():
        try:
            config_data = json.loads(CONFIG_FILE.read_text())
            system_prompt_content = config_data.get("system_prompt") # Get original content
            # Removed print here, will be printed below if added
        except json.JSONDecodeError:
            print(f"[⚠] Error reading {CONFIG_FILE}")
        except Exception as e:
            print(f"[⚠] Error loading config {CONFIG_FILE}: {e}") # Keep non-colored

    # Prepend system prompt if needed
    if system_prompt_content and (not data["messages"] or data["messages"][0].get("role") != "system"):
        ts_sys = _now() # Timestamp for system message
        formatted_sys_content = f"[{ts_sys}] {system_prompt_content}"
        data["messages"].insert(0, {
            "role": "system", 
            "content": formatted_sys_content, # Timestamp prepended
            "timestamp": ts_sys             # Timestamp as metadata
        })
        system_prompt_added_this_session = True # Mark for printing
    # ---------------------------------------------------------

    # MCP setup (if requested)
    tool_to_server = {}
    tools_json = []
    live_servers = []

    if mcp_cfg_path:
        tool_to_server, tools_json, live_servers = await prepare_mcp_servers(
            Path(mcp_cfg_path).expanduser()
        )

    # --- Print Initial Chat Info ---
    print(clr(f"\n── Chat {cid} ({model}) – /exit to quit, /save to save ──\n", Style.BRIGHT, Fore.BLUE))
    if system_prompt_added_this_session:
         # Display the system prompt content (which now includes the timestamp)
         print(clr(f"[System] {data['messages'][0]['content']}", Fore.YELLOW))
         print() # Add a newline after system prompt

    loop = asyncio.get_event_loop()
    try:
        while True:
            # --- Get User Input ---
            try:
                user_input = await loop.run_in_executor(None, input, clr("› ", Fore.GREEN))
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Interrupted]"); break # Keep non-colored
            if user_input in {"/exit","exit","quit",":q"}: break
            if user_input == "/save": _save(data,path); print("[✓] Saved"); continue # Keep non-colored
            if not user_input: continue

            # --- Prepare and Store User Message ---
            ts_user = _now()
            formatted_user_content = f"[{ts_user}] {user_input}"
            print(clr(f"[User] {formatted_user_content}", Fore.GREEN)) # Display with timestamp
            print() # Add space after user message

            data["messages"].append({
                "role":"user",
                "content": formatted_user_content, # Timestamp prepended
                "timestamp": ts_user             # Timestamp as metadata
            })

            # --- Prepare for Assistant Response ---
            assistant_message_accumulator = defaultdict(str)
            tool_calls_accumulator = []
            current_tool_call_args = {}
            ts_assistant_initial = _now() # Timestamp before API call

            try:
                # --- Initial Assistant Response Stream ---
                print(clr("[Assistant]", Fore.CYAN), end=" ", flush=True) # Assistant label

                response_stream = client.chat.completions.create(
                    model=model,
                    messages=data["messages"], # Send history (now with timestamps in content)
                    tools=tools_json,
                    stream=True,
                )

                finish_reason = None
                # --- Process Stream --- 
                for chunk in response_stream:
                    delta = chunk.choices[0].delta
                    if chunk.choices[0].finish_reason is not None:
                        finish_reason = chunk.choices[0].finish_reason
                    if delta.content:
                        print(clr(delta.content, Fore.CYAN), end="", flush=True) 
                        assistant_message_accumulator["content"] += delta.content
                    if delta.tool_calls:
                        for tool_call_chunk in delta.tool_calls:
                            index = tool_call_chunk.index
                            # Initialize tool call structure if seeing this index for the first time
                            if index >= len(tool_calls_accumulator):
                                tool_calls_accumulator.append({
                                    "id": None, # Will be filled later if provided
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                                current_tool_call_args[index] = "" # Initialize accumulator for this index

                            # Safely access the current accumulator entry
                            current_call = tool_calls_accumulator[index]

                            # Update fields if present in the chunk
                            if hasattr(tool_call_chunk, 'id') and tool_call_chunk.id:
                                current_call["id"] = tool_call_chunk.id
                            if hasattr(tool_call_chunk, 'type') and tool_call_chunk.type:
                                current_call["type"] = tool_call_chunk.type # Though likely always 'function'
                            if tool_call_chunk.function:
                                if tool_call_chunk.function.name:
                                    current_call["function"]["name"] = tool_call_chunk.function.name
                                if tool_call_chunk.function.arguments:
                                    current_tool_call_args[index] += tool_call_chunk.function.arguments
                                    # Keep arguments accumulator separate until the end

                print() # Newline after assistant stream

                # --- Store Accumulated Assistant Message --- 
                # Consolidate final arguments after stream ends
                for i, call in enumerate(tool_calls_accumulator):
                   if i in current_tool_call_args:
                        call["function"]["arguments"] = current_tool_call_args[i]
                   # Assign a default ID if none was received
                   if call["id"] is None:
                       call["id"] = f"call_{uuid.uuid4().hex[:6]}"

                accumulated_content = assistant_message_accumulator["content"]
                # Prepend timestamp only if there's actual text content
                formatted_assistant_content = f"[{ts_assistant_initial}] {accumulated_content}" if accumulated_content else None

                assistant_message = {
                    "role": "assistant",
                    "content": formatted_assistant_content, # Timestamp prepended (if content exists)
                    "timestamp": ts_assistant_initial      # Timestamp as metadata
                }
                if tool_calls_accumulator:
                   assistant_message["tool_calls"] = tool_calls_accumulator # Tool calls don't get timestamp prepended
                
                data["messages"].append(assistant_message)

                # --- Handle Tool Calls ---
                if finish_reason == "tool_calls":
                    tool_messages_to_append = [] 
                    # ts_tools_start = _now() # Timestamp for the batch isn't strictly needed per message

                    for call in tool_calls_accumulator:
                        name = call["function"]["name"]
                        call_id = call["id"]
                        try:
                            # Ensure arguments are valid JSON before parsing
                            arguments_str = call["function"]["arguments"]
                            arguments = json.loads(arguments_str or "{}")
                        except json.JSONDecodeError:
                             # Keep error printing non-colored or use RED
                             print(clr(f"\n[!] Invalid JSON arguments received for tool {name}: {arguments_str}", Fore.RED))
                             tool_messages_to_append.append({
                                 "role": "tool",
                                 "tool_call_id": call_id,
                                 "content": f"Error: Invalid JSON arguments received: {arguments_str}",
                                 "timestamp": _now() # Timestamp for this specific error
                             })
                             continue # Skip this tool call

                        srv = tool_to_server.get(name)
                        if not srv:
                            print(clr(f"\n[✗] Unknown tool: {name}", Fore.RED))
                            tool_messages_to_append.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": f"Error: Unknown tool '{name}'",
                                "timestamp": _now() # Timestamp for this specific error
                            })
                            continue # Skip this tool call

                        # Print Formatted Tool Call
                        print(clr(f"\n[Tool Call: {srv.name}] {name}({arguments})", Fore.MAGENTA)) # Tool call style

                        # --- call the tool on the correct MCP server ---
                        try:
                            tool_result = await srv.call_tool(name, arguments)

                            # Flatten CallToolResult → plain string for the Chat API
                            # Ensure _to_text is defined correctly within or accessible to chat_loop
                            def _to_text(result: Any) -> str:
                                lines = []
                                for item in getattr(result, "content", []):
                                    if getattr(item, "type", None) == "text":
                                        lines.append(item.text)
                                    else:
                                        lines.append(str(item))
                                return "\n".join(lines) or "<empty tool result>"

                            result_text = _to_text(tool_result)
                            # Tool output itself is not printed directly, only the call is logged.
                            # The result is sent back to the model via the tool message.

                            # --- Prepare and Store Tool Result Message ---
                            ts_tool_result = _now() # Timestamp for this specific result
                            formatted_tool_content = f"[{ts_tool_result}] {result_text}"

                            tool_messages_to_append.append({
                                    "role": "tool",
                                    "tool_call_id": call_id,
                                    "content": formatted_tool_content, # Timestamp prepended
                                    "timestamp": ts_tool_result       # Timestamp as metadata
                                })
                        except Exception as tool_err:
                             print(clr(f"\n[!] Error calling tool {name}: {tool_err}", Fore.RED))
                             # Also add timestamp to error message content
                             ts_tool_error = _now()
                             error_content = f"Error executing tool '{name}': {tool_err}"
                             formatted_error_content = f"[{ts_tool_error}] {error_content}"
                             tool_messages_to_append.append({
                                 "role": "tool",
                                 "tool_call_id": call_id,
                                 "content": formatted_error_content, # Timestamp prepended
                                 "timestamp": ts_tool_error        # Timestamp as metadata
                             })

                    data["messages"].extend(tool_messages_to_append)

                    # --- Ask the model to continue (Streaming) ---
                    ts_assistant_final = _now() # Timestamp before follow-up call
                    print(clr("\n[Assistant]", Fore.CYAN), end=" ", flush=True) # Assistant label for follow-up

                    final_assistant_msg_content = "" # Accumulator for final content
                    followup_stream = client.chat.completions.create(
                        model=model,
                        messages=data["messages"], # Send history including tool results
                        stream=True,
                    )
                    # --- Process Follow-up Stream --- 
                    for chunk in followup_stream:
                         delta = chunk.choices[0].delta
                         if delta.content:
                              print(clr(delta.content, Fore.CYAN), end="", flush=True) # Stream content
                              final_assistant_msg_content += delta.content
                    
                    print() # Newline after final stream

                    # --- Store Final Assistant Message --- 
                    if final_assistant_msg_content:
                         formatted_final_content = f"[{ts_assistant_final}] {final_assistant_msg_content}"
                         data["messages"].append({
                             "role": "assistant",
                             "content": formatted_final_content, # Timestamp prepended
                             "timestamp": ts_assistant_final      # Timestamp as metadata
                         })

            except (RateLimitError, APITimeoutError, APIError) as e:
                print(clr(f"\n[ERROR] {e}\n", Fore.RED))
                # Avoid popping if messages might be partially processed due to stream error
                # Consider how to handle partial message saving or retries
            except Exception as e:
                 print(clr(f"\n[UNEXPECTED ERROR] {e}\n", Fore.RED))
                 # Consider logging traceback for unexpected errors
                 # import traceback; traceback.print_exc()


    finally:
        # Graceful cleanup
        if live_servers:
             print("[MCP] Cleaning up servers...")
             # Use return_exceptions=True for gather
             results = await asyncio.gather(*(srv.cleanup() for srv in live_servers), return_exceptions=True)
             for i, res in enumerate(results):
                 if isinstance(res, Exception):
                     print(f"[⚠] Error cleaning up server {live_servers[i].name}: {res}")
        _save(data, path) # Save the history with timestamps
        print(f"[✓] Saved to {path}")

# ---------- CLI ----------
def main() -> None:
    ap=argparse.ArgumentParser(description="Chat CLI w/ MCP tool support")
    sub=ap.add_subparsers(dest="cmd",required=True)
    p_new=sub.add_parser("new"); p_new.add_argument("title"); p_new.add_argument("--model",default="gpt-4o")
    sub.add_parser("list")
    p_chat=sub.add_parser("chat"); p_chat.add_argument("chat_id")
    p_del=sub.add_parser("delete"); p_del.add_argument("chat_id")
    # ---------------------------------------------------------------------------
    # Updated CLI arguments
    # ---------------------------------------------------------------------------
    for p in (p_new,p_chat):
        p.add_argument("--mcp-config", default="~/.mcp_servers.json",
                       help="Path to JSON config listing MCP servers")
    args=ap.parse_args()

    mcp_config_arg = getattr(args, 'mcp_config', None) # Get config path if present

    if args.cmd=="new":
        cid=create_chat(args.title,args.model)
        asyncio.run(chat_loop(cid, mcp_config_arg)) # Pass config path
    elif args.cmd=="list":
        for p in sorted(CHAT_DIR.glob("*.json")):
            d=json.loads(p.read_text())
            print(f"{d['id']} | {d['title']} | {d['model']} | {d['created_at']} | {len(d['messages'])} msgs")
    elif args.cmd=="chat":
        asyncio.run(chat_loop(args.chat_id, mcp_config_arg)) # Pass config path
    elif args.cmd=="delete":
        p=_chat_path(args.chat_id); p.unlink(missing_ok=True); print(f"[✓] Deleted {args.chat_id}")
    else: ap.print_help()

if __name__=="__main__":
    main()