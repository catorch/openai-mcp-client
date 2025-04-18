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
# ---------------------------------------------------------------------------
# Updated chat_loop signature and MCP setup
# ---------------------------------------------------------------------------
async def chat_loop(cid: str, mcp_cfg_path: Optional[str]):
    data, path = _load(cid)
    model = data.get("model", "gpt-4o")
    client = OpenAI()

    # --- Load System Prompt from config.json (if available) ---
    system_prompt = None
    if CONFIG_FILE.exists():
        try:
            config_data = json.loads(CONFIG_FILE.read_text())
            system_prompt = config_data.get("system_prompt")
            if system_prompt:
                print(f"[✓] Loaded system prompt from {CONFIG_FILE}")
        except json.JSONDecodeError:
            print(f"[⚠] Error reading {CONFIG_FILE}")
        except Exception as e:
            print(f"[⚠] Error loading config {CONFIG_FILE}: {e}")

    # Prepend system prompt if loaded and not already present
    if system_prompt and (not data["messages"] or data["messages"][0].get("role") != "system"):
        data["messages"].insert(0, {"role": "system", "content": system_prompt})
        print("[✓] Added system prompt to messages.")
    # ---------------------------------------------------------

    # MCP setup (if requested)
    tool_to_server = {}
    tools_json = []
    live_servers = []

    if mcp_cfg_path:
        tool_to_server, tools_json, live_servers = await prepare_mcp_servers(
            Path(mcp_cfg_path).expanduser()
        )

    # ① prompt banner (colored)
    print(clr(f"\n── Chat {cid} ({model}) – /exit to quit, /save to save ──\n",
              Style.BRIGHT, Fore.BLUE))
    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                # ② user prompt (colored)
                user_input = await loop.run_in_executor(None, input, clr("› ", Fore.GREEN))
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Interrupted]"); break
            if user_input in {"/exit","exit","quit",":q"}: break
            if user_input == "/save": _save(data,path); print("[✓] Saved"); continue
            if not user_input: continue

            data["messages"].append({"role":"user","content":user_input})

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=data["messages"],
                    tools=tools_json, # Use the combined tools list
                    stream=False,
                )
            except (RateLimitError, APITimeoutError, APIError) as e:
                print(f"\n[ERROR] {e}\n"); data["messages"].pop(); continue

            msg = response.choices[0].message
            # Handle tool calls (if any)
            if getattr(msg, "tool_calls", None):
                # IMPORTANT: Add the assistant message with tool_calls before tool responses
                data["messages"].append(msg.model_dump()) # Use model_dump() for Pydantic v2

                for call in msg.tool_calls:
                    if call.type!="function": continue
                    name = call.function.name
                    arguments = json.loads(call.function.arguments or "{}")
                    
                    # ---------------------------------------------------------------------------
                    # Updated tool call routing logic
                    # ---------------------------------------------------------------------------
                    srv = tool_to_server.get(name)
                    if not srv:
                        print(f"[✗] Unknown tool: {name}")
                        # Optionally send an error message back to the model?
                        # data["messages"].append({"role": "tool", "tool_call_id": call.id, "content": f"Error: Unknown tool '{name}'"})
                        continue # Skip this tool call

                    # ③ tool routing line (colored)
                    print(clr(f"[tool:{srv.name}] {name}({arguments})", Fore.MAGENTA))
                    
                    # --- call the tool on the correct MCP server ---
                    tool_result = await srv.call_tool(name, arguments)

                    # Flatten CallToolResult → plain string for the Chat API
                    def _to_text(result: Any) -> str:
                        lines = []
                        for item in getattr(result, "content", []):
                            # TextContent is the common case
                            if getattr(item, "type", None) == "text":
                                lines.append(item.text)
                            else:                       # fallback: stringify any other item
                                lines.append(str(item))
                        return "\n".join(lines) or "<empty tool result>"

                    result_text = _to_text(tool_result)

                    # Send the tool output back to the model
                    data["messages"].append(
                        {
                            "role": "tool",          # ← required role for tool responses
                            "tool_call_id": call.id,
                            "content": result_text,
                        }
                    )
                
                # Ask the model to continue the conversation
                followup = client.chat.completions.create(
                    model=model,
                    messages=data["messages"],
                    stream=False,
                )
                msg = followup.choices[0].message
            
            # ④ assistant reply (colored)
            print(clr(msg.content, Fore.CYAN))
            data["messages"].append({"role":"assistant","content":msg.content})
    finally:
        # ---------------------------------------------------------------------------
        # Updated cleanup logic
        # ---------------------------------------------------------------------------
        if live_servers:
             print("[MCP] Cleaning up servers...")
             await asyncio.gather(*(srv.cleanup() for srv in live_servers), return_exceptions=True)
        _save(data, path)
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