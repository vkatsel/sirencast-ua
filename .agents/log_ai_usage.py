#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-interaction hook for logging AI usage into ai_usage.md.
Captures user prompts, model responses, and artifact annotations with contextual excerpts.
- Clean hierarchy without emojis; blockquoted user prompts.
- Single-line interaction headers with branch and timestamp.
- Contextual multi-line excerpts for artifact annotations with line indicators.
- Artifact snapshots reconstructed at prompt time from transcript tool calls.
- Strictly preserves manual user edits/deletions in ai_usage.md (no unprompted restorations).
- Zero duplicate appending: each turn is logged exactly once upon completion at Stop event.
- Master archive maintained in .git/ai_usage_archive.md.
- Explicit restore and rebuild via CLI flags (--restore, --rebuild).
"""

import os
import sys
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


try:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


import time

def read_stdin_safely(timeout: float = 0.15) -> str:
    """Read stdin safely without blocking on interactive shell, supporting UTF-8 on Windows."""
    if sys.stdin.isatty():
        return ""
    if sys.platform == "win32":
        try:
            import msvcrt
            import ctypes
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            avail = ctypes.c_ulong()
            start = time.time()
            while time.time() - start < timeout:
                res = ctypes.windll.kernel32.PeekNamedPipe(
                    handle, None, 0, None, ctypes.byref(avail), None
                )
                if not res:
                    break
                if avail.value > 0:
                    break
                time.sleep(0.01)

            if avail.value > 0:
                chunks = []
                while True:
                    res = ctypes.windll.kernel32.PeekNamedPipe(
                        handle, None, 0, None, ctypes.byref(avail), None
                    )
                    if not res or avail.value == 0:
                        break
                    raw = sys.stdin.buffer.read(avail.value)
                    if not raw:
                        break
                    chunks.append(raw)
                    time.sleep(0.01)
                return b"".join(chunks).decode("utf-8", errors="replace")
            return ""
        except Exception:
            return ""
    else:
        try:
            import select
            if select.select([sys.stdin], [], [], timeout)[0]:
                return sys.stdin.read()
        except Exception:
            pass
    return ""


def safe_print_json(data: dict):
    """Output valid JSON response to stdout as required by Antigravity hook contract."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(data, ensure_ascii=False))


def get_git_branch(repo_dir: Path) -> str:
    """Get the active Git branch name or detached commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode == 0:
            branch = res.stdout.strip()
            if branch and branch != "HEAD":
                return branch
            commit_res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=False
            )
            if commit_res.returncode == 0:
                return f"detached-at-{commit_res.stdout.strip()}"
    except Exception:
        pass
    return "unknown-branch"


def extract_clean_prompt(raw_text: str) -> str:
    """Clean user prompt by removing Antigravity control tags and metadata blocks."""
    c = raw_text
    if "<USER_REQUEST>" in c:
        c = c.split("<USER_REQUEST>", 1)[1]
        if "</USER_REQUEST>" in c:
            c = c.split("</USER_REQUEST>", 1)[0]
    if "<ADDITIONAL_METADATA>" in c:
        c = c.split("<ADDITIONAL_METADATA>", 1)[0]
    if "<USER_SETTINGS_CHANGE>" in c:
        c = c.split("<USER_SETTINGS_CHANGE>", 1)[0]
    return c.strip()


def generate_excerpt(lines: list[str] | None, l1: int, l2: int, context: int = 2) -> str:
    """
    Generate a clean contextual excerpt around the targeted line(s) with line numbers
    and a '>' marker pointing to the annotated line(s).
    """
    if lines is None:
        return "(Artifact file not found for excerpt)"
    if not (1 <= l1 <= len(lines)):
        return f"(Line L{l1} is out of bounds for artifact with {len(lines)} lines)"

    start_idx = max(1, l1 - context)
    end_idx = min(len(lines), l2 + context)

    res = []
    for i in range(start_idx, end_idx + 1):
        marker = ">" if l1 <= i <= l2 else " "
        line_content = lines[i - 1]
        res.append(f"{marker} {i:4d} | {line_content}")
    return "\n".join(res)


def parse_artifact_remarks(prompt_text: str, artifacts_snapshot: dict[str, list[str]], artifact_dirs: list[Path]):
    """
    Extract user remarks on artifacts with contextual excerpts around the referenced lines.
    Uses the artifact snapshot at prompt time or falls back to disk.
    """
    remarks = []

    header_pattern = re.compile(
        r'(?:^|\n)\s*\[([a-zA-Z0-9_\-\.\/\\]+\.(?:md|markdown|py|txt|json|ya?ml))\]',
        re.IGNORECASE
    )
    matches = list(header_pattern.finditer(prompt_text))

    for idx, match in enumerate(matches):
        filename = Path(match.group(1)).name
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(prompt_text)
        block_text = prompt_text[start_pos:end_pos]

        # Resolve artifact lines
        lines = artifacts_snapshot.get(filename)
        if lines is None:
            for adir in artifact_dirs:
                if not adir.exists():
                    continue
                art_file = adir / filename
                if art_file.exists():
                    try:
                        lines = art_file.read_text(encoding="utf-8", errors="replace").splitlines()
                        artifacts_snapshot[filename] = lines
                        break
                    except Exception:
                        pass

        line_pattern = re.compile(
            r'(?:L|Line\s*)(\d+)(?:\s*(?:-|–|to)\s*(?:L|Line\s*)?(\d+))?:\s*(.*?)(?=(?:\n\s*(?:L|Line\s*)\d+)|\Z)',
            re.DOTALL | re.IGNORECASE
        )

        item_remarks = []
        for lm in line_pattern.finditer(block_text):
            l1 = int(lm.group(1))
            l2 = int(lm.group(2)) if lm.group(2) else l1
            comment = lm.group(3).strip()
            excerpt = generate_excerpt(lines, l1, l2, context=2)

            item_remarks.append({
                "line_start": l1,
                "line_end": l2,
                "comment": comment,
                "excerpt": excerpt
            })

        if item_remarks:
            remarks.append({
                "artifact": filename,
                "remarks": item_remarks
            })

    approved_pattern = re.compile(
        r'(?:^|\n)\s*\[Approved\]\s*([a-zA-Z0-9_\-\.\/\\]+\.(?:md|markdown|py|txt|json|ya?ml))(?:\s+(.*?))?(?=\n\s*\[|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    for match in approved_pattern.finditer(prompt_text):
        filename = Path(match.group(1)).name
        note = match.group(2).strip() if match.group(2) else "Approved without additional inline remarks"
        if not any(r["artifact"] == filename for r in remarks):
            remarks.append({
                "artifact": filename,
                "remarks": [{
                    "line_start": 0,
                    "line_end": 0,
                    "comment": note,
                    "excerpt": None
                }]
            })

    return remarks


def load_transcript_records(transcript_path: Path):
    """Load transcript records, preferring transcript_full.jsonl to avoid truncation."""
    target_file = transcript_path
    full_file = transcript_path.parent / "transcript_full.jsonl"

    if full_file.exists():
        target_file = full_file

    if not target_file.exists():
        target_file = transcript_path.parent / "transcript.jsonl"
        if not target_file.exists():
            return []

    records = []
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
    except Exception:
        pass

    return records


def parse_completed_turns(records, artifact_dirs: list[Path]):
    """
    Parse complete user-model turns from transcript records, maintaining an active
    artifacts snapshot at the exact moment of each user turn and capturing the final model response.
    """
    turns = []
    current_turn = None
    artifacts_tracker: dict[str, list[str]] = {}

    for r in records:
        source = r.get("source")
        stype = r.get("type")
        step = r.get("step_index")
        content = r.get("content", "")
        created_at = r.get("created_at", "")

        # Track artifact modifications across model tool calls
        if source == "MODEL":
            for tc in r.get("tool_calls", []):
                name = tc.get("name", "")
                args = tc.get("args", {})
                if name == "write_to_file":
                    tgt = args.get("TargetFile", "")
                    code_content = args.get("CodeContent")
                    if tgt and code_content is not None:
                        fn = Path(tgt).name
                        artifacts_tracker[fn] = code_content.splitlines()
                elif name == "replace_file_content":
                    tgt = args.get("TargetFile", "")
                    target_content = args.get("TargetContent", "")
                    replacement = args.get("ReplacementContent", "")
                    if tgt:
                        fn = Path(tgt).name
                        if fn in artifacts_tracker and target_content:
                            joined = "\n".join(artifacts_tracker[fn])
                            if target_content in joined:
                                joined = joined.replace(target_content, replacement, 1)
                                artifacts_tracker[fn] = joined.splitlines()

        is_user = (
            (source == "USER_EXPLICIT" or stype == "USER_INPUT")
            and not content.startswith("The following is a <SYSTEM_MESSAGE>")
            and not content.startswith("[Message] timestamp=")
        )

        if is_user:
            if current_turn and (current_turn.get("final_response") or current_turn.get("last_content")):
                current_turn["model_response"] = current_turn.get("final_response") or current_turn.get("last_content")
                turns.append(current_turn)

            clean_prompt = extract_clean_prompt(content)
            # Freeze snapshot of artifacts as they existed when user submitted prompt
            snapshot = {k: list(v) for k, v in artifacts_tracker.items()}
            art_remarks = parse_artifact_remarks(clean_prompt, snapshot, artifact_dirs)

            current_turn = {
                "user_step": step,
                "created_at": created_at,
                "prompt": clean_prompt,
                "artifact_remarks": art_remarks,
                "model_response": None,
                "final_response": None,
                "last_content": None,
                "model_step": None
            }
        elif current_turn is not None and source == "MODEL" and stype == "PLANNER_RESPONSE":
            if content and content.strip():
                clean_c = content.strip()
                if not r.get("tool_calls"):
                    current_turn["final_response"] = clean_c
                    current_turn["model_step"] = step
                else:
                    current_turn["last_content"] = clean_c

    if current_turn and (current_turn.get("final_response") or current_turn.get("last_content")):
        current_turn["model_response"] = current_turn.get("final_response") or current_turn.get("last_content")
        turns.append(current_turn)

    return turns


def wrap_in_code_block(text: str, lang: str = "text") -> str:
    """Wrap text in a fenced code block with dynamic backticks count to prevent formatting conflicts."""
    if not text:
        return f"```{lang}\n(Empty)\n```"
    backtick_matches = re.findall(r"`+", text)
    max_ticks = max([len(m) for m in backtick_matches] + [2])
    fence = "`" * max(max_ticks + 1, 3)
    return f"{fence}{lang}\n{text}\n{fence}"


def format_turn_markdown(turn: dict, turn_num: int, branch: str) -> str:
    """
    Format a turn into clean Markdown.
    Single-line metadata header. Prompts and responses wrapped in fenced code blocks.
    Artifact annotations include contextual multi-line excerpts with line indicators.
    """
    created_at = turn.get("created_at", "")
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            time_str = created_at
    else:
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    user_step = turn.get("user_step", 0)
    prompt = turn.get("prompt", "").strip()
    model_response = turn.get("model_response", "").strip()
    art_remarks = turn.get("artifact_remarks", [])

    lines = []
    lines.append(f"### [Interaction #{turn_num}] · {time_str} · branch: [{branch}] · step: [{user_step}]")
    lines.append("")

    lines.append("#### [User Prompt]")
    lines.append(wrap_in_code_block(prompt if prompt else "(Empty prompt)", lang="text"))
    lines.append("")

    if art_remarks:
        lines.append("#### [Artifact Annotations]")
        for ar in art_remarks:
            art_name = ar["artifact"]
            lines.append(f"- **Artifact:** `[{art_name}]`")
            for rem in ar["remarks"]:
                line_start = rem["line_start"]
                line_end = rem["line_end"]
                comment = rem["comment"]
                excerpt = rem["excerpt"]

                if line_start > 0:
                    line_tag = f"Line L{line_start}" if line_start == line_end else f"Lines L{line_start}-L{line_end}"
                    lines.append(f"  - **[{line_tag}]:** {comment}")
                    if excerpt:
                        lines.append("    ```text")
                        for eline in excerpt.splitlines():
                            lines.append(f"    {eline}")
                        lines.append("    ```")
                else:
                    lines.append(f"  - {comment}")
        lines.append("")

    lines.append("#### [Model Response]")
    lines.append(wrap_in_code_block(model_response if model_response else "(No response text)", lang="text"))
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def get_paths(stdin_payload: dict):
    """Resolve all project, git, brain, and transcript paths."""
    if stdin_payload.get("workspacePaths"):
        workspace_dir = Path(stdin_payload["workspacePaths"][0]).resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        if script_dir.name == ".agents":
            workspace_dir = script_dir.parent
        elif (script_dir.parent / ".agents").exists():
            workspace_dir = script_dir.parent
        else:
            workspace_dir = Path(os.getcwd()).resolve()

    git_dir = workspace_dir / ".git"
    current_branch = get_git_branch(workspace_dir)

    conv_id = stdin_payload.get("conversationId")
    transcript_path_str = stdin_payload.get("transcriptPath")
    art_dir_str = stdin_payload.get("artifactDirectoryPath")

    brain_base = Path.home() / ".gemini" / "antigravity-cli" / "brain"

    if not conv_id:
        if brain_base.exists():
            candidates = []
            for d in brain_base.iterdir():
                if not d.is_dir():
                    continue
                tf = d / ".system_generated" / "logs" / "transcript.jsonl"
                tf_full = d / ".system_generated" / "logs" / "transcript_full.jsonl"
                active_tf = tf_full if tf_full.exists() else tf
                if active_tf.exists():
                    try:
                        mtime = active_tf.stat().st_mtime
                        with open(active_tf, "r", encoding="utf-8", errors="ignore") as f:
                            chunk = f.read(50000)
                        matches_ws = str(workspace_dir).lower() in chunk.lower() or workspace_dir.name.lower() in chunk.lower()
                        if matches_ws:
                            candidates.append((d.name, mtime))
                    except Exception:
                        pass
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                conv_id = candidates[0][0]
            else:
                conv_id = "default-chat"
        else:
            conv_id = "default-chat"

    chat_hash = conv_id[:8] if len(conv_id) >= 8 else conv_id

    if transcript_path_str:
        transcript_path = Path(transcript_path_str).resolve()
    else:
        transcript_path = brain_base / conv_id / ".system_generated" / "logs" / "transcript_full.jsonl"

    artifact_dirs = []
    if art_dir_str:
        artifact_dirs.append(Path(art_dir_str).resolve())
    artifact_dirs.append(brain_base / conv_id)
    artifact_dirs.append(workspace_dir)

    if git_dir.exists():
        state_file = git_dir / "ai_usage_state.json"
        archive_file = git_dir / "ai_usage_archive.md"
    else:
        state_dir = workspace_dir / ".agents"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "ai_usage_state.json"
        archive_file = state_dir / "ai_usage_archive.md"

    workspace_log_file = workspace_dir / "ai_usage.md"

    return {
        "workspace_dir": workspace_dir,
        "git_dir": git_dir,
        "current_branch": current_branch,
        "conv_id": conv_id,
        "chat_hash": chat_hash,
        "transcript_path": transcript_path,
        "artifact_dirs": artifact_dirs,
        "state_file": state_file,
        "archive_file": archive_file,
        "workspace_log_file": workspace_log_file
    }


def extract_session_steps_from_markdown(md_text: str) -> dict[str, set[int]]:
    """Extract which steps are currently present in markdown grouped by session ID."""
    session_blocks = re.split(r'(?=\n##\s*\[Chat Session\s*#[^\]]+\])', md_text)
    mapping = {}
    for block in session_blocks:
        conv_m = re.search(r'\*\*Session ID:\*\*\s*`([^`]+)`', block)
        if conv_m:
            cid = conv_m.group(1).strip()
            steps = set(int(m) for m in re.findall(r'step:\s*\[(\d+)\]', block))
            mapping[cid] = steps
    return mapping


def update_turn_response_in_markdown(md_text: str, conv_id: str, user_step: int, new_model_response: str) -> str:
    """Update #### [Model Response] for an existing interaction with the given user_step within conv_id session."""
    session_pattern = re.compile(
        rf'(##\s*\[Chat Session\s*#[^\]]+\]\s*\n[^\n]*\*\*Session ID:\*\*\s*`{re.escape(conv_id)}`.*?)'
        rf'(?=\n##\s*\[Chat Session|\Z)',
        re.DOTALL
    )
    sm = session_pattern.search(md_text)
    if not sm:
        target_block = md_text
        block_start = 0
        block_end = len(md_text)
    else:
        target_block = sm.group(1)
        block_start = sm.start(1)
        block_end = sm.end(1)

    turn_pattern = re.compile(
        rf'(###\s*\[Interaction\s*#\d+\][^\n]*step:\s*\[{user_step}\].*?####\s*\[Model Response\]\s*\n)(`{{3,}}[^\n]*\n.*?\n`{{3,}}|\(No response text\))(.*?)(?=\s*\n---|\s*\Z)',
        re.DOTALL
    )
    tm = turn_pattern.search(target_block)
    if not tm:
        turn_pattern_fallback = re.compile(
            rf'(###\s*\[Interaction\s*#\d+\][^\n]*step:\s*\[{user_step}\].*?####\s*\[Model Response\]\s*\n)(.*?)(?=\s*\n---|\s*\Z)',
            re.DOTALL
        )
        tm = turn_pattern_fallback.search(target_block)
        if not tm:
            return md_text

    new_wrapped = wrap_in_code_block(new_model_response, lang="text")
    if tm.group(2).strip() == new_wrapped.strip():
        return md_text

    updated_block = target_block[:tm.start(2)] + new_wrapped + target_block[tm.end(2):]
    return md_text[:block_start] + updated_block + md_text[block_end:]


def handle_restore(paths: dict):
    """Restore ai_usage.md from the full unedited archive upon explicit user command."""
    archive_file = paths["archive_file"]
    workspace_log = paths["workspace_log_file"]
    if not archive_file.exists():
        print(f"Archive file {archive_file} does not exist. Cannot restore.")
        return

    # Create safety backup of current workspace log
    if workspace_log.exists():
        bak = paths["workspace_dir"] / f"ai_usage.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        shutil.copy2(workspace_log, bak)
        print(f"Safety backup created at {bak.name}")

    shutil.copy2(archive_file, workspace_log)

    state_file = paths["state_file"]
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            state["deleted_turns"] = {}
            state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    print("ai_usage.md successfully restored from master archive.")


def handle_rebuild(paths: dict):
    """Rebuild ai_usage.md and archive from scratch using all transcript turns."""
    records = load_transcript_records(paths["transcript_path"])
    turns = parse_completed_turns(records, paths["artifact_dirs"])
    if not turns:
        print("No completed turns found to rebuild.")
        return

    header_content = (
        "# [AI Usage Log]\n\n"
        "> Automatically generated by Antigravity post-interaction hook.\n\n"
        "---\n\n"
    )

    chunks = [header_content]
    first_time = turns[0].get("created_at", "")
    if first_time:
        try:
            dt = datetime.fromisoformat(first_time.replace("Z", "+00:00"))
            f_time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            f_time_str = first_time
    else:
        f_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    chat_header = (
        f"## [Chat Session #{paths['chat_hash']}]\n"
        f"- **Session ID:** `{paths['conv_id']}`\n"
        f"- **Started:** {f_time_str}\n"
        f"- **Git Branch:** `{paths['current_branch']}`\n\n"
    )
    chunks.append(chat_header)

    logged_steps = []
    for idx, turn in enumerate(turns, 1):
        chunks.append(format_turn_markdown(turn, idx, paths["current_branch"]))
        logged_steps.append(turn["user_step"])

    full_text = "".join(chunks)

    # Write to workspace log and master archive
    paths["workspace_log_file"].write_text(full_text, encoding="utf-8")
    paths["archive_file"].write_text(full_text, encoding="utf-8")

    state = {
        "last_conversation_id": paths["conv_id"],
        "logged_turns": {paths["conv_id"]: logged_steps},
        "deleted_turns": {},
        "turn_counter": len(turns)
    }
    paths["state_file"].write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rebuilt ai_usage.md with {len(turns)} turns successfully.")


def main():
    # Handle manual CLI commands
    if "--restore" in sys.argv:
        paths = get_paths({})
        handle_restore(paths)
        return
    elif "--rebuild" in sys.argv:
        paths = get_paths({})
        handle_rebuild(paths)
        return

    # Normal hook execution (only triggered at Stop event)
    raw_input = read_stdin_safely()
    stdin_payload = {}
    if raw_input.strip():
        try:
            stdin_payload = json.loads(raw_input)
        except Exception:
            pass

    paths = get_paths(stdin_payload)
    conv_id = paths["conv_id"]
    chat_hash = paths["chat_hash"]
    current_branch = paths["current_branch"]
    state_file = paths["state_file"]
    archive_file = paths["archive_file"]
    workspace_log_file = paths["workspace_log_file"]

    # Load state
    state = {
        "last_conversation_id": None,
        "logged_turns": {},
        "deleted_turns": {},
        "turn_counter": 0
    }
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    header_content = (
        "# [AI Usage Log]\n\n"
        "> Automatically generated by Antigravity post-interaction hook.\n\n"
        "---\n\n"
    )

    # Ensure archive has header
    if not archive_file.exists() or archive_file.stat().st_size == 0:
        archive_file.write_text(header_content, encoding="utf-8")

    # If workspace log does not exist or is empty, write header ONLY.
    if not workspace_log_file.exists() or workspace_log_file.stat().st_size == 0:
        workspace_log_file.write_text(header_content, encoding="utf-8")

    current_workspace_content = workspace_log_file.read_text(encoding="utf-8")

    # Detect turns manually removed by the user in ai_usage.md ONLY for sessions actually present in the file
    existing_session_steps = extract_session_steps_from_markdown(current_workspace_content)
    if "deleted_turns" not in state:
        state["deleted_turns"] = {}

    for cid in existing_session_steps:
        present_in_file = existing_session_steps[cid]
        previously_logged = set(state.get("logged_turns", {}).get(cid, []))
        for st in previously_logged:
            if st not in present_in_file:
                if cid not in state["deleted_turns"]:
                    state["deleted_turns"][cid] = []
                if st not in state["deleted_turns"][cid]:
                    state["deleted_turns"][cid].append(st)

    # 1. Update any existing turns across all known sessions whose responses improved/completed
    updated_workspace_content = current_workspace_content
    brain_base = Path.home() / ".gemini" / "antigravity-cli" / "brain"

    sessions_to_sync = set(existing_session_steps.keys())
    if conv_id:
        sessions_to_sync.add(conv_id)

    for cid in sessions_to_sync:
        if cid in existing_session_steps:
            cid_full = brain_base / cid / ".system_generated" / "logs" / "transcript_full.jsonl"
            cid_norm = brain_base / cid / ".system_generated" / "logs" / "transcript.jsonl"
            cid_path = cid_full if cid_full.exists() else cid_norm
            if cid_path.exists():
                cid_records = load_transcript_records(cid_path)
                cid_turns = parse_completed_turns(cid_records, paths["artifact_dirs"])
                for turn in cid_turns:
                    u_step = turn["user_step"]
                    if u_step in existing_session_steps[cid]:
                        new_resp = (turn.get("model_response") or "").strip()
                        if new_resp:
                            updated_workspace_content = update_turn_response_in_markdown(
                                updated_workspace_content, cid, u_step, new_resp
                            )

    if updated_workspace_content != current_workspace_content:
        workspace_log_file.write_text(updated_workspace_content, encoding="utf-8")
        current_workspace_content = updated_workspace_content
        if archive_file.exists():
            arc_content = archive_file.read_text(encoding="utf-8")
            for cid in sessions_to_sync:
                if cid in existing_session_steps:
                    cid_full = brain_base / cid / ".system_generated" / "logs" / "transcript_full.jsonl"
                    cid_norm = brain_base / cid / ".system_generated" / "logs" / "transcript.jsonl"
                    cid_path = cid_full if cid_full.exists() else cid_norm
                    if cid_path.exists():
                        cid_records = load_transcript_records(cid_path)
                        cid_turns = parse_completed_turns(cid_records, paths["artifact_dirs"])
                        for turn in cid_turns:
                            u_step = turn["user_step"]
                            new_resp = (turn.get("model_response") or "").strip()
                            if new_resp:
                                arc_content = update_turn_response_in_markdown(
                                    arc_content, cid, u_step, new_resp
                                )
            archive_file.write_text(arc_content, encoding="utf-8")

    # 2. Append new completed turns for current session
    logged_steps_for_conv = set(state.get("logged_turns", {}).get(conv_id, []))
    deleted_steps_for_conv = set(state.get("deleted_turns", {}).get(conv_id, []))
    present_steps_in_this_conv = existing_session_steps.get(conv_id, set())

    # Parse completed turns from transcript
    records = load_transcript_records(paths["transcript_path"])
    completed_turns = parse_completed_turns(records, paths["artifact_dirs"])

    # New turns exclude already logged turns AND manually deleted turns AND already present turns
    all_seen_steps = logged_steps_for_conv | deleted_steps_for_conv | present_steps_in_this_conv
    new_turns = [t for t in completed_turns if t["user_step"] not in all_seen_steps]

    if new_turns:
        # Number interactions following the file's current numbering
        existing_turn_nums = [
            int(m) for m in re.findall(r'###\s*\[Interaction\s*#(\d+)\]', current_workspace_content)
        ]
        current_turn_counter = max(existing_turn_nums) if existing_turn_nums else 0

        new_text_chunks = []
        last_logged_conv = state.get("last_conversation_id")
        has_session_header = f"## [Chat Session #{chat_hash}]" in current_workspace_content

        for turn in new_turns:
            if conv_id != last_logged_conv or not has_session_header:
                first_time = turn.get("created_at", "")
                if first_time:
                    try:
                        dt = datetime.fromisoformat(first_time.replace("Z", "+00:00"))
                        f_time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                    except Exception:
                        f_time_str = first_time
                else:
                    f_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

                chat_header = (
                    f"## [Chat Session #{chat_hash}]\n"
                    f"- **Session ID:** `{conv_id}`\n"
                    f"- **Started:** {f_time_str}\n"
                    f"- **Git Branch:** `{current_branch}`\n\n"
                )
                new_text_chunks.append(chat_header)
                last_logged_conv = conv_id
                has_session_header = True

            current_turn_counter += 1
            turn_md = format_turn_markdown(turn, current_turn_counter, current_branch)
            new_text_chunks.append(turn_md)

            logged_steps_for_conv.add(turn["user_step"])

        combined_new_text = "".join(new_text_chunks)

        # 1. Append to master archive (safety backup)
        with open(archive_file, "a", encoding="utf-8") as f:
            f.write(combined_new_text)

        # 2. Append to workspace log (respecting manual user edits)
        with open(workspace_log_file, "a", encoding="utf-8") as f:
            f.write(combined_new_text)

        state["turn_counter"] = current_turn_counter
        state["last_conversation_id"] = last_logged_conv

    if "logged_turns" not in state:
        state["logged_turns"] = {}
    state["logged_turns"][conv_id] = sorted(list(logged_steps_for_conv | present_steps_in_this_conv))

    if "deleted_turns" not in state:
        state["deleted_turns"] = {}
    state["deleted_turns"][conv_id] = sorted(list(deleted_steps_for_conv))

    try:
        state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    safe_print_json({"decision": "allow", "terminationBehavior": ""})


if __name__ == "__main__":
    main()
