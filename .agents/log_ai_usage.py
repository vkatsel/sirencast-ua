#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-interaction hook for logging AI usage into ai_usage.md.
Captures user prompts, model responses, and artifact annotations with snippets.
- Clean hierarchy without emojis; blockquoted user prompts.
- Single-line interaction headers with branch and timestamp.
- Resistant to branch changes and multi-chat switching.
- Preserves manual user edits/deletions in ai_usage.md.
- Full backup archive maintained in .git/ai_usage_archive.md.
- Supports --restore and --rebuild CLI flags.
"""

import os
import sys
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def read_stdin_safely() -> str:
    """Read stdin if data is available, without blocking on Windows or Unix."""
    if sys.stdin.isatty():
        return ""

    if sys.platform == "win32":
        try:
            import msvcrt
            import ctypes
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            avail = ctypes.c_ulong()
            res = ctypes.windll.kernel32.PeekNamedPipe(
                handle, None, 0, None, ctypes.byref(avail), None
            )
            if res and avail.value > 0:
                return sys.stdin.read()
            return ""
        except Exception:
            pass
    else:
        try:
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
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
    print(json.dumps(data))


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


def parse_artifact_remarks(prompt_text: str, artifact_dirs: list[Path]):
    """Extract user remarks on artifacts with corresponding line snippets."""
    remarks = []

    artifact_block_pattern = re.compile(
        r'\[([a-zA-Z0-9_\-\.\/\\]+\.md)\](?:\s*\n|\s+)((?:(?:L|Line\s*)\d+:.*(?:\n|$))+)',
        re.IGNORECASE
    )
    for match in artifact_block_pattern.finditer(prompt_text):
        filename = Path(match.group(1)).name
        lines_block = match.group(2)
        line_pattern = re.compile(
            r'(?:L|Line\s*)(\d+):\s*(.*?)(?=\n(?:L|Line\s*)\d+:|$)',
            re.DOTALL | re.IGNORECASE
        )

        artifact_content = None
        for adir in artifact_dirs:
            if not adir.exists():
                continue
            art_file = adir / filename
            if art_file.exists():
                try:
                    artifact_content = art_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    break
                except Exception:
                    pass

        item_remarks = []
        for lm in line_pattern.finditer(lines_block):
            line_no = int(lm.group(1))
            comment = lm.group(2).strip()
            snippet = None
            if artifact_content and 1 <= line_no <= len(artifact_content):
                snippet = artifact_content[line_no - 1].strip()
            item_remarks.append({
                "line": line_no,
                "comment": comment,
                "snippet": snippet
            })

        if item_remarks:
            remarks.append({
                "artifact": filename,
                "remarks": item_remarks
            })

    approved_pattern = re.compile(r'\[Approved\]\s*([a-zA-Z0-9_\-\.\/\\]+\.md)', re.IGNORECASE)
    for match in approved_pattern.finditer(prompt_text):
        filename = Path(match.group(1)).name
        if not any(r["artifact"] == filename for r in remarks):
            remarks.append({
                "artifact": filename,
                "remarks": [{
                    "line": 0,
                    "comment": "Approved without additional inline remarks",
                    "snippet": None
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
    """Parse complete user-model turns from transcript records."""
    turns = []
    current_turn = None

    for r in records:
        source = r.get("source")
        stype = r.get("type")
        step = r.get("step_index")
        content = r.get("content", "")
        created_at = r.get("created_at", "")

        is_user = (
            (source == "USER_EXPLICIT" or stype == "USER_INPUT")
            and not content.startswith("The following is a <SYSTEM_MESSAGE>")
            and not content.startswith("[Message] timestamp=")
        )

        if is_user:
            if current_turn and (current_turn.get("model_response") or current_turn.get("interim_response")):
                if not current_turn.get("model_response"):
                    current_turn["model_response"] = current_turn.get("interim_response")
                turns.append(current_turn)

            clean_prompt = extract_clean_prompt(content)
            art_remarks = parse_artifact_remarks(content, artifact_dirs)

            current_turn = {
                "user_step": step,
                "created_at": created_at,
                "prompt": clean_prompt,
                "artifact_remarks": art_remarks,
                "model_response": None,
                "interim_response": None,
                "model_step": None
            }
        elif current_turn is not None and source == "MODEL" and stype == "PLANNER_RESPONSE":
            if content:
                if not r.get("tool_calls"):
                    current_turn["model_response"] = content
                    current_turn["model_step"] = step
                else:
                    current_turn["interim_response"] = content

    if current_turn and (current_turn.get("model_response") or current_turn.get("interim_response")):
        if not current_turn.get("model_response"):
            current_turn["model_response"] = current_turn.get("interim_response")
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
    Single-line metadata header. Prompts and responses wrapped in fenced code blocks
    so they don't break markdown layout.
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
    # Single-line compact header with turn number, timestamp, branch, and step in brackets
    lines.append(f"### [Interaction #{turn_num}] · {time_str} · branch: [{branch}] · step: [{user_step}]")
    lines.append("")

    # Raw fenced text block for user prompt
    lines.append("#### [User Prompt]")
    lines.append(wrap_in_code_block(prompt if prompt else "(Empty prompt)", lang="text"))
    lines.append("")

    # Artifact annotations if present
    if art_remarks:
        lines.append("#### [Artifact Annotations]")
        for ar in art_remarks:
            art_name = ar["artifact"]
            lines.append(f"- **Artifact:** `[{art_name}]`")
            for rem in ar["remarks"]:
                line_no = rem["line"]
                comment = rem["comment"]
                snippet = rem["snippet"]

                if line_no > 0:
                    lines.append(f"  - **[Line L{line_no}]:** {comment}")
                    if snippet:
                        lines.append("    ```text")
                        lines.append(f"    {snippet}")
                        lines.append("    ```")
                else:
                    lines.append(f"  - {comment}")
        lines.append("")

    # Raw fenced text block for model response
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
            try:
                subdirs = sorted(
                    [d for d in brain_base.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                    reverse=True
                )
                if subdirs:
                    conv_id = subdirs[0].name
            except Exception:
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

    # Persistence files inside .git to survive branch changes
    if git_dir.exists():
        state_file = git_dir / "ai_usage_state.json"
        archive_file = git_dir / "ai_usage_archive.md"
        active_copy_file = git_dir / "ai_usage_active.md"
    else:
        state_dir = workspace_dir / ".agents"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "ai_usage_state.json"
        archive_file = state_dir / "ai_usage_archive.md"
        active_copy_file = state_dir / "ai_usage_active.md"

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
        "active_copy_file": active_copy_file,
        "workspace_log_file": workspace_log_file
    }


def handle_restore(paths: dict):
    """Restore ai_usage.md from the full unedited archive."""
    archive_file = paths["archive_file"]
    workspace_log = paths["workspace_log_file"]
    if not archive_file.exists():
        print(f"Archive file {archive_file} does not exist. Cannot restore.")
        return

    # Create a quick safety backup of current workspace log if it exists
    if workspace_log.exists():
        bak = paths["workspace_dir"] / f"ai_usage.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        shutil.copy2(workspace_log, bak)
        print(f"Safety backup created at {bak.name}")

    shutil.copy2(archive_file, workspace_log)
    if paths["active_copy_file"]:
        shutil.copy2(archive_file, paths["active_copy_file"])
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

    # Write to workspace log, archive, and active copy
    paths["workspace_log_file"].write_text(full_text, encoding="utf-8")
    paths["archive_file"].write_text(full_text, encoding="utf-8")
    paths["active_copy_file"].write_text(full_text, encoding="utf-8")

    state = {
        "last_conversation_id": paths["conv_id"],
        "logged_turns": {paths["conv_id"]: logged_steps},
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

    # Normal hook execution
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
    active_copy_file = paths["active_copy_file"]
    workspace_log_file = paths["workspace_log_file"]

    # Load state
    state = {
        "last_conversation_id": None,
        "logged_turns": {},
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

    # If workspace log does not exist at all, initialize it from active_copy or archive
    if not workspace_log_file.exists() or workspace_log_file.stat().st_size == 0:
        if active_copy_file.exists() and active_copy_file.stat().st_size > 0:
            workspace_log_file.write_text(active_copy_file.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            workspace_log_file.write_text(archive_file.read_text(encoding="utf-8"), encoding="utf-8")

    # Parse turns from transcript
    records = load_transcript_records(paths["transcript_path"])
    completed_turns = parse_completed_turns(records, paths["artifact_dirs"])

    logged_steps_for_conv = set(state.get("logged_turns", {}).get(conv_id, []))
    new_turns = [t for t in completed_turns if t["user_step"] not in logged_steps_for_conv]

    if new_turns:
        new_text_chunks = []
        last_logged_conv = state.get("last_conversation_id")

        for turn in new_turns:
            # Check if chat changed or first turn under this session
            if conv_id != last_logged_conv:
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

            state["turn_counter"] = state.get("turn_counter", 0) + 1
            turn_md = format_turn_markdown(turn, state["turn_counter"], current_branch)
            new_text_chunks.append(turn_md)

            logged_steps_for_conv.add(turn["user_step"])

        combined_new_text = "".join(new_text_chunks)

        # 1. Always append to full master archive (safety net)
        with open(archive_file, "a", encoding="utf-8") as f:
            f.write(combined_new_text)

        # 2. Append to workspace log (respects whatever user edited before this point)
        with open(workspace_log_file, "a", encoding="utf-8") as f:
            f.write(combined_new_text)

        # 3. Save a copy of user's active state in .git to restore on checkout
        try:
            active_copy_file.write_text(workspace_log_file.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

        # 4. Update state
        if "logged_turns" not in state:
            state["logged_turns"] = {}
        state["logged_turns"][conv_id] = sorted(list(logged_steps_for_conv))
        state["last_conversation_id"] = last_logged_conv

        try:
            state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    else:
        # If no new turns, but user edited workspace_log_file, sync active_copy_file
        try:
            if workspace_log_file.exists():
                active_copy_file.write_text(workspace_log_file.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    # Ensure git post-checkout hook restores active_copy_file on branch switches
    if paths["git_dir"].exists():
        try:
            hooks_dir = paths["git_dir"] / "hooks"
            hooks_dir.mkdir(parents=True, exist_ok=True)
            post_checkout = hooks_dir / "post-checkout"
            hook_script = (
                "#!/bin/sh\n"
                "# Restore user active ai_usage.md upon checkout\n"
                "GIT_DIR=\"$(git rev-parse --git-dir)\"\n"
                "WORK_TREE=\"$(git rev-parse --show-toplevel)\"\n"
                "if [ -f \"$GIT_DIR/ai_usage_active.md\" ]; then\n"
                "  cp \"$GIT_DIR/ai_usage_active.md\" \"$WORK_TREE/ai_usage.md\"\n"
                "elif [ -f \"$GIT_DIR/ai_usage_archive.md\" ]; then\n"
                "  cp \"$GIT_DIR/ai_usage_archive.md\" \"$WORK_TREE/ai_usage.md\"\n"
                "fi\n"
                "exit 0\n"
            )
            post_checkout.write_text(hook_script, encoding="utf-8")
        except Exception:
            pass

    safe_print_json({"decision": "allow", "terminationBehavior": ""})


if __name__ == "__main__":
    main()
