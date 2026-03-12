import argparse
import json

from .adapter import SmolAgentsLLM


def _resolve_prompt(args) -> str:
    prompt = args.prompt

    # Resolve prompt source: --prompt > --prompt-file > stdin
    if (prompt is None or prompt == "") and args.prompt_file:
        if args.prompt_file.strip() == "-":
            import sys
            prompt = sys.stdin.read()
        else:
            from pathlib import Path
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    if prompt is None or prompt.strip() == "":
        import sys
        if not sys.stdin.isatty():
            prompt = sys.stdin.read()
        else:
            raise SystemExit("Provide --prompt, --prompt-file, or pipe prompt via stdin")

    return prompt


def _cmd_run(args) -> None:
    prompt = _resolve_prompt(args)
    llm = SmolAgentsLLM(model_name=args.model)
    result = llm.ask(prompt)

    # Backward compatibility: if older manager returns str, just print it
    if not isinstance(result, dict):
        print(result)
        return

    if args.out == "instructor":
        print(result.get("ui_text") or result.get("instructor", ""))
        return

    if args.out == "stats":
        print(json.dumps(result.get("statistics", {}), ensure_ascii=False, indent=2))
        return

    if args.out == "both":
        print(result.get("ui_text") or result.get("instructor", ""))
        print("\n--- statistics ---")
        print(json.dumps(result.get("statistics", {}), ensure_ascii=False, indent=2))
        return

    # args.out == "json"
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_report(args) -> None:
    llm = SmolAgentsLLM(model_name=args.model)
    report = llm.get_report(recording_dir=args.recording_dir)
    print(report)


def main():
    parser = argparse.ArgumentParser(description="broncho-mas CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run (turn-level) ---
    p_run = subparsers.add_parser("run", help="Run one turn of coaching (turn-level)")
    p_run.add_argument("--model", default="Qwen/Qwen3.5-27B",
                       help="HF repo id, e.g., Qwen/Qwen3.5-27B")
    p_run.add_argument("--prompt", default=None,
                       help="Full prompt text (include CURRENT_SITUATION/PREVIOUS_MSGS/STUDENT_QUESTION blocks)")
    p_run.add_argument("--prompt-file", default=None,
                       help="Path to a text file containing the full prompt. Use '-' to read from stdin.")
    p_run.add_argument("--out", default="both", choices=["instructor", "stats", "both", "json"],
                       help="What to print: instructor, stats, both, or full json.")
    p_run.set_defaults(func=_cmd_run)

    # --- report (session-level) ---
    p_report = subparsers.add_parser("report", help="Generate end-of-session report (session-level)")
    p_report.add_argument("--model", default="Qwen/Qwen3.5-27B",
                          help="HF repo id, e.g., Qwen/Qwen3.5-27B")
    p_report.add_argument("--recording-dir", required=True,
                          help="Directory containing timeline.json (and any other session artifacts).")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
