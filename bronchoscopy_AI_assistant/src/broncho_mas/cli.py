import argparse
import json
import os

from .adapter import SmolAgentsLLM
from .sim.offline_lab import main as simulate_main


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


def _apply_pipeline_args(args) -> None:
    if getattr(args, "pipeline", None):
        os.environ["BRONCHO_PIPELINE"] = args.pipeline
    if getattr(args, "strict_pipeline", False):
        os.environ["BRONCHO_STRICT_PIPELINE"] = "1"


def _cmd_run(args) -> None:
    _apply_pipeline_args(args)
    prompt = _resolve_prompt(args)
    llm = SmolAgentsLLM(model_name=args.model)
    result = llm.ask_structured(prompt)

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
    _apply_pipeline_args(args)
    llm = SmolAgentsLLM(model_name=args.model)
    report = llm.get_report(recording_dir=args.recording_dir)
    print(report)


def _cmd_simulate(args) -> None:
    _apply_pipeline_args(args)
    argv = ["--model", args.model]
    if args.input:
        argv.extend(["--input", args.input])
    if args.steps:
        argv.extend(["--steps", str(args.steps)])
    if args.sleep:
        argv.extend(["--sleep", str(args.sleep)])
    if args.json_out:
        argv.extend(["--json-out", args.json_out])
    simulate_main(argv)


def main():
    parser = argparse.ArgumentParser(description="broncho-mas CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run (turn-level) ---
    p_run = subparsers.add_parser("run", help="Run one turn of coaching (turn-level)")
    p_run.add_argument("--model", default="Qwen/Qwen3.5-27B",
                       help="HF repo id, e.g., Qwen/Qwen3.5-27B")
    p_run.add_argument("--pipeline", default=os.environ.get("BRONCHO_PIPELINE", "runtime"),
                       choices=["runtime", "mas", "research", "sas"],
                       help="Active orchestration line to run.")
    p_run.add_argument("--strict-pipeline", action="store_true",
                       help="Fail loudly instead of allowing any fallback behavior.")
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
    p_report.add_argument("--pipeline", default=os.environ.get("BRONCHO_PIPELINE", "sas"),
                          choices=["mas", "research", "sas", "runtime"],
                          help="Pipeline that should generate the report.")
    p_report.add_argument("--strict-pipeline", action="store_true",
                          help="Fail loudly instead of allowing any fallback behavior.")
    p_report.add_argument("--recording-dir", required=True,
                          help="Directory containing timeline.json (and any other session artifacts).")
    p_report.set_defaults(func=_cmd_report)

    # --- simulate (robot-free offline lab) ---
    p_sim = subparsers.add_parser("simulate", help="Run BronchoMAS offline with synthetic or recorded frames")
    p_sim.add_argument("--model", default="Qwen/Qwen3.5-27B",
                       help="Model id for the active pipeline.")
    p_sim.add_argument("--pipeline", default=os.environ.get("BRONCHO_PIPELINE", "runtime"),
                       choices=["runtime", "mas", "research", "sas"],
                       help="Pipeline to use during offline simulation.")
    p_sim.add_argument("--strict-pipeline", action="store_true",
                       help="Fail loudly instead of allowing any fallback behavior.")
    p_sim.add_argument("--input", default=None,
                       help="Optional path to a recorded timeline (.json or .jsonl).")
    p_sim.add_argument("--steps", type=int, default=0,
                       help="Optional max number of simulated frames.")
    p_sim.add_argument("--sleep", type=float, default=0.0,
                       help="Optional pause in seconds between frames.")
    p_sim.add_argument("--json-out", default=None,
                       help="Optional path to write the full offline results as JSON.")
    p_sim.set_defaults(func=_cmd_simulate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
