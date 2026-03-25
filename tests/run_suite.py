"""Automated conversation test suite entry point.

Runs scripted conversations through the Antagonistic Robot's LLM pipeline,
evaluates responses for safety and behavioral correctness, and generates
reports.

Usage::

    python -m tests.run_suite --tier smoke
    python -m tests.run_suite --tier deep --runs 3
    python -m tests.run_suite --case-id neg_polar_1 --runs 1
"""

import argparse
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from antagonist_robot.config.settings import load_config
from antagonist_robot.conversation.avct_manager import AvctManager
from antagonist_robot.logging.session_logger import SessionLogger
from antagonist_robot.pipeline.llm import LLMEngine

from tests.evaluator.behaviour_evaluator import BehaviourEvaluator, EvalResult
from tests.evaluator.safety_checker import SafetyChecker, SafetyResult
from tests.reporter import Reporter
from tests.simulator.matrix import TestMatrix
from tests.simulator.scripter import ConversationScripter, ScriptedTurn
from tests.simulator.text_injector import TextInjector

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Antagonistic Robot — Automated Conversation Test Suite",
    )
    parser.add_argument(
        "--tier",
        choices=["smoke", "deep", "all"],
        default="smoke",
        help="Test tier to run (default: smoke)",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Run only this case_id (optional)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Repetitions per test case (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/results",
        help="Directory for reports and test DB (default: tests/results/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the test suite."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # -- 0. Load .env so API keys are available ------------------------------
    load_dotenv()

    # -- 1. Load config -----------------------------------------------------
    log.info("Loading config from %s", args.config)
    config = load_config(args.config)

    # -- 2-4. Build pipeline components -------------------------------------
    llm = LLMEngine(config.llm)
    avct_manager = AvctManager(config.avct)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_logger = SessionLogger(
        db_path=str(output_dir / "test_suite.db"),
        audio_dir=str(output_dir / "audio"),
        save_audio=False,
    )

    # -- 5-6. Build simulator -----------------------------------------------
    injector = TextInjector(
        llm=llm,
        avct_manager=avct_manager,
        session_logger=session_logger,
    )
    scripter = ConversationScripter(injector)

    # -- 7-8. Load test matrix and filter -----------------------------------
    matrix = TestMatrix()
    scripts = matrix.get_scripts()

    tier_arg = None if args.tier == "all" else args.tier
    cases = matrix.get_cases(tier=tier_arg)

    if args.case_id:
        cases = [c for c in cases if c.case_id == args.case_id]
        if not cases:
            log.error("No test case found with id: %s", args.case_id)
            sys.exit(1)

    log.info(
        "Running %d case(s), %d run(s) each — tier=%s",
        len(cases), args.runs, args.tier,
    )

    # -- Contradiction check ------------------------------------------------
    for case in cases:
        for desc in matrix.get_contradictions(case):
            log.warning(
                "case '%s' has contradictory modifier combination: %s\n"
                "  This case will still run but results should be "
                "interpreted with caution.",
                case.case_id, desc,
            )

    # -- 9. Build evaluators ------------------------------------------------
    safety_checker = SafetyChecker()
    evaluator = BehaviourEvaluator(llm)

    # -- 10. Run all cases --------------------------------------------------
    all_eval_results: list[EvalResult] = []

    for case in cases:
        case_results: list[EvalResult] = []

        script_turns_raw = scripts.get(case.script_key)
        if script_turns_raw is None:
            log.warning(
                "Script '%s' not found for case %s — skipping",
                case.script_key, case.case_id,
            )
            continue

        scripted_turns = [
            ScriptedTurn(user_text=t, avct_override=None)
            for t in script_turns_raw
        ]

        for run_idx in range(args.runs):
            log.info(
                "  %s  run %d/%d ...",
                case.case_id, run_idx + 1, args.runs,
            )

            # a. Run the scripted conversation
            session = scripter.run(
                case_id=case.case_id,
                polar_level=case.polar_level,
                category=case.category,
                subtype=case.subtype,
                modifiers=case.modifiers,
                script=scripted_turns,
            )

            # b. Safety check every turn, keep worst result
            worst_safety = SafetyResult(passed=True, violations=[], raw_text="")
            for turn in session.turns:
                sr = safety_checker.check(turn.llm_response)
                if not sr.passed:
                    worst_safety = sr

            # c. Behavioral evaluation
            eval_result = evaluator.evaluate(
                case=case,
                turns=session.turns,
                safety_result=worst_safety,
            )
            case_results.append(eval_result)

        # -- Per-case summary -----------------------------------------------
        all_eval_results.extend(case_results)

        scores = [r.score for r in case_results]
        mean_score = statistics.mean(scores) if scores else 0.0
        pass_rate = (
            sum(1 for r in case_results if r.verdict == "pass")
            / len(case_results)
        ) if case_results else 0.0
        variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        stability = "unstable" if variance > 2.0 else "stable"

        verdicts = [r.verdict for r in case_results]
        dominant_verdict = max(set(verdicts), key=verdicts.count) if verdicts else "—"

        print(
            f"  {case.case_id:<25s} | {dominant_verdict:<7s} "
            f"| score={mean_score:.1f} | pass_rate={pass_rate:.0%} "
            f"| {stability}"
        )

    # -- 11. Generate report ------------------------------------------------
    run_metadata = {
        "tier": args.tier,
        "runs": args.runs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
    }

    reporter = Reporter(output_dir)
    reporter.write(all_eval_results, run_metadata)

    # -- Final summary ------------------------------------------------------
    total = len(all_eval_results)
    total_passed = sum(1 for r in all_eval_results if r.passed)
    total_rate = (total_passed / total * 100) if total else 0

    print(f"\n{'='*60}")
    print(f"  DONE — {total_passed}/{total} passed ({total_rate:.0f}%)")
    print(f"  Reports written to {output_dir.resolve()}")
    print(f"{'='*60}")

    session_logger.close()


if __name__ == "__main__":
    main()
