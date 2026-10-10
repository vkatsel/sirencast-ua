#!/usr/bin/env python3
"""
SirenCast UA: End-to-End Pipeline CLI Execution Script.
Allows running individual phases or the entire end-to-end pipeline.

Usage:
    python run_experiment.py --phase all
    python run_experiment.py --phase 1    # Data Cleaning & Deduplication
    python run_experiment.py --phase 2    # 30m Regular Grid & Spatial Radar Features
    python run_experiment.py --phase 3    # Baseline Models & Walk-Forward Validation
    python run_experiment.py --phase 4    # Telegram NLP Fusion & Multimodal Backtesting
    python run_experiment.py --test       # Run test suite
"""
import argparse
import logging
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# If invoked with system python, automatically re-exec with local .venv python
venv_python = BASE_DIR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if venv_python.exists():
    try:
        current_exe = Path(sys.executable).resolve()
        target_exe = venv_python.resolve()
        if current_exe != target_exe:
            import subprocess
            res = subprocess.run([str(target_exe)] + sys.argv, cwd=str(BASE_DIR))
            sys.exit(res.returncode)
    except Exception:
        pass

from src.data_loader import load_and_clean_official_data, get_clean_alerts
from src.features_ts import build_30m_feature_store, get_feature_dataset
from src.features_text import build_multimodal_dataset, get_multimodal_dataset
from src.evaluate import run_walk_forward_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sirencast")


def run_phase_1():
    print("\n" + "=" * 60)
    print("PHASE 1: Data Cleaning & Deduplication Audit")
    print("=" * 60)
    df, stats = load_and_clean_official_data()
    print(f"Clean alerts dataset ready: {len(df):,} valid alerts.")
    dup_pct = (stats['duplicates_removed'] / stats['initial_rows']) * 100
    print(f"Date range: {stats['min_date']} to {stats['max_date']}")
    print(f"Duplicates removed: {stats['duplicates_removed']:,} records ({dup_pct:.2f}% of raw dump).")
    return df


def run_phase_2():
    print("\n" + "=" * 60)
    print("PHASE 2: 30-Minute Grid & 4-Tier Spatial Radar Features")
    print("=" * 60)
    df = build_30m_feature_store()
    print(f"Feature dataset shape: {df.shape[0]:,} time steps x {df.shape[1]} features.")
    print(f"Timeline: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("Spatial radar tiers: Tier 0 (Kyiv), Tier 1 (7 Kyivska raions), Tier 2 (5 adjacent oblasts), Tier 3 (Launch corridors), Tier 4 (Macro speed).")
    return df


def run_phase_3():
    print("\n" + "=" * 60)
    print("PHASE 3: Baseline Models & Walk-Forward Backtesting (2026)")
    print("=" * 60)
    summary_df, _ = run_walk_forward_evaluation(save_plots=True)
    print("\nWalk-Forward Backtesting complete! Diagnostic figures exported to reports/figures/.")
    return summary_df


def run_phase_4():
    print("\n" + "=" * 60)
    print("PHASE 4: Multimodal Fusion with Telegram Threat NLP Signals")
    print("=" * 60)
    df = build_multimodal_dataset()
    print(f"Multimodal dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns.")
    print("Evaluating Multimodal model against Baselines across 2026 Walk-Forward folds...")
    summary_df, _ = run_walk_forward_evaluation(save_plots=True)
    return summary_df


def run_tests():
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.discover(str(BASE_DIR / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SirenCast UA: Machine Learning Forecasting Pipeline for Kyiv Air Alerts"
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Pipeline phase to execute (1: Cleaning, 2: Spatial Features, 3: Baselines, 4: Multimodal NLP, all: End-to-end)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit test suite",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run real-time Kyiv risk prediction from live siren map and Telegram defense channels",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="When used with --live, poll and update live predictions continuously every 30 seconds",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the local interactive web dashboard",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for local web dashboard (default: 8000)",
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if args.live:
        from src.live_forecast import format_cli_dashboard
        import time

        if args.watch:
            print("Starting SirenCast UA Live Monitor (Ctrl+C to exit)...")
            try:
                while True:
                    # Clear screen on Windows/Unix
                    os_cls = "cls" if sys.platform == "win32" else "clear"
                    # We just print without forced clear or with simple clear
                    print("\n" + format_cli_dashboard())
                    time.sleep(30)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
        else:
            print(format_cli_dashboard())
        return

    if args.serve:
        from src.web import start_server
        start_server(port=args.port)
        return

    if args.phase == "1":
        run_phase_1()
    elif args.phase == "2":
        run_phase_2()
    elif args.phase == "3":
        run_phase_3()
    elif args.phase == "4":
        run_phase_4()
    elif args.phase == "all":
        run_phase_1()
        run_phase_2()
        run_phase_4()
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE!")
        print("=" * 60)


if __name__ == "__main__":
    main()
