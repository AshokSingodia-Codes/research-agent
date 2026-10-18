#!/usr/bin/env python
"""
Automated Test Runner for Multi-Agent Research Assistant.
Executes test suites by tier, validates budgets and crash-resumption,
and generates coverage reports.

Usage:
    python run_tests.py                 # Run all tests with coverage
    python run_tests.py --unit          # Unit tests only
    python run_tests.py --integration   # Integration tests only
    python run_tests.py --budget        # Budget tests only
    python run_tests.py --durability    # Redis crash-resumption tests
"""

import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant Test Automation Runner")
    parser.add_argument("--unit", action="store_true", help="Run isolated unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--budget", action="store_true", help="Run budget and circuit breaker tests only")
    parser.add_argument("--durability", action="store_true", help="Run Redis persistence & crash tests only")
    parser.add_argument("--no-cov", action="store_true", help="Disable coverage reporting for faster execution")

    args = parser.parse_args()

    cmd = [sys.executable, "-m", "pytest"]

    # Filter by marker
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.budget:
        cmd.extend(["-m", "budget"])
    elif args.durability:
        cmd.extend(["-m", "durability"])

    if args.no_cov:
        cmd.append("--no-cov")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("[RUN] EXECUTING MULTI-AGENT AUTOMATION TEST SUITE")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    print("=" * 70)
    if result.returncode == 0:
        print(f"[SUCCESS] ALL AUTOMATED TESTS PASSED in {elapsed:.2f}s!")
        print("Coverage report available in: coverage_html/index.html")
    else:
        print(f"[FAILURE] TEST SUITE FAILED (exit code {result.returncode}) in {elapsed:.2f}s")
    print("=" * 70)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
