"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP — Master Runner.

Orchestrates all experiment steps. Run individual steps or the full pipeline.
"""

import argparse
import os
import sys

# Ensure GPT2/ module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def Main():
    parser = argparse.ArgumentParser(
        description="GPT-2 Taylor-MLP Experiment Runner")

    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "baseline", "collect", "analyze",
                                 "single", "cumulative"],
                        help="Which step to run (default: all)")

    parser.add_argument("--model", type=str, default=None,
                        help="Model name override")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name override")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max token samples override")
    parser.add_argument("--k", type=str, default=None,
                        help="Comma-separated k values, e.g. '1,4,16,64'")
    parser.add_argument("--layers", type=str, default=None,
                        help="Layer indices for single-replace, e.g. '0,5,11'")

    args = parser.parse_args()

    k_values = None
    if args.k:
        k_values = [int(x.strip()) for x in args.k.split(",")]

    layer_indices = None
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    kwargs = {}
    if args.model:
        kwargs["model_name"] = args.model
    if args.dataset:
        kwargs["dataset_name"] = args.dataset
    if args.max_samples:
        kwargs["max_samples"] = args.max_samples
    if k_values:
        kwargs["k_values"] = k_values

    steps = []

    if args.step == "all":
        steps = ["baseline", "collect", "analyze", "single", "cumulative"]
    else:
        steps = [args.step]

    for step in steps:
        print(f"\n{'#'*60}")
        print(f"#  Running: {step}")
        print(f"{'#'*60}\n")

        if step == "baseline":
            from eval_baseline import Evaluate_Baseline
            Evaluate_Baseline(**{k: v for k, v in kwargs.items()
                                 if k in ["model_name", "dataset_name",
                                          "max_samples"]})

        elif step == "collect":
            from collect_data import Collect_Data
            Collect_Data(**kwargs)

        elif step == "analyze":
            from layer_analysis import Analyze_Layers
            Analyze_Layers(**kwargs)

        elif step == "single":
            from single_replace import Test_Single_Replacements
            skwargs = dict(kwargs)
            if layer_indices:
                skwargs["layer_indices"] = layer_indices
            Test_Single_Replacements(**skwargs)

        elif step == "cumulative":
            from cumulative_replace import Test_Cumulative
            Test_Cumulative(**kwargs)

        print(f"\n{'='*60}")
        print(f"Step '{step}' complete.")
        print(f"{'='*60}")

    print("\nAll done.")


if __name__ == "__main__":
    Main()
