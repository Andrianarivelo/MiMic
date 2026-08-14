# -*- coding: utf-8 -*-
"""Utilities for inspecting and validating bundled pretrained models."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import json

import torch

from argparse import ArgumentParser

from vocalpy.nn.classifier import VocalClassifier
from vocalpy.nn.segmenter import VocalSegmenter
from vocalpy.nn.pretrained_models import PRETRAINED_MODEL_SPECS, get_pretrained_model_spec, validate_pretrained_model_file


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "command",
        choices=["list", "validate"],
        nargs="?",
        default="validate",
        help="inspect metadata or validate bundled checkpoints",
    )
    parser.add_argument(
        "--network-type",
        choices=["class", "noise", "segment", "all"],
        default="all",
        help="model type to inspect or validate",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="load the checkpoint on CPU and run a deterministic dummy forward pass",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of plain text",
    )
    return parser


def get_selected_specs(network_type):
    if network_type == "all":
        return [PRETRAINED_MODEL_SPECS["noise"], PRETRAINED_MODEL_SPECS["class"], PRETRAINED_MODEL_SPECS["segment"]]
    return [get_pretrained_model_spec(network_type)]


def list_specs(specs):
    return [
        {
            "network_type": spec.network_type,
            "filename": spec.filename,
            "architecture": spec.architecture,
            "num_classes": spec.num_classes,
            "classes": list(spec.classes),
            "input_shape": list(spec.input_shape),
            "checkpoint_keys": list(spec.checkpoint_keys),
            "prediction_type": spec.prediction_type,
            "default_threshold": spec.default_threshold,
            "sha256": spec.sha256,
            "storage": spec.storage,
            "source": spec.source,
            "source_version": spec.source_version,
            "repo_history": spec.repo_history,
            "notes": spec.notes,
            "path": str(spec.path),
            "metadata_path": str(spec.metadata_path),
        }
        for spec in specs
    ]


def run_smoke_test(spec):
    if spec.network_type == "segment":
        segmenter = VocalSegmenter.__new__(VocalSegmenter)
        model = segmenter.load_segmentation_model(torch.device("cpu"))
        dummy_input = torch.zeros((1, *spec.input_shape), dtype=torch.float32)
        output_a = model(dummy_input)
        output_b = model(dummy_input)
        torch.testing.assert_close(output_a, output_b)
        return {
            "output_shape": list(output_a.shape),
            "finite": bool(torch.isfinite(output_a).all()),
        }

    classifier = VocalClassifier.__new__(VocalClassifier)
    loader = classifier.load_pretrained_class_model if spec.network_type == "class" else classifier.load_pretrained_noise_model
    model = loader(torch.device("cpu"))
    dummy_input = torch.zeros((1, *spec.input_shape), dtype=torch.float32)
    output_a = model(dummy_input)
    output_b = model(dummy_input)
    torch.testing.assert_close(output_a, output_b)
    return {
        "output_shape": list(output_a.shape),
        "finite": bool(torch.isfinite(output_a).all()),
    }


def validate_specs(specs, smoke_test=False):
    results = []
    for spec in specs:
        result = {
            "network_type": spec.network_type,
            "path": str(spec.path),
            "metadata_path": str(spec.metadata_path),
            "sha256": validate_pretrained_model_file(spec.path, expected_sha256=spec.sha256),
        }
        if smoke_test:
            result["smoke_test"] = run_smoke_test(spec)
        results.append(result)
    return results


def print_plain(command, results):
    for result in results:
        print(f"[{result['network_type']}]")
        for key, value in result.items():
            if key == "network_type":
                continue
            print(f"{key}: {value}")
        if command == "list":
            print()


def main(argv=None):
    args = build_parser().parse_args(argv)
    specs = get_selected_specs(args.network_type)
    if args.command == "list":
        results = list_specs(specs)
    else:
        results = validate_specs(specs, smoke_test=args.smoke_test)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_plain(args.command, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
