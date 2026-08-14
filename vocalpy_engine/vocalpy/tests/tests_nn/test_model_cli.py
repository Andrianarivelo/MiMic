# -*- coding: utf-8 -*-
"""Regression tests for pretrained model CLI utilities."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import json

from vocalpy.model_cli import main


def test_model_cli_lists_bundled_metadata_as_json(capsys):
    exit_code = main(["list", "--json"])

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert [entry["network_type"] for entry in payload] == ["noise", "class", "segment"]
    assert payload[0]["filename"] == "noise_model.pth.tar"


def test_model_cli_validate_uses_requested_network_type(monkeypatch, capsys):
    captured = {}

    def fake_validate(specs, smoke_test=False):
        captured["specs"] = [spec.network_type for spec in specs]
        captured["smoke_test"] = smoke_test
        return [{"network_type": "class", "sha256": "ok"}]

    monkeypatch.setattr("vocalpy.model_cli.validate_specs", fake_validate)

    exit_code = main(["validate", "--network-type", "class", "--smoke-test", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured == {"specs": ["class"], "smoke_test": True}
    assert payload == [{"network_type": "class", "sha256": "ok"}]
