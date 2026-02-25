"""Task 5 tests: live inference CLI, CTGov search wrapper, openFDA client.

All HTTP is stubbed (monkeypatched requests) - the suite must run fully
offline. The model itself is stubbed where a checkpoint would be needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import predict  # noqa: E402
from fda_predictor.inference.ctgov_client import _parse_study  # noqa: E402
from fda_predictor.inference.openfda_client import (  # noqa: E402
    check_drug_approvals,
    fetch_drug_application,
    parse_approval_info,
)
from fda_predictor.inference.search import (  # noqa: E402
    search_studies_by_intervention,
    search_trials,
)


# ----------------------------------------------------------------- fixtures

def _study(nct_id: str, drug: str = "wonderdrug") -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2025-03-01"},
                "completionDateStruct": {"date": "2027-06-01"},
            },
            "designModule": {"phases": ["PHASE3"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "ACME Pharma"}},
            "armsInterventionsModule": {
                "interventions": [{"name": drug.title(), "type": "Drug"}]
            },
            "eligibilityModule": {
                "eligibilityCriteria": "Inclusion Criteria:\\n - adults 18+"
            },
        }
    }


class _FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_search_parses_records(monkeypatch):
    payloads = [_FakeResponse({"studies": [_study("NCT00000001")]})]

    def fake_get(self, url, params=None, timeout=None):
        return payloads.pop(0)

    monkeypatch.setattr("requests.Session.get", fake_get)
    records = search_trials("wonderdrug", limit=5)
    assert len(records) == 1
    assert records[0].nct_id == "NCT00000001"
    assert records[0].phase_index == 2


def test_search_timeout_is_explicit_and_bounded(monkeypatch):
    seen_timeouts = []

    def fake_get(self, url, params=None, timeout=None):
        seen_timeouts.append(timeout)
        return _FakeResponse({"studies": [_study("NCT00000002")]})

    monkeypatch.setattr("requests.Session.get", fake_get)
    out = search_studies_by_intervention("wonderdrug", limit=3)
    assert len(out) == 1
    assert seen_timeouts and all(t is not None and t <= 15.0 for t in seen_timeouts)


def test_search_network_failure_returns_empty(monkeypatch):
    import requests

    def fake_get(self, url, params=None, timeout=None):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("requests.Session.get", fake_get)
    out = search_studies_by_intervention("wonderdrug", limit=3, max_attempts=1)
    assert out == []


def test_search_pagination_respects_limit(monkeypatch):
    page1 = _FakeResponse(
        {
            "studies": [_study(f"NCT0000010{i}") for i in range(5)],
            "nextPageUrl": "https://clinicaltrials.gov/api/v2/studies?pageToken=2",
        }
    )
    page2 = _FakeResponse({"studies": [_study(f"NCT0000020{i}") for i in range(5)]})
    calls = {"n": 0}

    def fake_get(self, url, params=None, timeout=None):
        calls["n"] += 1
        return page1 if calls["n"] == 1 else page2

    monkeypatch.setattr("requests.Session.get", fake_get)
    out = search_studies_by_intervention("wonderdrug", limit=8)
    assert len(out) == 8
    assert calls["n"] == 2


# ----------------------------------------------------------------- openFDA

_OPENFDA_PAYLOAD = {
    "results": [
        {
            "application_number": "NDA000111",
            "openfda": {"brand_name": ["Testrol"]},
            "submissions": [
                {
                    "submission_type": "ORIG",
                    "submission_status": "APPROVED",
                    "submission_status_date": "19980412",
                },
                {
                    "submission_type": "SUPPL",
                    "submission_status": "APPROVED",
                    "submission_status_date": "20010515",
                },
            ],
        }
    ]
}


class TestOpenFDAClient:
    def test_parses_canned_response(self):
        info = parse_approval_info(_OPENFDA_PAYLOAD, query="testrol")
        assert info.has_prior_approval is True
        assert info.first_approval_date == "19980412"
        assert info.application_numbers == ["NDA000111"]
        assert info.brand_names == ["Testrol"]
        d = info.to_dict()
        assert d["has_prior_approval"] and d["first_approval_date"] == "19980412"

    def test_empty_payload_is_no_approval(self):
        info = parse_approval_info(None, query="x")
        assert info.has_prior_approval is False
        info = parse_approval_info({"results": []}, query="x")
        assert info.has_prior_approval is False

    def test_transient_error_returns_none(self, monkeypatch, tmp_path):
        import requests

        import fda_predictor.inference.openfda_client as ofda

        monkeypatch.setattr(ofda, "_cache_dir", lambda: tmp_path)

        def fake_get(url, timeout=None):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr("requests.get", fake_get)
        payload = fetch_drug_application(
            "transientdrug", use_cache=False, max_attempts=1
        )
        assert payload is None

    def test_404_is_definitive_miss_and_cached(self, monkeypatch, tmp_path):
        import fda_predictor.inference.openfda_client as ofda

        monkeypatch.setattr(ofda, "_cache_dir", lambda: tmp_path)

        def fake_get(url, timeout=None):
            return _FakeResponse(status_code=404)

        monkeypatch.setattr("requests.get", fake_get)
        payload = fetch_drug_application("unknowndrug", use_cache=True, max_attempts=1)
        assert payload == {"results": [], "not_found": True}
        cached = ofda._cache_path("active_ingredient:unknowndrug")
        assert cached.exists()
        again = fetch_drug_application("unknowndrug", use_cache=True)
        assert again == {"results": [], "not_found": True}

    def test_check_drug_approvals_offline_safe(self, monkeypatch, tmp_path):
        import requests

        import fda_predictor.inference.openfda_client as ofda

        monkeypatch.setattr(ofda, "_cache_dir", lambda: tmp_path)

        def fake_get(url, timeout=None):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr("requests.get", fake_get)
        info = check_drug_approvals("offlinedrug", use_cache=False)
        assert info.has_prior_approval is False
        assert info.first_approval_date is None


# ----------------------------------------------------------------- predict CLI

def _write_config(root: Path) -> Path:
    cfg_dir = root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "config.yaml"
    if not path.exists():
        real = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
        path.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def test_parse_args_requires_input():
    with pytest.raises(SystemExit):
        predict.parse_args([])


def test_parse_args_accepts_nct_ids():
    args = predict.parse_args(["NCT00000001", "NCT00000002"])
    assert args.nct_ids == ["NCT00000001", "NCT00000002"]
    assert args.device == "cpu"


def test_read_nct_file(tmp_path):
    f = tmp_path / "ncts.txt"
    f.write_text("NCT00000001\n\n# comment\nNCT00000002\n", encoding="utf-8")
    assert predict.read_nct_file(f) == ["NCT00000001", "NCT00000002"]


def test_find_default_checkpoint_prefers_newest(tmp_path):
    import os

    old = tmp_path / "stage2_clinicalbert_a.pt"
    new = tmp_path / "stage2_clinicalbert_b.pt"
    old.write_bytes(b"x")
    new.write_bytes(b"yy")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    found = predict.find_default_checkpoint(tmp_path)
    assert found == new
    empty = tmp_path / "none"
    empty.mkdir()
    assert predict.find_default_checkpoint(empty) is None


def test_trial_frame_row_masks():
    rec = _parse_study(_study("NCT00000003"))
    assert rec is not None
    row = predict.trial_frame_row(rec, ticker=None, max_drugs=3)
    assert row["nctid"] == "NCT00000003"
    assert row["stock_mask"] == 0
    assert row["molecule_mask"] in (0, 1)
    assert row["label"] == 0.0


def test_build_report_schema():
    rec = _parse_study(_study("NCT00000004"))
    assert rec is not None
    row = predict.trial_frame_row(rec, ticker=None, max_drugs=3)
    report = predict.build_report(
        records={"NCT00000004": rec},
        probabilities={"NCT00000004": 0.7321},
        frame_rows_by_nct={"NCT00000004": row},
        checkpoint=Path("fake.pt"),
        layout=4,
        device_str="cpu",
        openfda_enabled=False,
    )
    assert report["checkpoint"]["checkpoint_layout"] == 4
    assert "model_version_note" in report
    t = report["trials"][0]
    assert t["nct_id"] == "NCT00000004"
    assert t["success_probability"] == 0.7321
    assert "phase" in t and "sponsor" in t and "drugs" in t


def test_score_records_with_stub_net(monkeypatch):
    """score_records must consume the collator batch and sigmoid the logits."""

    class StubNet:
        def forward_with_approval(self, **kwargs):
            import torch

            b = kwargs["batch_size"]
            succ = torch.linspace(-2.0, 2.0, b).unsqueeze(-1)
            appr = torch.linspace(2.0, -2.0, b).unsqueeze(-1)
            return succ, appr

    rec = _parse_study(_study("NCT00000005"))
    assert rec is not None
    rows = [predict.trial_frame_row(rec, ticker=None, max_drugs=1)]
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    probs = predict.score_records(StubNet(), rows, config, device="cpu", batch_size=2)
    assert set(probs.keys()) == {"NCT00000005"}
    p = probs["NCT00000005"]
    assert 0.0 < p["success"] < 1.0
    assert 0.0 < p["approval"] < 1.0


def test_main_reports_missing_checkpoint_gracefully(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predict, "CHECKPOINTS_DIR", tmp_path / "empty")
    args = predict.parse_args(["NCT00000006", "--no-openfda"])
    rc = predict.main(["NCT00000006", "--no-openfda", "--device", "cpu"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "No checkpoint" in err or "ERROR" in err
