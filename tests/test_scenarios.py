import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import Request, urlopen

from ab_aa_lab.scenarios import catalog, config_from_form, presets, run_form
from server import Handler


def test_presets_build_the_two_stories():
    forms = presets()
    selection = config_from_form(forms["selection"])
    fade = config_from_form(forms["fade"])
    assert [phase["name"] for phase in selection["phases"]] == ["search", "ramp"]
    assert selection["phases"][1]["exposure"] == 1
    assert selection["phases"][1]["strategy_active"] is False
    assert [phase["name"] for phase in fade["phases"]] == ["search", "hold"]
    assert fade["phases"][1]["evolve"] is True
    assert fade["process"]["name"] == "ar1"
    assert [run["name"] for run in selection["runs"]] == [
        "honest",
        "extremes_hackable",
        "extremes_unhackable",
        "oracle_upper",
        "oracle_lower",
    ]
    assert selection["runs"][3]["strategy"]["arm_fraction"] == 0.2


def test_form_rejects_an_unknown_comparison():
    form = dict(presets()["selection"], compare=["missing"])
    try:
        config_from_form(form)
    except ValueError as exc:
        assert "未知" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_form_returns_finite_series():
    form = dict(presets()["selection"], n_users=1000, search_rounds=2)
    payload = run_form(form)
    assert payload["n_users"] == 1000
    assert len(payload["runs"]) == 5
    hack = next(run for run in payload["runs"] if run["id"] == "extremes_hackable")
    assert hack["system"] == "hackable"
    assert len(hack["series"]) == 3
    assert all(point["ab_claim"] is not None for point in hack["series"])


def test_http_serves_catalog_and_a_run():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/catalog") as response:
            body = json.load(response)
        assert "selection" in body["presets"]
        form = dict(body["presets"]["selection"], n_users=1000, search_rounds=1, compare=["honest"])
        request = Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps(form).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            payload = json.load(response)
        assert payload["runs"][0]["id"] == "honest"
        with urlopen(f"http://127.0.0.1:{port}/") as response:
            assert "流量层实验室" in response.read().decode()
    finally:
        httpd.shutdown()


def test_catalog_covers_every_preset_choice():
    info = catalog()
    ids = {run["id"] for run in info["runs"]}
    for form in info["presets"].values():
        assert set(form["compare"]) <= ids
        assert form["metric"] in {item["id"] for item in info["metrics"]}
        assert form["process"] in {item["id"] for item in info["processes"]}
        assert form["afterward"] in {item["id"] for item in info["afterwards"]}
