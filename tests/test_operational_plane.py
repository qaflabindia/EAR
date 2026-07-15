"""Phase 3 completion -- the remaining operational command centres.

Phase 3's first two items (AECC envelope enforcement, ATC adversarial hook)
have their own tests. This covers the third: rolling the remaining
operational centres -- HRCC, TAIC, ALGCC, ARCC, AITCC -- through the existing
Phase-1 bind and Phase-2 compile machinery. No new mechanism: the point is
that the same machinery generalizes across the whole operational plane.

All offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ear import Governor, Intent, Runtime
from ear.compiler import compile_command_centre
from ear.enterprise import (
    OPERATIONAL,
    CommandCentre,
    bind_command_centres,
    load_command_centres,
    plane_of,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "command_centres"
OPERATIONAL_CENTRES = ["afcc", "hrcc", "taic", "algcc", "arcc", "aitcc"]
NEW_CENTRES = ["hrcc", "taic", "algcc", "arcc", "aitcc"]


@pytest.mark.parametrize("slug", NEW_CENTRES)
def test_centre_loads_on_the_operational_plane(slug):
    centre = CommandCentre.load(FIXTURES / slug)
    assert centre.plane == OPERATIONAL
    assert plane_of(slug) == OPERATIONAL
    assert len(centre.constitution.rules) == 3


@pytest.mark.parametrize("slug", NEW_CENTRES)
def test_centre_binds_its_constitution(slug):
    centre = CommandCentre.load(FIXTURES / slug)
    runtime = Runtime(name=slug)
    binding = centre.bind(runtime)
    assert len(binding.enforced) == 3
    assert len(runtime.policies) == 3


@pytest.mark.parametrize("slug", NEW_CENTRES)
def test_centre_compiles_to_a_runnable_stack(slug, tmp_path):
    stack = compile_command_centre(FIXTURES / slug, tmp_path, verify=True)
    assert stack.skills  # capabilities compiled
    assert stack.workflows  # procedures compiled
    assert stack.knowledge  # references copied as knowledge
    runtime = stack.load()
    assert len(runtime.policies) == 3
    assert runtime.processes


# -- domain rules actually fire (deterministic fallbacks) -------------------


def test_hrcc_compensation_change_parks():
    runtime = Runtime(name="hr")
    CommandCentre.load(FIXTURES / "hrcc").bind(runtime)
    parked = Governor().govern(runtime, Intent(text="raise", context={"compensation_change": True}))
    cr = next(p for p in parked if p.name.startswith("CR-HR-01"))
    assert cr.approval_required is True


def test_taic_offer_over_band_parks():
    runtime = Runtime(name="ta")
    CommandCentre.load(FIXTURES / "taic").bind(runtime)
    parked = Governor().govern(
        runtime, Intent(text="offer", context={"offer_amount": 260000, "band_maximum": 240000})
    )
    assert any(p.name.startswith("CR-TA-02") for p in parked)


def test_algcc_high_value_shipment_parks_and_hazmat_blocks():
    runtime = Runtime(name="lg")
    CommandCentre.load(FIXTURES / "algcc").bind(runtime)
    parked = Governor().govern(
        runtime,
        Intent(text="ship", context={"shipment_value": 80000, "hazardous": False, "hazmat_certified_route": True}),
    )
    assert any(p.name.startswith("CR-LG-02") for p in parked)

    hazmat = Governor().govern(
        runtime,
        Intent(text="ship", context={"shipment_value": 100, "hazardous": True, "hazmat_certified_route": False}),
    )
    assert any(p.name.startswith("CR-LG-01") for p in hazmat)


def test_arcc_over_discount_parks_and_consent_blocks():
    runtime = Runtime(name="re")
    CommandCentre.load(FIXTURES / "arcc").bind(runtime)
    discount = Governor().govern(runtime, Intent(text="offer", context={"discount_pct": 35}))
    assert any(p.name.startswith("CR-RE-02") for p in discount)

    consent = Governor().govern(
        runtime, Intent(text="email", context={"marketing_use": True, "marketing_consent": False})
    )
    assert any(p.name.startswith("CR-RE-03") for p in consent)


def test_aitcc_freeze_and_privilege_block():
    runtime = Runtime(name="it")
    CommandCentre.load(FIXTURES / "aitcc").bind(runtime)
    frozen = Governor().govern(
        runtime,
        Intent(text="deploy", context={"production_change": True, "change_freeze_active": True, "access_grant": False}),
    )
    assert any(p.name.startswith("CR-IT-03") for p in frozen)

    over_priv = Governor().govern(
        runtime,
        Intent(text="grant", context={"access_grant": True, "privilege_exceeds_task": True, "production_change": False}),
    )
    assert any(p.name.startswith("CR-IT-02") for p in over_priv)


def test_clean_context_clears_each_operational_centre():
    for slug, clean in (
        ("hrcc", {"compensation_change": False}),
        ("taic", {"offer_amount": 200000, "band_maximum": 240000}),
        ("algcc", {"shipment_value": 100, "hazardous": False, "hazmat_certified_route": True}),
        ("arcc", {"discount_pct": 10, "marketing_use": False, "marketing_consent": True}),
        ("aitcc", {"production_change": False, "access_grant": False, "change_freeze_active": False, "privilege_exceeds_task": False}),
    ):
        runtime = Runtime(name=slug)
        CommandCentre.load(FIXTURES / slug).bind(runtime)
        assert Governor().govern(runtime, Intent(text="routine", context=clean)) == [], slug


# -- the whole plane, through load/bind_command_centres ---------------------


def test_load_command_centres_discovers_the_whole_operational_plane():
    centres = load_command_centres(FIXTURES)
    for slug in OPERATIONAL_CENTRES:
        assert slug in centres, slug
        assert centres[slug].plane == OPERATIONAL


def test_bind_command_centres_binds_governance_before_operational():
    centres = load_command_centres(FIXTURES)
    runtime = Runtime(name="Enterprise")
    bindings = bind_command_centres(runtime, centres)
    planes = [b.plane for b in bindings]
    # Every governance binding precedes every operational one.
    last_governance = max(i for i, p in enumerate(planes) if p == "governance")
    first_operational = min(i for i, p in enumerate(planes) if p == OPERATIONAL)
    assert last_governance < first_operational
