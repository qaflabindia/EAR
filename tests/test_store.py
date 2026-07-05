"""Tests for the named on-disk catalogues (Store, SkillStore, PersonaStore,
TaskStore, WorkflowStore, ProcessStore, PolicyStore, Stores)."""

from __future__ import annotations

import pytest

from ear import (
    Persona,
    PersonaStore,
    Policy,
    PolicyStore,
    Process,
    ProcessStore,
    Skill,
    SkillStore,
    Store,
    Stores,
    TaskDefinition,
    TaskStore,
    Workflow,
    WorkflowStore,
)


def test_skill_store_round_trips_a_skill(tmp_path):
    store = SkillStore(tmp_path / "skills")
    skill = Skill(name="Band Credit Profile", description="Bands score and DTI", prompt="Band the score and DTI.")
    store.save(skill)

    assert store.list() == ["Band Credit Profile"]
    loaded = store.load("band-credit-profile")
    assert loaded.name == "Band Credit Profile"
    assert loaded.description == "Bands score and DTI"
    assert loaded.prompt == "Band the score and DTI."


def test_skill_store_load_all_keys_by_normalized_name(tmp_path):
    store = SkillStore(tmp_path / "skills")
    store.save(Skill(name="Draft Note", prompt="Draft a short note."))
    catalogue = store.load_all()
    assert set(catalogue) == {"draft note"}


def test_persona_store_resolves_skills_reference(tmp_path):
    skills = SkillStore(tmp_path / "skills")
    skill = Skill(name="Write Customer Note", prompt="Draft a courteous note.")
    skills.save(skill)

    personas = PersonaStore(tmp_path / "personas")
    persona = Persona(name="Customer Advocate", instructions="Speak plainly and kindly.")
    persona.add_skill(skill)
    personas.save(persona)

    loaded = personas.load("Customer Advocate", skills.load_all())
    assert loaded.instructions == "Speak plainly and kindly."
    assert [s.name for s in loaded.skills] == ["Write Customer Note"]


def test_persona_store_unknown_skill_reference_fails_loudly(tmp_path):
    personas = PersonaStore(tmp_path / "personas")
    persona = Persona(name="Lone Persona", instructions="Solo.")
    persona.add_skill(Skill(name="Ghost Skill", prompt="Not catalogued."))
    personas.save(persona)

    with pytest.raises(ValueError, match="Unknown skill"):
        personas.load("Lone Persona", skills={})


def test_policy_store_round_trips_approval_fields(tmp_path):
    store = PolicyStore(tmp_path / "policies")
    policy = Policy(
        name="No PII in Dashboard",
        statement="The dashboard must never display raw customer PII.",
        approval_required=True,
        approvers=["reviewer@example.com"],
        escalation="after 2 days",
    )
    store.save(policy)

    loaded = store.load("No PII in Dashboard")
    assert loaded.statement == "The dashboard must never display raw customer PII."
    assert loaded.approval_required is True
    assert loaded.approvers == ["reviewer@example.com"]
    assert loaded.escalation_days == 2.0


def test_task_store_round_trips_sipoc_fields(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    task = TaskDefinition(
        name="Sanity Check",
        instruction="Validate schema, count, nulls, duplicates, running validate_data.py.",
        persona_name="Data Engineer",
        supplier="Step 1",
        inputs="Loaded data, validate_data.py",
        process="Validate schema, count, nulls, duplicates",
        outputs="Clean data",
        customer="Step 3",
        artifact="validate_data.py",
    )
    store.save(task)

    loaded = store.load("Sanity Check")
    assert loaded.instruction.startswith("Validate schema")
    assert loaded.persona_name == "Data Engineer"
    assert loaded.supplier == "Step 1"
    assert loaded.artifact == "validate_data.py"
    assert "Artifact: validate_data.py" in loaded.sipoc()


def test_task_store_round_trips_org_id(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    store.save(TaskDefinition(name="Reconcile Ledger", instruction="Reconcile.", org_id="org_acme_prod"))

    loaded = store.load("Reconcile Ledger")
    assert loaded.org_id == "org_acme_prod"
    assert "Org id: org_acme_prod" in store.store.read("Reconcile Ledger")


def test_task_definition_without_org_id_omits_the_field(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    store.save(TaskDefinition(name="Untagged", instruction="Do it."))

    assert "Org id" not in store.store.read("Untagged")
    assert store.load("Untagged").org_id == ""


def test_workflow_store_resolves_personas_and_policies(tmp_path):
    persona = Persona(name="Data Engineer", instructions="Careful and methodical.")
    policy = Policy(name="No PII", statement="Never expose PII.")

    workflow = Workflow(name="Sales MIS Workflow")
    workflow.add_step("Validate the staged data.", persona=persona)
    workflow.add_policy(policy)

    store = WorkflowStore(tmp_path / "workflows")
    store.save(workflow)

    loaded = store.load(
        "Sales MIS Workflow",
        personas={"data engineer": persona},
        policies={"no pii": policy},
    )
    assert [step.instruction for step in loaded.steps] == ["Validate the staged data."]
    assert loaded.steps[0].persona.name == "Data Engineer"
    assert [p.name for p in loaded.policies] == ["No PII"]


def test_process_store_resolves_workflows(tmp_path):
    workflow = Workflow(name="Sales MIS Workflow")
    process = Process(name="Sales MIS Process", description="Generates the daily dashboard.")
    process.add_workflow(workflow)

    store = ProcessStore(tmp_path / "processes")
    store.save(process)

    loaded = store.load("Sales MIS Process", workflows={"sales mis workflow": workflow})
    assert loaded.description == "Generates the daily dashboard."
    assert [w.name for w in loaded.workflows] == ["Sales MIS Workflow"]


def test_stores_load_all_resolves_every_cross_reference_in_order(tmp_path):
    stores = Stores(tmp_path / "store")

    skill = Skill(name="Sanity Check Data", prompt="Check nulls and duplicates.")
    stores.skills.save(skill)

    persona = Persona(name="Data Engineer", instructions="Careful and methodical.")
    persona.add_skill(skill)
    stores.personas.save(persona)

    policy = Policy(name="No PII", statement="Never expose PII.")
    stores.policies.save(policy)

    workflow = Workflow(name="Sales MIS Workflow")
    workflow.add_step("Validate the staged data.", persona=persona)
    workflow.add_policy(policy)
    stores.workflows.save(workflow)

    process = Process(name="Sales MIS Process")
    process.add_workflow(workflow)
    stores.processes.save(process)

    catalogues = stores.load_all()
    assert set(catalogues) == {"skills", "personas", "policies", "tasks", "workflows", "processes"}
    loaded_workflow = catalogues["workflows"]["sales mis workflow"]
    assert loaded_workflow.steps[0].persona.name == "Data Engineer"
    assert catalogues["processes"]["sales mis process"].workflows[0] is loaded_workflow


def test_store_list_reads_names_from_headings_not_filenames(tmp_path):
    store = Store(tmp_path / "objects")
    store.write("Credit Risk Guru", "## Credit Risk Guru\n\nUnderwrite conservatively.\n")
    assert store.list() == ["Credit Risk Guru"]
    assert store.path_for("credit-risk-guru") == store.path_for("Credit Risk Guru")


def test_store_read_missing_name_reports_known_names(tmp_path):
    store = Store(tmp_path / "objects")
    store.write("Known Thing", "## Known Thing\n\nSomething.\n")
    with pytest.raises(FileNotFoundError, match="Known Thing"):
        store.read("Missing Thing")


def test_task_definition_to_step_delegates_to_the_given_persona():
    persona = Persona(name="Data Engineer")
    task = TaskDefinition(name="Sanity Check", instruction="Check the data.")
    step = task.to_step(persona=persona)
    assert step.instruction == "Check the data."
    assert step.persona is persona
    assert step.name == "Sanity Check"


def test_stores_defaults_to_file_backend_with_no_strategy_opt_in(tmp_path):
    from ear.strategy import Strategy

    stores = Stores.from_strategy(tmp_path / "store", Strategy())
    assert stores.backend_name == "file"
    assert isinstance(stores.skills.store, Store)


def test_stores_stays_file_backed_when_store_is_explicitly_false(tmp_path):
    from ear.strategy import Strategy

    strategy = Strategy.from_markdown("## Catalogue Store\n\nStore: false.\n")
    stores = Stores.from_strategy(tmp_path / "store", strategy)
    assert stores.backend_name == "file"


def test_stores_opts_into_postgres_age_when_declared(tmp_path):
    from ear.strategy import Strategy

    strategy = Strategy.from_markdown(
        "## Catalogue Store\n\nStore: true\nBackend: apache-age\n"
        "Connection: `postgresql://ear:secret@localhost/ear`\n"
    )
    assert strategy.catalogue_store_enabled is True
    assert strategy.catalogue_backend == "apache-age"
    assert strategy.catalogue_connection == "postgresql://ear:secret@localhost/ear"

    # psycopg is not installed in this environment, so opting in must fail
    # loudly and immediately with an actionable message -- never silently
    # fall back to file storage once the user has explicitly opted in.
    with pytest.raises(ImportError, match="psycopg"):
        Stores.from_strategy(tmp_path / "store", strategy)


def test_kind_stores_require_either_directory_or_backend():
    with pytest.raises(ValueError, match="directory.*backend"):
        SkillStore()


def test_loader_wires_runtime_stores_from_directory(tmp_path):
    from ear.loader import load_runtime

    (tmp_path / "skills.md").write_text("# Skills\n\n## Greet\n\nSay hello.\n")
    runtime = load_runtime(tmp_path)
    assert runtime.stores is not None
    assert runtime.stores.backend_name == "file"

    skill = Skill(name="Persisted Skill", prompt="Do the thing.")
    runtime.stores.skills.save(skill)
    assert runtime.stores.skills.load("Persisted Skill").prompt == "Do the thing."
