"""Conversation routes exercised through the provider lifecycle and memory tools."""

import json

import pytest

from conftest import FakeClient


@pytest.mark.parametrize(
    "platform,chat_id,thread_id,expected_bank",
    [
        ("telegram", "-100", "7", "automation"),
        ("telegram", "-100", "8", "automation"),
        ("telegram", "-100", "9", "group"),
        ("telegram", "-100", "", "group"),
        ("telegram", "100", "7", "other-project"),
        ("telegram", "-200", "7", "repository"),
        ("discord", "-100", "7", "repository"),
    ],
)
def test_routes_choose_the_bank_before_repository_and_template_settings(
    provider, tmp_path, platform, chat_id, thread_id, expected_bank
):
    repository = tmp_path / "project"
    (repository / ".git").mkdir(parents=True)
    (repository / ".hindsight").mkdir()
    (repository / ".hindsight" / "config.toml").write_text('bank_id = "repository"\n')
    instance, fake = provider(
        {
            "bank_id": "personal",
            "bank_id_template": "template-{session}",
            "trusted_project_dirs": [str(tmp_path)],
            "bank_routes": [
                {"platform": "telegram", "chat_id": "-100", "thread_id": "7", "bank_id": "automation"},
                {"platform": "telegram", "chat_id": "-100", "bank_id": "group"},
                {"platform": "telegram", "chat_id": "-100", "thread_id": "8", "bank_id": "automation"},
                {"platform": "telegram", "chat_id": "100", "thread_id": "7", "bank_id": "other-project"},
            ],
        },
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        cwd=str(repository),
    )
    try:
        instance.handle_tool_call("hindsight_retain", {"content": "a project fact"})
        instance.handle_tool_call("hindsight_recall", {"query": "project facts"})
        assert [call["bank_id"] for call in fake.retains] == [expected_bank]
        assert [call["bank_id"] for call in fake.recalls] == [expected_bank]
    finally:
        instance.shutdown()


@pytest.mark.parametrize(
    "route_settings,expected_recall_banks",
    [
        ({}, ["project", "global"]),
        ({"recall_additional_banks": ["shared", "project"]}, ["project", "shared"]),
        ({"recall_additional_banks": []}, ["project"]),
    ],
)
def test_route_recall_banks_apply_to_tools_and_automatic_recall_without_receiving_writes(
    provider, route_settings, expected_recall_banks
):
    instance, fake = provider(
        {
            "bank_id": "personal",
            "recall_sync": True,
            "recall_additional_banks": ["global"],
            "bank_routes": [{"platform": "telegram", "chat_id": "-100", "bank_id": "project", **route_settings}],
        },
        client=FakeClient(recall_texts=["a shared fact"], reflect_text="project reflection"),
        platform="telegram",
        chat_id="-100",
    )
    try:
        result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "facts"}))
        assert result == {"result": "1. a shared fact"}
        assert "- a shared fact" in instance.prefetch("facts")
        assert [call["bank_id"] for call in fake.recalls] == expected_recall_banks * 2
        result = json.loads(instance.handle_tool_call("hindsight_reflect", {"query": "facts"}))
        assert result == {"result": "project reflection"}
        assert [call["bank_id"] for call in fake.reflects] == ["project"]
        instance.handle_tool_call("hindsight_retain", {"content": "an explicit fact"})
        instance.sync_turn("a project discussion", "a reply")
    finally:
        instance.shutdown()
    assert [call["bank_id"] for call in fake.retains] == ["project", "project"]


def test_route_survives_session_resets_resumes_and_provider_recreation(provider):
    config = {
        "bank_id_template": "template-{session}",
        "retain_every_n_turns": 2,
        "bank_routes": [{"platform": "telegram", "chat_id": "-100", "thread_id": "7", "bank_id": "project"}],
    }
    instance, fake = provider(config, platform="telegram", chat_id="-100", thread_id="7")
    try:
        instance.sync_turn("before reset", "reply")
        instance.on_session_switch("session-2", reset=True)
        instance.sync_turn("after reset", "reply")
        instance.on_session_switch("resumed-session")
        instance.sync_turn("after resume", "reply")
    finally:
        instance.shutdown()
    assert [call["bank_id"] for call in fake.retains] == ["project"] * 3
    assert [call["document_id"] for call in fake.retains] == ["session-1", "session-2", "resumed-session"]

    recreated, new_client = provider(
        config, session_id="resumed-session", platform="telegram", chat_id="-100", thread_id="7"
    )
    try:
        recreated.handle_tool_call("hindsight_retain", {"content": "after restart"})
        assert [call["bank_id"] for call in new_client.retains] == ["project"]
    finally:
        recreated.shutdown()


@pytest.mark.parametrize("thread_settings", [{}, {"thread_id": "7"}])
def test_duplicate_route_identities_are_rejected(provider, thread_settings):
    route = {"platform": "telegram", "chat_id": "-100", "bank_id": "project", **thread_settings}
    with pytest.raises(ValueError, match="Duplicate bank route"):
        provider({"bank_routes": [route, {**route, "bank_id": "other-project"}]})


def test_selected_bank_failure_does_not_write_to_a_fallback(provider):
    class UnavailableClient(FakeClient):
        async def aretain_batch(self, **kwargs):
            await super().aretain_batch(**kwargs)
            raise RuntimeError("selected bank unavailable")

    instance, fake = provider(
        {
            "bank_id": "personal",
            "additional_banks": ["shared"],
            "bank_routes": [{"platform": "telegram", "chat_id": "-100", "bank_id": "project"}],
        },
        client=UnavailableClient(),
        platform="telegram",
        chat_id="-100",
    )
    try:
        result = instance.handle_tool_call("hindsight_retain", {"content": "a fact"})
        assert "Failed to store memory: selected bank unavailable" in result
        assert [call["bank_id"] for call in fake.retains] == ["project"]
    finally:
        instance.shutdown()
