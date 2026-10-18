"""Conversation routes exercised through the provider lifecycle and memory tools."""

import json

import pytest

from conftest import FakeClient


@pytest.mark.parametrize(
    "local,platform,chat_id,thread_id,expected",
    [
        (False, "slack", "C123", "171.234", "project-C123-171-234"),
        (True, "slack", "C123", "171.234", "project-C123-171-234"),
        (False, "telegram", "", "", "project"),
        (True, "telegram", "chat", "", "project-chat"),
    ],
)
def test_chat_and_thread_templates(provider, local, platform, chat_id, thread_id, expected):
    template = "project-{chat}-{thread}"
    config = {"bank_id_template": template}
    if local:
        config = {"channel_routes": [{"platform": platform, "chat_id": chat_id, "bank_id_template": template}]}
    instance, fake = provider(config, platform=platform, chat_id=chat_id, thread_id=thread_id)
    try:
        instance.handle_tool_call("hindsight_retain", {"content": "fact"})
        instance.handle_tool_call("hindsight_recall", {"query": "fact"})
        assert [call["bank_id"] for call in fake.retains] == [expected]
        assert [call["bank_id"] for call in fake.recalls] == [expected]
    finally:
        instance.shutdown()


@pytest.mark.parametrize("platform", ["telegram", "cli", "desktop"])
def test_project_templates_with_and_without_matching_route(provider, tmp_path, platform):
    repository = tmp_path / "project"
    (repository / ".git").mkdir(parents=True)
    instance, fake = provider(
        {
            "bank_id_template": "global" if platform == "telegram" else "hermes-{project}",
            "channel_routes": [{"platform": "telegram", "chat_id": "chat", "bank_id_template": "hermes-{project}"}],
        },
        platform=platform,
        chat_id="chat",
        cwd=str(repository),
    )
    try:
        instance.handle_tool_call("hindsight_retain", {"content": "fact"})
        assert [call["bank_id"] for call in fake.retains] == ["hermes-project"]
    finally:
        instance.shutdown()


@pytest.mark.parametrize(
    "template,fallback,source",
    [
        ("{unknown}", "repository", "repository config"),
        ("{", "global-session-1", "template"),
        ("{user}", "personal", "bank_id"),
    ],
)
def test_invalid_route_template_falls_back_and_pins_bank(provider, tmp_path, caplog, template, fallback, source):
    caplog.set_level("INFO")
    repository = tmp_path / "project"
    (repository / ".git").mkdir(parents=True)
    (repository / ".hindsight").mkdir()
    (repository / ".hindsight" / "config.toml").write_text('bank_id = "repository"\n')
    instance, fake = provider(
        {
            "bank_id": "personal",
            "bank_id_template": "global-{session}" if fallback != "personal" else "",
            "trusted_project_dirs": [str(tmp_path)] if fallback == "repository" else [],
            "recall_additional_banks": ["inherited"],
            "channel_routes": [
                {"platform": "telegram", "chat_id": "chat", "bank_id": "whole-chat"},
                {
                    "platform": "telegram",
                    "chat_id": "chat",
                    "thread_id": "7",
                    "bank_id_template": template,
                    "recall_additional_banks": ["shared"],
                },
            ],
        },
        platform="telegram",
        chat_id="chat",
        thread_id="7",
        cwd=str(repository),
    )
    try:
        assert f"using {source} bank {fallback!r}" in caplog.text
        assert f"bank={fallback}, bank_source=channel route fallback to {source}" in caplog.text
        instance.on_session_switch("session-2", reset=True)
        instance.on_session_switch("resumed")
        instance.handle_tool_call("hindsight_retain", {"content": "fact"})
        instance.handle_tool_call("hindsight_recall", {"query": "fact"})
        assert [call["bank_id"] for call in fake.retains] == [fallback]
        assert [call["bank_id"] for call in fake.recalls] == [fallback, "shared"]
    finally:
        instance.shutdown()


@pytest.mark.parametrize(
    "destination", [{}, {"bank_id": "a", "bank_id_template": "b"}, {"bank_id": ""}, {"bank_id_template": " "}]
)
def test_route_requires_one_nonempty_destination(provider, destination):
    with pytest.raises(ValueError):
        provider({"channel_routes": [{"platform": "telegram", "chat_id": "chat", **destination}]})


@pytest.mark.parametrize(
    "chat_id,expected", [("parent", "parent-bank"), ("thread", "thread-bank"), ("other-thread", "personal")]
)
def test_discord_threads_use_their_own_chat_id(provider, chat_id, expected):
    instance, fake = provider(
        {
            "bank_id": "personal",
            "channel_routes": [
                {"platform": "discord", "chat_id": "parent", "bank_id": "parent-bank"},
                {"platform": "discord", "chat_id": "thread", "bank_id": "thread-bank"},
            ],
        },
        platform="discord",
        chat_id=chat_id,
    )
    try:
        instance.handle_tool_call("hindsight_retain", {"content": "fact"})
        assert [call["bank_id"] for call in fake.retains] == [expected]
    finally:
        instance.shutdown()


@pytest.mark.parametrize(
    "platform,chat_id,thread_id,expected_bank",
    [
        ("telegram", "-100", "7", "automation"),
        ("telegram", "-100", "8", "automation"),
        ("telegram", "-100", "9", "group-9"),
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
            "channel_routes": [
                {"platform": "telegram", "chat_id": "-100", "thread_id": "7", "bank_id": "automation"},
                {"platform": "telegram", "chat_id": "-100", "bank_id_template": "group-{thread}"},
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
            "channel_routes": [
                {
                    "platform": "telegram",
                    "chat_id": "-100",
                    "bank_id": "whole-chat",
                    "recall_additional_banks": ["whole-chat-only"],
                },
                {"platform": "telegram", "chat_id": "-100", "thread_id": "7", "bank_id": "project", **route_settings},
            ],
        },
        client=FakeClient(recall_texts=["a shared fact"], reflect_text="project reflection"),
        platform="telegram",
        chat_id="-100",
        thread_id="7",
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


@pytest.mark.parametrize("destination", [{"bank_id": "project"}, {"bank_id_template": "project-{session}"}])
def test_route_survives_session_resets_resumes_and_provider_recreation(provider, destination):
    config = {
        "bank_id_template": "template-{session}",
        "retain_every_n_turns": 2,
        "channel_routes": [{"platform": "telegram", "chat_id": "-100", "thread_id": "7", **destination}],
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
    expected = "project" if "bank_id" in destination else "project-session-1"
    assert [call["bank_id"] for call in fake.retains] == [expected] * 3
    assert [call["document_id"] for call in fake.retains] == ["session-1", "session-2", "resumed-session"]

    recreated, new_client = provider(
        config, session_id="resumed-session", platform="telegram", chat_id="-100", thread_id="7"
    )
    try:
        recreated.handle_tool_call("hindsight_retain", {"content": "after restart"})
        expected = "project" if "bank_id" in destination else "project-resumed-session"
        assert [call["bank_id"] for call in new_client.retains] == [expected]
    finally:
        recreated.shutdown()


@pytest.mark.parametrize("thread_settings", [{}, {"thread_id": "7"}])
def test_duplicate_route_identities_are_rejected(provider, thread_settings):
    route = {"platform": "telegram", "chat_id": "-100", "bank_id": "project", **thread_settings}
    with pytest.raises(ValueError, match="Duplicate channel route"):
        provider({"channel_routes": [route, {**route, "bank_id": "other-project"}]})


def test_selected_bank_failure_does_not_write_to_a_fallback(provider):
    class UnavailableClient(FakeClient):
        async def aretain_batch(self, **kwargs):
            await super().aretain_batch(**kwargs)
            raise RuntimeError("selected bank unavailable")

    instance, fake = provider(
        {
            "bank_id": "personal",
            "additional_banks": ["shared"],
            "channel_routes": [{"platform": "telegram", "chat_id": "-100", "bank_id": "project"}],
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
