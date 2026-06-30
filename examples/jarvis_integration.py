"""Integration sketch: swap an existing OpenRouter call for the drop-in client.

Before (typical agent code):

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)
    resp = client.chat.completions.create(model="anthropic/claude-3.5-sonnet",
                                          messages=messages)

After (router picks the model per request, logs the decision):

    from smartrouter import RouterClient, RouterConfig
    client = RouterClient(RouterConfig.from_yaml("router.yaml"))
    resp = client.chat.completions.create(messages=messages)   # no model= needed
    print(resp.choices[0].message.content, resp.routing.tier)

Per-request escape hatches keep an agent's control:

    # A quick intent classification can stay local and free:
    client.chat.completions.create(messages=msgs, local_only=True)

    # A user-facing final answer can be pinned to the best model:
    client.chat.completions.create(messages=msgs, force_tier="frontier")

Closing the loop — when a downstream check fails, label it so the router learns:

    resp = client.chat.completions.create(messages=msgs,
                                          response_format={"type": "json_object"})
    if not parsed_ok(resp.choices[0].message.content):
        client.feedback(resp.decision_id, label=1)   # this one was harder than scored
"""
