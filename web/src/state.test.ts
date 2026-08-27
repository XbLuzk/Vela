import { describe, expect, it } from "vitest";

import { initialState, reducer } from "./state";
import type { Bootstrap } from "./types";

const bootstrap: Bootstrap = {
  version: "0.5.0",
  ready: true,
  cwd: "/workspace",
  project_extensions_pending: false,
  project_trusted: true,
  config: {
    llm: {
      provider: "deepseek",
      model: "deepseek-chat",
      api_key: "***",
      max_tokens: 8192,
      temperature: 0.7,
    },
    policy: { approval_mode: "ask" },
    prompt: { agent_mode: "react" },
  },
  model_profiles: [],
  warnings: [],
  session: {
    id: "session-1",
    title: "Demo",
    created_at: "now",
    updated_at: "now",
    message_count: 0,
    messages: [],
  },
};

describe("runtime event reducer", () => {
  it("builds one live run from streaming agent events", () => {
    let state = reducer(initialState, { type: "loaded", bootstrap });
    state = reducer(state, {
      type: "event",
      event: {
        type: "user_message",
        run_id: "run-1",
        message: { role: "user", content: "hello" },
      },
    });
    state = reducer(state, {
      type: "event",
      event: { type: "text_delta", run_id: "run-1", text: "Hi" },
    });
    state = reducer(state, {
      type: "event",
      event: {
        type: "tool_call",
        run_id: "run-1",
        tool_call_id: "call-1",
        name: "read_file",
        input: { path: "README.md" },
      },
    });
    state = reducer(state, {
      type: "event",
      event: {
        type: "tool_result",
        run_id: "run-1",
        tool_call_id: "call-1",
        result: "Vela",
      },
    });

    expect(state.bootstrap?.session?.messages?.[0].content).toBe("hello");
    expect(state.liveRun?.text).toBe("Hi");
    expect(state.liveRun?.tools[0]).toMatchObject({
      name: "read_file",
      result: "Vela",
    });
  });

  it("replaces optimistic history with persisted session", () => {
    const loaded = reducer(initialState, { type: "loaded", bootstrap });
    const state = reducer(loaded, {
      type: "event",
      event: {
        type: "session_updated",
        session: {
          ...bootstrap.session,
          message_count: 2,
          messages: [
            { role: "user", content: "hello" },
            { role: "assistant", content: "Hi" },
          ],
        },
      },
    });

    expect(state.liveRun).toBeNull();
    expect(state.bootstrap?.session?.messages).toHaveLength(2);
  });
});
