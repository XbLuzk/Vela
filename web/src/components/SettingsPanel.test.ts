import { describe, expect, it } from "vitest";

import { resolveBaseUrl } from "./SettingsPanel";
import type { ModelProfile } from "../types";

const profiles: ModelProfile[] = [
  {
    name: "DeepSeek V4 Flash",
    provider: "deepseek",
    model: "deepseek-v4-flash",
    base_url: "https://api.deepseek.com/v1",
    context_window: 1_000_000,
    description: "",
  },
];

describe("settings defaults", () => {
  it("uses the selected model profile when the saved Base URL is empty", () => {
    expect(resolveBaseUrl(null, "deepseek", "deepseek-v4-flash", profiles)).toBe(
      "https://api.deepseek.com/v1",
    );
  });

  it("keeps an explicitly configured Base URL", () => {
    expect(
      resolveBaseUrl(
        "https://proxy.example/v1",
        "deepseek",
        "deepseek-v4-flash",
        profiles,
      ),
    ).toBe("https://proxy.example/v1");
  });
});
