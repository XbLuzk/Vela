import { describe, expect, it } from "vitest";

import { messageForSubmission } from "./Composer";

describe("messageForSubmission", () => {
  it("accepts a trimmed message when the composer is ready", () => {
    expect(messageForSubmission("  inspect this file  ", false, false, false)).toBe("inspect this file");
  });

  it("rejects a second submission while the first request is pending", () => {
    expect(messageForSubmission("inspect this file", false, false, true)).toBeNull();
  });

  it("rejects submissions while a task is active", () => {
    expect(messageForSubmission("inspect this file", true, false, false)).toBeNull();
  });
});
