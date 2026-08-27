export type AgentMode = "react" | "plan";

export interface Message {
  role: "user" | "assistant" | "tool" | "system";
  content: string | Array<Record<string, unknown>>;
  name?: string | null;
  tool_call_id?: string | null;
  tool_calls?: Array<Record<string, unknown>>;
}

export interface SessionSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  messages?: Message[];
}

export interface TaskSnapshot {
  active: boolean;
  state: string;
  run_id?: string | null;
  approval?: {
    tool_name: string;
    input: Record<string, unknown>;
    danger_level: string;
    description?: string;
  } | null;
  awaiting_plan_review: boolean;
  review_feedback_pending: boolean;
  error?: string | null;
}

export interface PublicConfig {
  llm: {
    provider: string;
    model: string;
    api_key: string;
    base_url?: string | null;
    context_window?: number | null;
    max_tokens: number;
    temperature: number;
  };
  policy: { approval_mode: "ask" | "auto" };
  prompt: { agent_mode: AgentMode };
}

export interface ModelProfile {
  name: string;
  provider: string;
  model: string;
  base_url: string;
  context_window: number;
  description: string;
}

export interface Bootstrap {
  version: string;
  ready: boolean;
  error?: string | null;
  cwd: string;
  trust_required: boolean;
  project_trusted: boolean;
  config: PublicConfig;
  model_profiles: ModelProfile[];
  warnings: string[];
  model?: string;
  provider?: string;
  context_window?: number;
  mode?: AgentMode;
  task?: TaskSnapshot;
  session?: SessionSummary;
  sessions?: SessionSummary[];
  tool_count?: number;
}

export interface ToolActivity {
  id: string;
  name: string;
  input?: Record<string, unknown>;
  result?: string;
  isError?: boolean;
}

export interface PlanTaskActivity {
  id: string;
  description: string;
  status: "pending" | "running" | "completed" | "failed";
}

export interface LiveRun {
  id: string;
  text: string;
  thinking: string;
  tools: ToolActivity[];
  plan: PlanTaskActivity[];
  status: "running" | "completed" | "cancelled" | "failed";
  error?: string;
}

export interface AppState {
  bootstrap: Bootstrap | null;
  liveRun: LiveRun | null;
  connected: boolean;
}

export type RuntimeEvent = Record<string, unknown> & { type: string };
