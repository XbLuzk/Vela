import { useEffect, useState } from "react";

import type { Bootstrap, ModelProfile } from "../types";
import { CloseIcon } from "./Icons";

export function resolveBaseUrl(
  configured: string | null | undefined,
  provider: string,
  model: string,
  profiles: ModelProfile[],
): string {
  const explicit = configured?.trim();
  if (explicit) return explicit;
  return (
    profiles.find((profile) => profile.provider === provider && profile.model === model)?.base_url ?? ""
  );
}

interface SettingsPanelProps {
  open: boolean;
  bootstrap: Bootstrap;
  onClose: () => void;
  onSave: (settings: Record<string, unknown>) => Promise<boolean>;
}

export function SettingsPanel({ open, bootstrap, onClose, onSave }: SettingsPanelProps) {
  const config = bootstrap.config;
  const [provider, setProvider] = useState(config.llm.provider);
  const [model, setModel] = useState(config.llm.model);
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState(
    resolveBaseUrl(config.llm.base_url, config.llm.provider, config.llm.model, bootstrap.model_profiles),
  );
  const [approvalMode, setApprovalMode] = useState(config.policy.approval_mode);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setProvider(config.llm.provider);
    setModel(config.llm.model);
    setBaseUrl(
      resolveBaseUrl(
        config.llm.base_url,
        config.llm.provider,
        config.llm.model,
        bootstrap.model_profiles,
      ),
    );
    setApprovalMode(config.policy.approval_mode);
  }, [bootstrap.model_profiles, config]);

  if (!open) return null;

  async function save() {
    setSaving(true);
    try {
      const ready = await onSave({
        provider,
        model,
        api_key: apiKey || undefined,
        base_url: baseUrl,
        approval_mode: approvalMode,
      });
      if (ready) {
        setApiKey("");
        onClose();
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="panel-backdrop" role="presentation" onMouseDown={onClose}>
      <aside className="settings-panel" role="dialog" aria-modal="true" aria-label="Settings" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div>
            <span className="eyebrow">Local configuration</span>
            <h2>Settings</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close settings"><CloseIcon /></button>
        </header>

        <div className="settings-body">
          <label>
            Model preset
            <select
              value={`${provider}/${model}`}
              onChange={(event) => {
                const profile = bootstrap.model_profiles.find(
                  (item) => `${item.provider}/${item.model}` === event.target.value,
                );
                if (!profile) return;
                setProvider(profile.provider);
                setModel(profile.model);
                setBaseUrl(profile.base_url);
              }}
            >
              {bootstrap.model_profiles.map((profile) => (
                <option key={`${profile.provider}/${profile.model}`} value={`${profile.provider}/${profile.model}`}>
                  {profile.name}
                </option>
              ))}
              {!bootstrap.model_profiles.some(
                (profile) => profile.provider === provider && profile.model === model,
              ) ? <option value={`${provider}/${model}`}>{model}</option> : null}
            </select>
          </label>

          <div className="field-row">
            <label>
              Provider
              <input value={provider} onChange={(event) => setProvider(event.target.value)} />
            </label>
            <label>
              Model
              <input value={model} onChange={(event) => setModel(event.target.value)} />
            </label>
          </div>

          <label>
            API Key
            <input
              type="password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder={config.llm.api_key === "***" ? "Configured; leave blank to keep it" : "Enter API key"}
              autoComplete="off"
            />
          </label>

          <label>
            Base URL
            <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
          </label>

          <fieldset>
            <legend>Tool approval</legend>
            <label className="radio-row">
              <input
                type="radio"
                name="approval"
                checked={approvalMode === "ask"}
                onChange={() => setApprovalMode("ask")}
              />
              <span><strong>Ask</strong><small>Confirm before file changes or risky commands</small></span>
            </label>
            <label className="radio-row">
              <input
                type="radio"
                name="approval"
                checked={approvalMode === "auto"}
                onChange={() => setApprovalMode("auto")}
              />
              <span><strong>Auto</strong><small>Let the safety policy decide automatically</small></span>
            </label>
          </fieldset>
        </div>

        <footer>
          <button type="button" className="quiet-button" onClick={onClose}>Cancel</button>
          <button type="button" className="primary-button" onClick={() => void save()} disabled={saving || !provider || !model}>
            {saving ? "Saving…" : "Save and reload"}
          </button>
        </footer>
      </aside>
    </div>
  );
}
