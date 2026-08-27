import { useEffect, useState } from "react";

import type { Bootstrap } from "../types";

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
  const [baseUrl, setBaseUrl] = useState(config.llm.base_url ?? "");
  const [approvalMode, setApprovalMode] = useState(config.policy.approval_mode);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setProvider(config.llm.provider);
    setModel(config.llm.model);
    setBaseUrl(config.llm.base_url ?? "");
    setApprovalMode(config.policy.approval_mode);
  }, [config]);

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
      <aside className="settings-panel" role="dialog" aria-modal="true" aria-label="设置" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div>
            <span className="eyebrow">Local configuration</span>
            <h2>设置</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="关闭设置">×</button>
        </header>

        <div className="settings-body">
          <label>
            预设模型
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
              placeholder={config.llm.api_key === "***" ? "已配置；留空保持不变" : "输入 API Key"}
              autoComplete="off"
            />
          </label>

          <label>
            Base URL
            <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
          </label>

          <fieldset>
            <legend>工具审批</legend>
            <label className="radio-row">
              <input
                type="radio"
                name="approval"
                checked={approvalMode === "ask"}
                onChange={() => setApprovalMode("ask")}
              />
              <span><strong>Ask</strong><small>修改文件和运行危险命令前确认</small></span>
            </label>
            <label className="radio-row">
              <input
                type="radio"
                name="approval"
                checked={approvalMode === "auto"}
                onChange={() => setApprovalMode("auto")}
              />
              <span><strong>Auto</strong><small>由安全策略直接决定是否执行</small></span>
            </label>
          </fieldset>
        </div>

        <footer>
          <button type="button" className="quiet-button" onClick={onClose}>取消</button>
          <button type="button" className="primary-button" onClick={() => void save()} disabled={saving || !provider || !model}>
            {saving ? "保存中…" : "保存并重载"}
          </button>
        </footer>
      </aside>
    </div>
  );
}
