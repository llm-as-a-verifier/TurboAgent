import { useState } from 'react';
import type { NodeData } from '../utils/buildGraph';

type TabId = 'rendered' | 'pairwise' | 'raw';

export default function NodeDetail({ data, tab: externalTab, onTabChange, onClose }: {
  data: NodeData;
  tab?: string;
  onTabChange: (tab: string) => void;
  onClose: () => void;
}) {
  const hasPairwise = data.nodeType === 'verifier' &&
    (data.detail as Record<string, unknown>).comparisons &&
    ((data.detail as Record<string, unknown>).comparisons as unknown[]).length > 0;

  const tabs: { id: TabId; label: string }[] = [
    { id: 'rendered', label: 'Rendered' },
    ...(hasPairwise ? [{ id: 'pairwise' as TabId, label: 'Pairwise' }] : []),
    { id: 'raw', label: 'Raw JSON' },
  ];

  const tab = (tabs.some(t => t.id === externalTab) ? externalTab : 'rendered') as TabId;
  const setTab = onTabChange;

  return (
    <div style={{
      position: 'absolute', right: 0, top: 0, bottom: 0, width: 420,
      background: '#0f172a', borderLeft: '1px solid #1e293b',
      display: 'flex', flexDirection: 'column', zIndex: 10,
      fontFamily: 'ui-monospace, SFMono-Regular, monospace',
    }}>
      <div style={{
        padding: '12px 16px', borderBottom: '1px solid #1e293b',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 14 }}>{data.label}</span>
        <button onClick={onClose} style={{
          background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: 18,
        }}>x</button>
      </div>

      <div style={{ display: 'flex', borderBottom: '1px solid #1e293b' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            flex: 1, padding: '8px', background: tab === t.id ? '#1e293b' : 'transparent',
            border: 'none', color: tab === t.id ? '#e2e8f0' : '#64748b', cursor: 'pointer',
            fontSize: 12, fontFamily: 'inherit',
          }}>
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        {tab === 'rendered' ? (
          <RenderedView data={data} />
        ) : tab === 'pairwise' ? (
          <PairwiseView data={data} />
        ) : (
          <pre style={{ color: '#94a3b8', fontSize: 11, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {JSON.stringify(data.detail, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

function RenderedView({ data }: { data: NodeData }) {
  const detail = data.detail as Record<string, unknown>;

  if (data.nodeType === 'request') {
    const messages = (detail.messages as Array<{ role: string; content: unknown }>) || [];
    return (
      <div>
        <SectionHeader>Messages ({messages.length})</SectionHeader>
        {messages.map((m, i) => (
          <MessageBlock key={i} role={m.role} content={m.content} />
        ))}
      </div>
    );
  }

  if (data.nodeType === 'response') {
    const resp = (detail as { model: string; response: { choices: Array<{ message: { content: string | null; tool_calls?: unknown[] } }>; usage?: Record<string, number> } });
    const msg = resp.response?.choices?.[0]?.message;
    const usage = resp.response?.usage;
    return (
      <div>
        <SectionHeader>Model: {resp.model}</SectionHeader>
        {usage && (
          <div style={{ color: '#64748b', fontSize: 11, marginBottom: 8 }}>
            Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out / {usage.total_tokens} total
          </div>
        )}
        {msg?.content && (
          <>
            <SectionHeader>Content</SectionHeader>
            <ContentBlock>{msg.content}</ContentBlock>
          </>
        )}
        {msg?.tool_calls && (msg.tool_calls as unknown[]).length > 0 && (
          <>
            <SectionHeader>Tool Calls</SectionHeader>
            <pre style={{ color: '#f59e0b', fontSize: 11, whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(msg.tool_calls, null, 2)}
            </pre>
          </>
        )}
      </div>
    );
  }

  if (data.nodeType === 'reflection') {
    const action = detail as { original: string; reflected: string };
    return (
      <div>
        <SectionHeader>Original</SectionHeader>
        <ContentBlock>{action.original}</ContentBlock>
        <SectionHeader>Reflected</SectionHeader>
        <ContentBlock>{action.reflected}</ContentBlock>
      </div>
    );
  }

  if (data.nodeType === 'verifier') {
    const v = detail as { scores?: Array<{ index: number; model: string; score: number }>; bestIndex?: number; bestModel?: string; bestScore?: number };
    return (
      <div>
        <SectionHeader>Best: #{v.bestIndex} ({v.bestModel}) — Score: {v.bestScore?.toFixed(3)}</SectionHeader>
        {v.scores?.map((s, i) => (
          <div key={i} style={{
            padding: '8px 10px', marginBottom: 6, borderRadius: 6,
            background: s.index === v.bestIndex ? '#1a2e05' : '#0f172a',
            border: `1px solid ${s.index === v.bestIndex ? '#22c55e' : '#1e293b'}`,
          }}>
            <span style={{ color: '#e2e8f0' }}>#{s.index} {s.model}</span>
            <span style={{
              float: 'right', fontWeight: 700,
              color: s.score > 0.7 ? '#22c55e' : s.score > 0.3 ? '#f59e0b' : '#ef4444',
            }}>
              {s.score.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    );
  }

  if (data.nodeType === 'progress') {
    const pm = detail as {
      enabled: boolean;
      score?: number;
      details?: { score: number; rawProbs: Record<string, number>; generatedText: string };
      error?: string;
    };
    const score = pm.score ?? pm.details?.score ?? 0;
    const pct = Math.round(score * 100);
    const barColor = score > 0.7 ? '#22c55e' : score > 0.4 ? '#f59e0b' : '#ef4444';
    return (
      <div>
        <SectionHeader>Progress Score</SectionHeader>
        {/* Progress bar */}
        <div style={{ marginBottom: 12 }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
            marginBottom: 6,
          }}>
            <span style={{ color: barColor, fontWeight: 700, fontSize: 24 }}>{pct}%</span>
            <span style={{ color: '#94a3b8', fontSize: 11 }}>
              {score.toFixed(3)} / 1.000
            </span>
          </div>
          <div style={{
            width: '100%', height: 12, borderRadius: 6,
            background: '#1e293b', overflow: 'hidden',
          }}>
            <div style={{
              width: `${pct}%`, height: '100%', borderRadius: 6,
              background: `linear-gradient(90deg, ${barColor}cc, ${barColor})`,
              transition: 'width 0.3s ease',
            }} />
          </div>
        </div>
        {/* Raw probability distribution */}
        {pm.details?.rawProbs && Object.keys(pm.details.rawProbs).length > 0 && (
          <>
            <SectionHeader>Score Distribution</SectionHeader>
            <div style={{
              display: 'flex', gap: 4, alignItems: 'flex-end',
              height: 80, padding: '8px 10px',
              background: '#020617', borderRadius: 6, border: '1px solid #1e293b',
              marginBottom: 8,
            }}>
              {Array.from({ length: 9 }, (_, i) => {
                const key = String(i + 1);
                const prob = pm.details!.rawProbs[key] ?? 0;
                const maxProb = Math.max(...Object.values(pm.details!.rawProbs), 0.01);
                const heightPct = (prob / maxProb) * 100;
                return (
                  <div key={key} style={{
                    flex: 1, display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'flex-end', height: '100%',
                  }}>
                    <div style={{
                      width: '100%', borderRadius: 3,
                      background: prob > 0 ? barColor : '#1e293b',
                      height: `${Math.max(heightPct, 2)}%`,
                      opacity: prob > 0 ? 1 : 0.3,
                      minHeight: 2,
                    }} />
                    <span style={{ color: '#64748b', fontSize: 9, marginTop: 3 }}>{key}</span>
                  </div>
                );
              })}
            </div>
          </>
        )}
        {pm.details?.generatedText && (
          <>
            <SectionHeader>Model Output</SectionHeader>
            <ContentBlock>{pm.details.generatedText}</ContentBlock>
          </>
        )}
        {pm.error && (
          <>
            <SectionHeader>Error</SectionHeader>
            <div style={{
              padding: '8px 10px', borderRadius: 6,
              background: '#1a0505', border: '1px solid #ef4444',
              color: '#fca5a5', fontSize: 12, whiteSpace: 'pre-wrap',
            }}>
              {pm.error}
            </div>
          </>
        )}
      </div>
    );
  }

  if (data.nodeType === 'final') {
    const resp = detail as { choices: Array<{ message: { content: string | null; tool_calls?: unknown[] } }>; usage?: Record<string, number>; model: string };
    const msg = resp.choices?.[0]?.message;
    return (
      <div>
        <SectionHeader>Model: {resp.model}</SectionHeader>
        {resp.usage && (
          <div style={{ color: '#64748b', fontSize: 11, marginBottom: 8 }}>
            Tokens: {resp.usage.prompt_tokens} in / {resp.usage.completion_tokens} out
          </div>
        )}
        {msg?.content && (
          <>
            <SectionHeader>Content</SectionHeader>
            <ContentBlock>{msg.content}</ContentBlock>
          </>
        )}
        {msg?.tool_calls && (msg.tool_calls as unknown[]).length > 0 && (
          <>
            <SectionHeader>Tool Calls</SectionHeader>
            <pre style={{ color: '#f59e0b', fontSize: 11, whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(msg.tool_calls, null, 2)}
            </pre>
          </>
        )}
      </div>
    );
  }

  // Generic fallback
  return (
    <pre style={{ color: '#94a3b8', fontSize: 11, whiteSpace: 'pre-wrap' }}>
      {JSON.stringify(detail, null, 2)}
    </pre>
  );
}

interface PairwiseComparisonData {
  i: number;
  j: number;
  rating_A: number;
  rating_B: number;
  winner: string;
  request: Array<{ role: string; content: string }>;
  text: string;
  reverse_request?: Array<{ role: string; content: string }>;
  reverse_text?: string;
  fwd_rating_A?: number;
  fwd_rating_B?: number;
  rev_rating_A?: number;
  rev_rating_B?: number;
}

function PairwiseView({ data }: { data: NodeData }) {
  const detail = data.detail as {
    comparisons?: PairwiseComparisonData[];
    scores?: Array<{ index: number; model: string; score: number }>;
  };
  const comparisons = detail.comparisons || [];
  const scores = detail.scores || [];
  const [expanded, setExpanded] = useState<number | null>(null);

  const modelLabel = (idx: number) => {
    const s = scores.find(s => s.index === idx);
    return s ? `#${idx} ${s.model}` : `#${idx}`;
  };

  return (
    <div>
      <SectionHeader>Pairwise Comparisons ({comparisons.length})</SectionHeader>
      {comparisons.map((c, ci) => {
        const isExpanded = expanded === ci;
        const winnerColor = c.winner === 'A' ? '#22c55e' : c.winner === 'B' ? '#ef4444' : '#f59e0b';
        const winnerLabel = c.winner === 'A' ? modelLabel(c.i) : c.winner === 'B' ? modelLabel(c.j) : 'Tie';

        return (
          <div key={ci} style={{
            marginBottom: 8, borderRadius: 6,
            border: '1px solid #1e293b', overflow: 'hidden',
          }}>
            <div
              onClick={() => setExpanded(isExpanded ? null : ci)}
              style={{
                padding: '10px 12px', cursor: 'pointer',
                background: '#0a0f1a',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}
            >
              <div style={{ color: '#e2e8f0', fontSize: 12 }}>
                <span style={{ color: '#94a3b8' }}>{modelLabel(c.i)}</span>
                <span style={{ color: '#64748b', margin: '0 6px' }}>vs</span>
                <span style={{ color: '#94a3b8' }}>{modelLabel(c.j)}</span>
              </div>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', fontSize: 11 }}>
                <span>
                  A: <span style={{ color: c.winner === 'A' ? '#22c55e' : '#e2e8f0', fontWeight: 600 }}>{c.rating_A.toFixed(1)}</span>
                </span>
                <span>
                  B: <span style={{ color: c.winner === 'B' ? '#22c55e' : '#e2e8f0', fontWeight: 600 }}>{c.rating_B.toFixed(1)}</span>
                </span>
                <span style={{
                  color: winnerColor, fontWeight: 700, fontSize: 11,
                  padding: '2px 6px', borderRadius: 4,
                  background: `${winnerColor}15`,
                }}>
                  {winnerLabel}
                </span>
                <span style={{ color: '#64748b', fontSize: 10 }}>{isExpanded ? '\u25B2' : '\u25BC'}</span>
              </div>
            </div>

            {isExpanded && (
              <div style={{ padding: '10px 12px', background: '#020617', borderTop: '1px solid #1e293b' }}>
                {c.reverse_request ? (
                  <>
                    <SectionHeader>Forward ({modelLabel(c.i)} as A, {modelLabel(c.j)} as B)</SectionHeader>
                    <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 6 }}>
                      A: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{c.fwd_rating_A?.toFixed(1)}</span>
                      {' '}B: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{c.fwd_rating_B?.toFixed(1)}</span>
                    </div>
                    <JudgeCallDetail request={c.request} text={c.text} />

                    <SectionHeader>Reverse ({modelLabel(c.j)} as A, {modelLabel(c.i)} as B)</SectionHeader>
                    <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 6 }}>
                      A: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{c.rev_rating_A?.toFixed(1)}</span>
                      {' '}B: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{c.rev_rating_B?.toFixed(1)}</span>
                    </div>
                    <JudgeCallDetail request={c.reverse_request} text={c.reverse_text || ''} />

                    <SectionHeader>Averaged</SectionHeader>
                    <div style={{ color: '#94a3b8', fontSize: 11 }}>
                      {modelLabel(c.i)}: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{c.rating_A.toFixed(1)}</span>
                      {' '}{modelLabel(c.j)}: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{c.rating_B.toFixed(1)}</span>
                    </div>
                  </>
                ) : (
                  <JudgeCallDetail request={c.request} text={c.text} />
                )}
              </div>
            )}
          </div>
        );
      })}

      {comparisons.length === 0 && (
        <div style={{ color: '#64748b', fontSize: 12, padding: 8 }}>
          No pairwise comparison data available.
        </div>
      )}
    </div>
  );
}

function JudgeCallDetail({ request, text }: { request: Array<{ role: string; content: string }>; text: string }) {
  return (
    <div style={{ marginBottom: 10 }}>
      {request && request.length > 0 && (
        <>
          <div style={{ color: '#64748b', fontSize: 10, fontWeight: 600, marginBottom: 4 }}>REQUEST</div>
          {request.map((msg, mi) => (
            <div key={mi} style={{
              marginBottom: 6, padding: '6px 10px', borderRadius: 6,
              background: '#0f172a', border: '1px solid #1e293b',
            }}>
              <div style={{ color: msg.role === 'system' ? '#06b6d4' : '#3b82f6', fontWeight: 700, fontSize: 10, marginBottom: 3 }}>
                {msg.role.toUpperCase()}
              </div>
              <div style={{
                color: '#94a3b8', fontSize: 11, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                maxHeight: 300, overflow: 'auto',
              }}>
                {msg.content}
              </div>
            </div>
          ))}
        </>
      )}
      <div style={{ color: '#64748b', fontSize: 10, fontWeight: 600, marginBottom: 4 }}>RESPONSE</div>
      <div style={{
        color: '#cbd5e1', fontSize: 11, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
        maxHeight: 400, overflow: 'auto',
        padding: '8px 10px', borderRadius: 6, background: '#0f172a',
        border: '1px solid #1e293b',
      }}>
        {text || '(no text)'}
      </div>
    </div>
  );
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ color: '#94a3b8', fontSize: 11, fontWeight: 700, marginTop: 12, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
      {children}
    </div>
  );
}

function MessageBlock({ role, content }: { role: string; content: unknown }) {
  const roleColor = role === 'user' ? '#3b82f6' : role === 'assistant' ? '#22c55e' : '#94a3b8';
  let text = '';
  if (typeof content === 'string') {
    text = content;
  } else if (Array.isArray(content)) {
    text = (content as Array<{ type: string; text?: string }>)
      .filter(b => b.type === 'text')
      .map(b => b.text || '')
      .join('\n');
  }
  // Strip system reminders for readability
  text = text.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '').trim();
  if (!text) return null;

  return (
    <div style={{ marginBottom: 10, padding: '8px 10px', borderRadius: 6, background: '#020617', border: '1px solid #1e293b' }}>
      <div style={{ color: roleColor, fontWeight: 700, fontSize: 11, marginBottom: 4 }}>{role.toUpperCase()}</div>
      <div style={{ color: '#cbd5e1', fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{text.slice(0, 2000)}</div>
    </div>
  );
}

function ContentBlock({ children }: { children: string }) {
  return (
    <div style={{
      padding: '8px 10px', borderRadius: 6, background: '#020617', border: '1px solid #1e293b',
      color: '#cbd5e1', fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-word', marginBottom: 8,
    }}>
      {children.slice(0, 3000)}
    </div>
  );
}
