import { Handle, Position } from '@xyflow/react';
import type { NodeData } from '../utils/buildGraph';

const TYPE_COLORS: Record<string, { bg: string; border: string; accent: string }> = {
  request:    { bg: '#0c1629', border: '#3b82f6', accent: '#3b82f6' },
  context:    { bg: '#0c1920', border: '#06b6d4', accent: '#06b6d4' },
  response:   { bg: '#0c1a14', border: '#22c55e', accent: '#22c55e' },
  reflection: { bg: '#1a0f29', border: '#a855f7', accent: '#a855f7' },
  verifier:   { bg: '#1a1408', border: '#f59e0b', accent: '#f59e0b' },
  progress:   { bg: '#0c1a1a', border: '#14b8a6', accent: '#14b8a6' },
  final:      { bg: '#1a0820', border: '#ec4899', accent: '#ec4899' },
};

export default function PipelineNode({ data }: { data: NodeData }) {
  const colors = TYPE_COLORS[data.nodeType] || TYPE_COLORS.request;
  const isBest = data.isBest;

  return (
    <div
      style={{
        background: colors.bg,
        border: `2px solid ${isBest ? '#f59e0b' : colors.border}`,
        borderRadius: 10,
        padding: '10px 14px',
        width: 220,
        fontFamily: 'ui-monospace, SFMono-Regular, monospace',
        fontSize: 12,
        color: '#e2e8f0',
        boxShadow: isBest ? '0 0 12px rgba(245, 158, 11, 0.4)' : `0 0 8px ${colors.border}33`,
        position: 'relative',
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: colors.accent, width: 8, height: 8 }} />

      <div style={{ fontWeight: 700, color: colors.accent, marginBottom: 4, fontSize: 13 }}>
        {data.label}
        {isBest && <span style={{ color: '#f59e0b', marginLeft: 6 }}>★ Best</span>}
      </div>

      {data.model && (
        <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 2 }}>
          {data.model}
        </div>
      )}

      {data.score !== undefined && data.nodeType === 'progress' && (
        <div style={{ marginTop: 4 }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3,
          }}>
            <span style={{ color: '#94a3b8' }}>Progress</span>
            <span style={{
              color: data.score > 0.7 ? '#22c55e' : data.score > 0.4 ? '#f59e0b' : '#ef4444',
              fontWeight: 700,
            }}>
              {Math.round(data.score * 100)}%
            </span>
          </div>
          <div style={{
            width: '100%', height: 6, borderRadius: 3,
            background: '#1e293b', overflow: 'hidden',
          }}>
            <div style={{
              width: `${Math.round(data.score * 100)}%`, height: '100%', borderRadius: 3,
              background: data.score > 0.7 ? '#22c55e' : data.score > 0.4 ? '#f59e0b' : '#ef4444',
            }} />
          </div>
        </div>
      )}

      {data.score !== undefined && data.nodeType !== 'progress' && (
        <div style={{ fontSize: 11 }}>
          Score: <span style={{ color: data.score > 0.7 ? '#22c55e' : data.score > 0.3 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
            {data.score.toFixed(3)}
          </span>
        </div>
      )}

      <Handle type="source" position={Position.Bottom} style={{ background: colors.accent, width: 8, height: 8 }} />
    </div>
  );
}
