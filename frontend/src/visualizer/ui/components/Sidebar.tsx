import { useEffect, useState } from 'react';
import type { EntryListItem } from '../types';

export default function Sidebar({
  selectedId,
  onSelect,
}: {
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [entries, setEntries] = useState<EntryListItem[]>([]);

  useEffect(() => {
    fetch('/visualizer/api/entries')
      .then(r => r.json())
      .then(setEntries)
      .catch(() => {});
  }, []);

  return (
    <div style={{
      width: 280, background: '#0f172a', borderRight: '1px solid #1e293b',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      <div style={{
        padding: '14px 16px', borderBottom: '1px solid #1e293b',
        fontSize: 14, fontWeight: 700, color: '#e2e8f0',
      }}>
        Turbo Agent Logs
        <span style={{ color: '#64748b', fontWeight: 400, fontSize: 12, marginLeft: 8 }}>
          ({entries.length})
        </span>
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        {entries.map(entry => {
          const isSelected = entry.id === selectedId;
          // Parse timestamp from filename
          const tsMatch = entry.id.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})/);
          const dateStr = tsMatch
            ? `${tsMatch[1]}-${tsMatch[2]}-${tsMatch[3]} ${tsMatch[4]}:${tsMatch[5]}:${tsMatch[6]}`
            : entry.id;
          const shortId = entry.id.split('_').pop() || '';

          return (
            <div
              key={entry.id}
              onClick={() => onSelect(entry.id)}
              style={{
                padding: '10px 16px', cursor: 'pointer',
                background: isSelected ? '#1e293b' : 'transparent',
                borderLeft: isSelected ? '3px solid #3b82f6' : '3px solid transparent',
                borderBottom: '1px solid #0f172a',
              }}
            >
              <div style={{ color: isSelected ? '#e2e8f0' : '#94a3b8', fontSize: 12 }}>
                {dateStr}
              </div>
              <div style={{ color: '#64748b', fontSize: 10, marginTop: 2 }}>
                {shortId}
              </div>
            </div>
          );
        })}
        {entries.length === 0 && (
          <div style={{ padding: 16, color: '#64748b', fontSize: 12, textAlign: 'center' }}>
            No logs found in .turbo-agent/
          </div>
        )}
      </div>
    </div>
  );
}
