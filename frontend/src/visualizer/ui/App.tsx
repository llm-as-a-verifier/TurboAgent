import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DAGView from './components/DAGView';
import type { RequestLogEntry } from './types';

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [entry, setEntry] = useState<RequestLogEntry | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedId) {
      setEntry(null);
      return;
    }
    setLoading(true);
    fetch(`/visualizer/api/entries/${selectedId}`)
      .then(r => r.json())
      .then(data => {
        setEntry(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [selectedId]);

  return (
    <div style={{ display: 'flex', width: '100%', height: '100%' }}>
      <Sidebar selectedId={selectedId} onSelect={setSelectedId} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {loading && (
          <div style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#64748b', fontSize: 14,
          }}>
            Loading...
          </div>
        )}
        {!loading && !entry && (
          <div style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#64748b', fontSize: 14, flexDirection: 'column', gap: 8,
          }}>
            <div style={{ fontSize: 32 }}>Turbo Agent Visualizer</div>
            <div>Select a log entry from the sidebar</div>
          </div>
        )}
        {!loading && entry && <DAGView entry={entry} />}
      </div>
    </div>
  );
}
