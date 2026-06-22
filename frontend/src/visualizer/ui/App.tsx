import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DAGView from './components/DAGView';
import type { RequestLogEntry } from './types';

// How often the visualizer re-polls the logs (the proxy writes them while it
// serves requests, and re-writes them when the background progress monitor
// finishes), so the UI stays live without a manual refresh.
const REFRESH_MS = 2000;

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [entry, setEntry] = useState<RequestLogEntry | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedId) {
      setEntry(null);
      return;
    }
    let cancelled = false;
    let first = true;
    let lastSerialized = '';

    const load = () => {
      if (first) setLoading(true);
      fetch(`/visualizer/api/entries/${selectedId}`)
        .then(r => r.json())
        .then(data => {
          if (cancelled) return;
          // Only swap in new data when it actually changed, so the DAG doesn't
          // re-layout / lose zoom+pan on every poll.
          const serialized = JSON.stringify(data);
          if (serialized !== lastSerialized) {
            lastSerialized = serialized;
            setEntry(data);
          }
          setLoading(false);
          first = false;
        })
        .catch(() => {
          if (cancelled) return;
          setLoading(false);
          first = false;
        });
    };

    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
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
