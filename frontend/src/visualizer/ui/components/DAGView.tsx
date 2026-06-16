import { useCallback, useMemo, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import type { RequestLogEntry } from '../types';
import { buildGraph, type NodeData } from '../utils/buildGraph';
import PipelineNode from './PipelineNode';
import NodeDetail from './NodeDetail';

const nodeTypes = { pipelineNode: PipelineNode };

const MINIMAP_COLORS: Record<string, string> = {
  request: '#3b82f6',
  context: '#06b6d4',
  response: '#22c55e',
  reflection: '#a855f7',
  verifier: '#f59e0b',
  progress: '#14b8a6',
  final: '#ec4899',
};

export default function DAGView({ entry }: { entry: RequestLogEntry }) {
  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => buildGraph(entry), [entry]);
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<{ id: string; data: NodeData } | null>(null);
  const [tabPerNode, setTabPerNode] = useState<Record<string, string>>({});

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode({ id: node.id, data: node.data as NodeData });
  }, []);

  return (
    <div style={{ flex: 1, position: 'relative' }}>
      {/* Stats bar */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, zIndex: 5,
        padding: '8px 16px', background: '#0f172aee',
        borderBottom: '1px solid #1e293b', display: 'flex', gap: 16, alignItems: 'center',
        fontSize: 12, color: '#94a3b8',
      }}>
        <span>
          <strong style={{ color: '#e2e8f0' }}>{entry.id}</strong>
        </span>
        <span>API: <strong style={{ color: '#3b82f6' }}>{entry.api}</strong></span>
        <span>Responses: <strong style={{ color: '#22c55e' }}>{entry.responses.length}</strong></span>
        <span>Best: <strong style={{ color: '#f59e0b' }}>#{entry.verifier.bestIndex} ({entry.verifier.bestModel})</strong></span>
        {entry.progressMonitor?.enabled && entry.progressMonitor.score != null && (
          <span>Progress: <strong style={{
            color: entry.progressMonitor.score > 0.7 ? '#22c55e' : entry.progressMonitor.score > 0.4 ? '#f59e0b' : '#ef4444',
          }}>{Math.round(entry.progressMonitor.score * 100)}%</strong></span>
        )}
        <span>Time: <strong style={{ color: '#e2e8f0' }}>{(entry.elapsedMs / 1000).toFixed(1)}s</strong></span>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        nodesDraggable={false}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        style={{ background: '#020617' }}
        defaultEdgeOptions={{
          style: { stroke: '#334155', strokeWidth: 2 },
          animated: true,
        }}
      >
        <Background color="#1e293b" gap={20} />
        <Controls
          style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
        />
        <MiniMap
          nodeColor={(node) => MINIMAP_COLORS[(node.data as NodeData)?.nodeType] || '#64748b'}
          style={{ background: '#0a0f1a', border: '1px solid #1e293b', borderRadius: 8 }}
          maskColor="#020617cc"
        />
      </ReactFlow>

      {selectedNode && (
        <NodeDetail
          data={selectedNode.data}
          tab={tabPerNode[selectedNode.id]}
          onTabChange={(tab) => setTabPerNode(prev => ({ ...prev, [selectedNode.id]: tab }))}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
}
