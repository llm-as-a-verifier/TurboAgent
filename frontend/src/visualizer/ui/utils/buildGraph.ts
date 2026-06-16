import type { Node, Edge } from '@xyflow/react';
import type { RequestLogEntry } from '../types';
import dagre from '@dagrejs/dagre';

export type NodeData = {
  label: string;
  nodeType: 'request' | 'context' | 'response' | 'reflection' | 'verifier' | 'progress' | 'final';
  detail: unknown;
  score?: number;
  isBest?: boolean;
  model?: string;
  responseIndex?: number;
};

const NODE_WIDTH = 220;
const NODE_HEIGHT = 80;

export function buildGraph(entry: RequestLogEntry): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // 1. Request node
  const userMessages = entry.request.messages.filter(m => m.role === 'user');
  const lastUserMsg = userMessages[userMessages.length - 1];
  let requestPreview = '';
  if (lastUserMsg) {
    const content = lastUserMsg.content;
    if (typeof content === 'string') {
      requestPreview = content.slice(0, 120);
    } else if (Array.isArray(content)) {
      const textBlock = (content as Array<{ type: string; text?: string }>).filter(b => b.type === 'text' && b.text && !b.text.startsWith('<system-reminder>')).pop();
      requestPreview = textBlock?.text?.slice(0, 120) || '';
    }
  }

  nodes.push({
    id: 'request',
    type: 'pipelineNode',
    position: { x: 0, y: 0 },
    data: {
      label: `Request (${entry.api})`,
      nodeType: 'request',
      detail: entry.request,
      model: entry.request.model,
    } satisfies NodeData,
  });

  // 2. Context refinement node (if enabled)
  let prevNodeId = 'request';
  if (entry.contextRefinement.enabled) {
    nodes.push({
      id: 'context',
      type: 'pipelineNode',
      position: { x: 0, y: 0 },
      data: {
        label: 'Context Refinement',
        nodeType: 'context',
        detail: entry.contextRefinement,
      } satisfies NodeData,
    });
    edges.push({ id: 'e-req-ctx', source: 'request', target: 'context', type: 'smoothstep' });
    prevNodeId = 'context';
  }

  // 3. Response nodes (fan out)
  const responseIds: string[] = [];
  for (let i = 0; i < entry.responses.length; i++) {
    const resp = entry.responses[i];
    const id = `response-${i}`;
    responseIds.push(id);
    const msg = resp.response.choices?.[0]?.message;
    const contentPreview = msg?.content?.slice(0, 100) || '';
    const hasToolCalls = msg?.tool_calls && (msg.tool_calls as unknown[]).length > 0;

    nodes.push({
      id,
      type: 'pipelineNode',
      position: { x: 0, y: 0 },
      data: {
        label: `Response ${i}`,
        nodeType: 'response',
        detail: resp,
        model: resp.model,
        responseIndex: i,
        score: entry.verifier.scores?.find(s => s.index === i)?.score,
        isBest: entry.verifier.bestIndex === i,
      } satisfies NodeData,
    });
    edges.push({
      id: `e-${prevNodeId}-${id}`,
      source: prevNodeId,
      target: id,
      type: 'smoothstep',
    });
  }

  // 4. Reflection nodes (if enabled, one per response)
  let reflectionIds: string[] = [];
  if (entry.reflection.enabled && entry.reflection.actions) {
    for (let i = 0; i < entry.reflection.actions.length; i++) {
      const action = entry.reflection.actions[i];
      const id = `reflection-${i}`;
      reflectionIds.push(id);
      nodes.push({
        id,
        type: 'pipelineNode',
        position: { x: 0, y: 0 },
        data: {
          label: `Reflection ${i}`,
          nodeType: 'reflection',
          detail: action,
          responseIndex: i,
        } satisfies NodeData,
      });
      if (i < responseIds.length) {
        edges.push({
          id: `e-${responseIds[i]}-${id}`,
          source: responseIds[i],
          target: id,
          type: 'smoothstep',
        });
      }
    }
  }

  // 5. Verifier node
  const sourceIds = reflectionIds.length > 0 ? reflectionIds : responseIds;
  let preFinalNodeId = '';
  if (entry.verifier.enabled) {
    nodes.push({
      id: 'verifier',
      type: 'pipelineNode',
      position: { x: 0, y: 0 },
      data: {
        label: `Verifier (${entry.config.verifier?.method?.name || entry.config.verifier?.scoring?.method || 'pivot_tournament'})`,
        nodeType: 'verifier',
        detail: entry.verifier,
      } satisfies NodeData,
    });
    for (const srcId of sourceIds) {
      edges.push({
        id: `e-${srcId}-verifier`,
        source: srcId,
        target: 'verifier',
        type: 'smoothstep',
      });
    }
    preFinalNodeId = 'verifier';
  } else {
    // No verifier: connect best response directly
    preFinalNodeId = sourceIds[entry.verifier.bestIndex ?? 0] || sourceIds[0] || 'request';
  }

  // 5b. Progress monitor node (if enabled)
  if (entry.progressMonitor?.enabled) {
    nodes.push({
      id: 'progress',
      type: 'pipelineNode',
      position: { x: 0, y: 0 },
      data: {
        label: 'Progress Monitor',
        nodeType: 'progress',
        detail: entry.progressMonitor,
        score: entry.progressMonitor.score,
      } satisfies NodeData,
    });
    edges.push({
      id: `e-${preFinalNodeId}-progress`,
      source: preFinalNodeId,
      target: 'progress',
      type: 'smoothstep',
    });
    edges.push({ id: 'e-progress-final', source: 'progress', target: 'final', type: 'smoothstep' });
  } else {
    edges.push({ id: `e-${preFinalNodeId}-final`, source: preFinalNodeId, target: 'final', type: 'smoothstep' });
  }

  // 6. Final response node
  const finalMsg = entry.finalResponse.choices?.[0]?.message;
  nodes.push({
    id: 'final',
    type: 'pipelineNode',
    position: { x: 0, y: 0 },
    data: {
      label: 'Final Response',
      nodeType: 'final',
      detail: entry.finalResponse,
      model: entry.finalResponse.model,
    } satisfies NodeData,
  });

  // Layout with dagre
  return layoutGraph(nodes, edges);
}

function layoutGraph(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', nodesep: 40, ranksep: 80 });

  for (const node of nodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  const laidOut = nodes.map(node => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: {
        x: pos.x - NODE_WIDTH / 2,
        y: pos.y - NODE_HEIGHT / 2,
      },
    };
  });

  return { nodes: laidOut, edges };
}
