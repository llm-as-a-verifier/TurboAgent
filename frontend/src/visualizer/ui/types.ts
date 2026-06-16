export interface EntryListItem {
  id: string;
  filename: string;
}

export interface RequestLogEntry {
  id: string;
  timestamp: string;
  api: string;
  config: {
    backend: {
      models: Array<{
        name: string;
        num_candidates: number;
        temperature: number;
      }>;
    };
    verifier: {
      model: { name: string };
      method?: { name: string; pivots?: number; n_verifications?: number };
      scoring?: { method: string };
      pairwise?: { template: string; tau: number };
    };
  };
  request: {
    model: string;
    messages: Array<{ role: string; content: unknown }>;
    system?: unknown;
    tools?: unknown[];
    max_tokens?: number;
    [key: string]: unknown;
  };
  contextRefinement: {
    enabled: boolean;
    originalMessages?: unknown[];
    refinedMessages?: unknown[];
  };
  responses: Array<{
    model: string;
    response: {
      choices: Array<{
        message: { role: string; content: string | null; tool_calls?: unknown[] };
        finish_reason: string;
      }>;
      usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
      model: string;
      id: string;
    };
  }>;
  reflection: {
    enabled: boolean;
    actions?: Array<{
      original: string;
      reflected: string;
    }>;
  };
  verifier: {
    enabled: boolean;
    scores?: Array<{
      index: number;
      model: string;
      score: number;
      details: { score: number; criterionScores: unknown[] };
    }>;
    comparisons?: Array<{
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
    }>;
    bestIndex?: number;
    bestModel?: string;
    bestScore?: number;
  };
  progressMonitor: {
    enabled: boolean;
    score?: number;
    details?: {
      score: number;
      rawProbs: Record<string, number>;
      generatedText: string;
    };
    error?: string;
  };
  finalResponse: {
    choices: Array<{
      message: { role: string; content: string | null; tool_calls?: unknown[] };
    }>;
    usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
    model: string;
    [key: string]: unknown;
  };
  elapsedMs: number;
}
