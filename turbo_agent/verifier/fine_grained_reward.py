"""
Fine-grained reward model.

Rather than collapsing a verifier's judgement into one discrete label (as in
LLM-as-a-Judge), LLM-as-a-Verifier reads the model's probability distribution
over an ordered set of score tokens and takes its expectation. The reward of a
candidate response tau given the conversation history t is

    R(t, tau) = (1 / C K) * sum_c sum_k sum_g  p_theta(v_g | t, c, tau) * phi(v_g)

  C  = number of evaluation criteria
  K  = number of repeated verifications
  G  = number of ordered score tokens (granularity; here G = 20, letters A-T)
  v_g = the g-th score token,  phi(v_g) = its scalar value
  p_theta = probability the verifier assigns to token v_g (from logprobs)

This module provides the granularity-20 scale, the async Gemini logprob client,
the score-token expectation `extract_score`, and the directed pairwise prompt.
A "directed" comparison places candidate `a` in slot A and `b` in slot B; the
ring pass of the Pivot Preference Tournament relies on this to cancel slot bias.
"""

import math
import os
import re

GRANULARITY = 20

SCALE = {
    "scale_description": (
        "Rate how likely the agent correctly solved the task on a "
        "20-point scale using letters A through T:\n"
        "  A = clearly and completely succeeded with verified output (best)\n"
        "  B-D = succeeded with only minor issues\n"
        "  E-G = above average, mostly correct with some issues\n"
        "  H-J = uncertain, leans toward success\n"
        "  K-M = uncertain, leans toward failure\n"
        "  N-P = below average, significant issues remain\n"
        "  Q-S = failed with some partial progress\n"
        "  T = clearly and completely failed (worst)"
    ),
    "score_format": "LETTER_A_TO_T",
    "valid_tokens": {
        **{chr(65 + i): float(GRANULARITY - i) for i in range(GRANULARITY)},
        **{chr(97 + i): float(GRANULARITY - i) for i in range(GRANULARITY)},
    },
}


# ---------------------------------------------------------------------------
# Async Gemini client (logprobs)
# ---------------------------------------------------------------------------

def create_gemini_client(api_key=None, provider=None):
    """Create a google-genai client. Vertex AI is preferred for Gemini 2.5
    logprobs; a plain Gemini API key also works."""
    from google import genai

    resolved = api_key or os.environ.get("VERTEX_API_KEY") \
        or os.environ.get("GEMINI_API_KEY", "")
    use_vertex = provider == "vertex_ai" or (
        provider is None and os.environ.get("VERTEX_API_KEY") and not api_key
    )
    if use_vertex:
        return genai.Client(vertexai=True, api_key=resolved)
    return genai.Client(api_key=resolved)


async def call_gemini(client, model_id, prompt, top_logprobs=20,
                      temperature=1.0, max_tokens=2048):
    """Call a Gemini model with logprobs. Returns (text, tokens,
    position_logprobs) where position_logprobs[i] is the list of
    (token, log_probability) alternatives at output position i."""
    from google.genai.types import (
        Content, GenerateContentConfig, Part, ThinkingConfig)

    config = GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        response_logprobs=True,
        logprobs=top_logprobs,
        thinking_config=ThinkingConfig(thinking_budget=0),
    )

    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[Content(role="user", parts=[Part(text=prompt)])],
        config=config,
    )

    text = response.text or ""
    tokens = None
    position_logprobs = None

    candidate = response.candidates[0]
    if candidate.logprobs_result and candidate.logprobs_result.top_candidates:
        position_logprobs = []
        for pos in candidate.logprobs_result.top_candidates:
            alts = [(lp.token, lp.log_probability) for lp in pos.candidates]
            position_logprobs.append(alts)
        if candidate.logprobs_result.chosen_candidates:
            tokens = [c.token
                      for c in candidate.logprobs_result.chosen_candidates]

    return text, tokens, position_logprobs


# ---------------------------------------------------------------------------
# Score-token expectation:  sum_g p(v_g) * phi(v_g), normalized to [0, 1]
# ---------------------------------------------------------------------------

def _find_tag_logprobs(tokens, position_logprobs, tag):
    if not tokens or not position_logprobs:
        return None
    text_so_far = ""
    for i, tok in enumerate(tokens):
        text_so_far += tok
        if text_so_far.rstrip().endswith(tag):
            if i + 1 < len(position_logprobs):
                return position_logprobs[i + 1]
    return None


def extract_score(text, tokens, position_logprobs, tag):
    """Expected score over the verifier's token distribution at `tag`,
    normalized to [0, 1]. Falls back to parsing the literal text token."""
    valid_tokens = SCALE["valid_tokens"]

    tag_lp = _find_tag_logprobs(tokens, position_logprobs, tag)
    probs = {}
    if tag_lp:
        for tok_str, logprob in tag_lp:
            tok = tok_str.strip()
            if tok in valid_tokens:
                val = valid_tokens[tok]
                p = math.exp(logprob)
                probs[val] = max(probs.get(val, 0.0), p)

    if probs:
        unique_vals = sorted(set(valid_tokens.values()))
        min_val, max_val = min(unique_vals), max(unique_vals)
        total_p = sum(probs.values())
        expected = sum(v * p for v, p in probs.items()) / total_p
        return (expected - min_val) / (max_val - min_val) \
            if max_val > min_val else 0.5

    tag_name = tag.strip("<>")
    pattern = rf"<{re.escape(tag_name)}>\s*(.+?)\s*</{re.escape(tag_name)}>"
    match = re.search(pattern, text or "", re.IGNORECASE)
    if match:
        tok = match.group(1).strip()
        raw_val = valid_tokens.get(tok)
        if raw_val is None:
            for vt, val in valid_tokens.items():
                if tok.lower() == vt.lower():
                    raw_val = val
                    break
        if raw_val is not None:
            unique_vals = sorted(set(valid_tokens.values()))
            min_val, max_val = min(unique_vals), max(unique_vals)
            return (raw_val - min_val) / (max_val - min_val) \
                if max_val > min_val else 0.5

    return 0.5


# ---------------------------------------------------------------------------
# Directed pairwise prompt (one evaluation criterion)
# ---------------------------------------------------------------------------

def build_prompt(history, action_a, action_b, criterion_name,
                 criterion_description, note):
    """One pairwise prompt focused on a single evaluation criterion. Candidate
    `a`'s response is shown as Trajectory A and `b`'s as Trajectory B."""
    note_block = f"{note}\n\n" if note else ""
    return (
        "You are an expert evaluator of AI coding agents. "
        "You will see a task (the conversation so far) and two candidate "
        "agent responses. "
        f"Your job is to evaluate them on ONE specific criterion: "
        f"**{criterion_name}**.\n\n"
        f"{note_block}"
        f"**Task / Conversation so far:**\n{history}\n\n"
        f"**Trajectory A:**\n{action_a}\n\n"
        f"**Trajectory B:**\n{action_b}\n\n"
        f"**Evaluation Guideline — {criterion_name}:**\n"
        f"{criterion_description}\n\n"
        f"Score each trajectory ONLY on this specific criterion. Ignore other "
        f"aspects of the trajectory that are not relevant to "
        f"\"{criterion_name}\".\n\n"
        f"**Rating Scale:**\n{SCALE['scale_description']}\n\n"
        "Then output your final scores:\n"
        f"<score_A>{SCALE['score_format']}</score_A>\n"
        f"<score_B>{SCALE['score_format']}</score_B>\n\n"
        "Begin your analysis now."
    )
