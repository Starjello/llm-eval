#!/usr/bin/env python3
"""
Env-aware Gemini diagnostic:
- Reads GOOGLE_API_KEY from env
- Tests both gemini-2.5-pro and gemini-2.5-flash
- Tries several max_output_tokens to surface MAX_TOKENS behavior
- Robust text extraction (candidates[].content.parts[].text)
"""

import os, sys
MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]
PROMPT = "Write a minimal Python function `square(n)` that returns n*n."
LADDER = [120, 90, 60, 40, 28]

def extract_text(resp):
    t = getattr(resp, "text", None) or ""
    if t.strip():
        return t.strip()
    out = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            txt = getattr(part, "text", "")
            if txt:
                out.append(txt)
    return "".join(out).strip()

def main():
    api = os.getenv("GOOGLE_API_KEY")
    if not api:
        print("❌ GOOGLE_API_KEY not found in your environment.")
        sys.exit(2)

    try:
        from google import genai
        from google.genai import types
        sdk_ok = True
    except Exception as e:
        print("❌ google-genai not installed or import failed:", e)
        print("   Try: pip install -U google-genai")
        sys.exit(2)

    client = genai.Client(api_key=api)
    print("✅ Using GOOGLE_API_KEY from environment")
    try:
        import google.genai as gg
        print("SDK:", gg.__version__)
    except Exception:
        pass

    sys_hint = "Think silently. Output ONLY Python code. No markdown, no comments."
    for model in MODELS:
        print(f"\n=== Model: {model} ===")
        for mot in LADDER:
            print(f" - max_output_tokens={mot}")
            try:
                cfg = types.GenerateContentConfig(
                    system_instruction=sys_hint,
                    temperature=0.2,
                    max_output_tokens=mot,
                    candidate_count=1,
                    top_k=1,
                    stop_sequences=["```", "\n\n\n"],
                )
                resp = client.models.generate_content(
                    model=model,
                    contents=[PROMPT],
                    config=cfg,
                )
            except Exception as e:
                print("   ❌ API error:", e)
                continue

            txt = extract_text(resp)
            frs = [getattr(c, "finish_reason", None) for c in (getattr(resp, "candidates", []) or [])]
            usage = getattr(resp, "usage_metadata", None)
            print(f"   finish_reasons={frs} | usage={usage} | text_len={len(txt)}")
            if len(txt) and txt.startswith("```"):
                # just in case
                txt = "\n".join(ln for ln in txt.splitlines() if not ln.strip().startswith("```")).strip()

            # Show a tiny preview so you can see it's code
            print("   preview:", (txt[:80] + ("…" if len(txt) > 80 else "")) or "<EMPTY>")

    print("\nDone. If all previews show code, your key + SDK are good.")
    print("If you see MAX_TOKENS or <EMPTY>, use the tighter output ladder in your evaluator.")
if __name__ == "__main__":
    main()
