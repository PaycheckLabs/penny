# Penny Anchor

Last updated: 2026-02-24  
Repo: https://github.com/PaycheckLabs/penny  
Default branch: main

## 1) What Penny is
Penny is the official AI advisor for Paycheck Labs: product guidance, community operations, and education. She is calm, precise, and protective.

Canonical role + boundaries live here:
- docs/CORE_CONTRACT.md

## 2) Non-negotiables
1. Penny stays general-purpose by default.
2. Penny does not mention Paycheck Labs / Checks unless the user explicitly asks.
3. In groups, Penny only responds when invoked (command, @mention, or reply-to-bot).
4. Reliability over cleverness: avoid disruptive refactors in working paths.

## 3) Current capabilities (shipped)
Telegram bot:
- DMs: responds normally
- Groups: responds only when invoked
Commands:
- /start
- /help
- /penny <question>
- /price <SYMBOL> [CONVERT]  (example: /price BTC or /price BTC USD)

Crypto price:
- CoinMarketCap free API via src/price_cmc.py (CMC_API_KEY required)

Optional community gate:
- Welcome/verification gate exists in src/welcome_gate.py and is toggled via env vars

## 4) Architecture map
src/bot.py
- Telegram routing (DM vs group rules)
- Small-talk fast-path (avoids calling OpenAI for greetings)
- Calls OpenAI via src/answer.py using asyncio.to_thread to avoid blocking
- /price is isolated and does not affect other flows

src/answer.py
- Builds model messages + handles context injection
- Context injection only triggers on explicit Paycheck/Checks queries
- Uses local markdown retrieval (keyword scoring) over synced docs
- Uses OpenAI Responses API (client.responses.create)

src/knowledge_base.py
- Small static reference notes
- Only injected when a Paycheck/Checks query is detected

src/checks_rag.py
- Additional keyword-based retrieval module (stable, no embeddings)
- Candidate dirs include src/data/checks_whitepaper and docs/ trees

src/price_cmc.py
- CMC quote fetch + small in-memory cache
- Formats a single-line response for chat

scripts/sync_whitepaper.py + .github/workflows/sync-checks-whitepaper.yml
- Syncs md/txt content from PaycheckLabs/checks-gitbook
- Destination: src/data/checks_whitepaper
- Runs on workflow_dispatch + weekly schedule

## 5) Environment variables
Required:
- TELEGRAM_BOT_TOKEN
- OPENAI_API_KEY

Common optional:
- OPENAI_MODEL
- OPENAI_MAX_OUTPUT_TOKENS
- PENNY_SYSTEM_PROMPT
- PENNY_MAX_TURNS
- CMC_API_KEY
- CMC_CACHE_TTL_SECONDS

Welcome gate (optional):
- WELCOME_GATE_ENABLED
- ADMIN_USER_IDS
- ALLOWED_USER_IDS
- VERIFY_TIMEOUT_SECONDS
- KICK_IF_NOT_VERIFIED
- CAPTCHA_MAX_ATTEMPTS
- KICK_ON_TOO_MANY_WRONG
- WELCOME_GIF_FILE_ID / WELCOME_GIF_LOCAL_PATH

Whitepaper sync (GitHub Actions):
- CHECKS_GITBOOK_TOKEN (repo secret)

## 6) “Roadmap” (replaces docs/ROADMAP.md)
Near-term priorities:
P1) Natural-language price intent routing
- Detect messages like “what is the price of BTC” and route to the existing CMC logic
- Keep this additive (no refactor of core flow)

P2) News (command-first)
- Add /news <topic> first (stable)
- Later: detect “any news on BTC?” and route to /news

P3) Digest
- /digest: top headlines + BTC/ETH price lines

## 7) Handoffs
Engineering handoffs live outside docs/ so they are not accidentally used as RAG context:
- handoffs/YYYY-MM-DD.md

Latest: handoffs/2026-02-24.md
