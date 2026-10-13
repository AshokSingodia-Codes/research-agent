# Changelog

All notable changes to the `research-agent` platform will be documented in this file.

## [Phase 1] - Typed Specialists & LLM Provider Abstraction
- Built `app/llm/` provider abstraction (`LLMProvider` ABC, `AnthropicProvider`, `GroqProvider`, `FakeProvider`).
- Integrated per-provider pricing tables and accurate token/USD cost accounting.
- Implemented Pydantic v2 schemas for all hand-offs (`ResearchPlan`, `SubQuestion`, `ResearchFindings`, `Finding`, `Report`, `ReviewDecision`, `Contradiction`).
- Implemented `Planner`, `Researcher`, and `Writer` specialists with retry-with-repair (max 2 repairs).
- Verified with comprehensive test suite including rule-based quality checks across 5 technical questions.
