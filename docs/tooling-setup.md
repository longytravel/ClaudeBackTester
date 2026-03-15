# Tooling Setup Guide — Context & Documentation System

Everything needed to keep Claude productive across sessions.

## 1. Claude-Mem (Automatic Session Memory)

**What it does:** Automatically captures everything Claude does during sessions, compresses it with AI, and injects relevant context into future sessions. No manual work needed — it hooks into the session lifecycle.

**Install (in Claude Code):**
```
/plugin marketplace add thedotmack/claude-mem
/plugin install claude-mem
```

Then **restart Claude Code**.

**Important:** Do NOT use `npm install -g claude-mem` — that only installs the SDK, not the plugin hooks.

**Verify it's working:** After restart, you should see context from previous sessions automatically appearing at the start of new sessions.

**Configuration:** Settings auto-created at `~/.claude-mem/settings.json` on first run. Defaults are fine.

---

## 2. Claude-Mermaid MCP (Architecture Diagrams)

**What it does:** Lets Claude generate and preview Mermaid diagrams (architecture, flowcharts, sequence diagrams) with live reload in your browser.

**Install (in Claude Code):**
```
/plugin marketplace add veelenga/claude-mermaid
/plugin install claude-mermaid@claude-mermaid
```

Then **restart Claude Code**.

**Verify:**
```
/mcp
```
You should see `mermaid` in the MCP server list.

**How it works:** Claude generates a Mermaid diagram → opens in browser at `http://localhost:3737/` → auto-refreshes when diagram is updated. Export to SVG/PNG/PDF.

**Prerequisite:** Node.js and npm must be installed and in PATH.

---

## 3. GitDiagram (Instant Repo Visualization)

**What it does:** Converts any GitHub repo into an interactive architecture diagram. Zero install — it's a web tool.

**How to use:** Replace `github.com` with `gitdiagram.com` in the repo URL:
```
https://gitdiagram.com/YOUR_USERNAME/ClaudeBackTester
```

**Note:** Repo must be public, or you need to auth with GitHub.

---

## 4. CLAUDE.md Structure (Lean Rules File)

**Target:** Under 200 lines. Rules and conventions only — NO architecture docs, NO history.

**What belongs in CLAUDE.md:**
- Build commands, test commands
- Code conventions (naming, imports, patterns)
- Critical gotchas (things that will break if ignored)
- Pointers to docs: "See docs/architecture.md for system overview"
- Session continuity protocol

**What does NOT belong:**
- Architecture descriptions → `docs/architecture.md`
- Historical learnings → `.claude/memory/`
- Pipeline documentation → `docs/research-pipeline.md`
- Bug post-mortems → `.claude/memory/`

---

## 5. Docs Structure (Architecture & System Docs)

Claude reads these on demand when referenced from CLAUDE.md.

```
docs/
  architecture.md          — Full system overview, component diagram, data flow
  research-pipeline.md     — Article ingestion → strategy build → test pipeline
  tooling-setup.md         — This file (how to set up tooling)
```

**Key principle:** Put reference material in `docs/`, not in CLAUDE.md. Claude loads CLAUDE.md every session but only reads docs/ when needed.

---

## 6. CURRENT_TASK.md (Session Pickup)

**Updated at every session end.** Contains:
- Exact next steps (numbered, specific)
- Blockers or decisions needed
- What was just completed
- Any in-progress work

**Template:**
```markdown
# Current Task

## Last Completed
- [What was done in the last session]

## Next Steps
1. [Exact next action — specific enough to start immediately]
2. [Second action]
3. [Third action]

## Blockers
- [Any decisions or issues blocking progress]

## Context
- [Any important state that isn't captured elsewhere]
```

---

## Installation Checklist

Run these in order in Claude Code:

1. **Claude-Mem:**
   ```
   /plugin marketplace add thedotmack/claude-mem
   /plugin install claude-mem
   ```

2. **Claude-Mermaid:**
   ```
   /plugin marketplace add veelenga/claude-mermaid
   /plugin install claude-mermaid@claude-mermaid
   ```

3. **Restart Claude Code**

4. **Verify both installed:**
   ```
   /mcp
   ```
   Should show `mermaid` in the list. Claude-mem hooks are invisible but active.

5. **Generate system architecture diagram:**
   Ask Claude to create the architecture overview with Mermaid diagrams.

6. **Slim CLAUDE.md to <200 lines:**
   Move architecture and history content to docs/ and memory files.
