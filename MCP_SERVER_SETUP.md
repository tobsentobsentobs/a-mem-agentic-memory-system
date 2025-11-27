# MCP Server Setup für A-MEM

## 📋 Übersicht

Der A-MEM MCP Server stellt Tools für das Agentic Memory System bereit.

## 🛠️ Verfügbare Tools

### 1. `create_atomic_note`
Speichert eine neue Information im Memory System.

**Parameter:**
- `content` (string, required): Der Text der Notiz/Erinnerung
- `source` (string, optional): Quelle der Information (Standard: "user_input")

**Beispiel:**
```json
{
  "content": "Python async/await ermöglicht nicht-blockierende I/O Operationen",
  "source": "user_input"
}
```

### 2. `retrieve_memories`
Sucht nach relevanten Erinnerungen basierend auf semantischer Ähnlichkeit.

**Parameter:**
- `query` (string, required): Die Suchanfrage
- `max_results` (integer, optional): Maximale Anzahl Ergebnisse (Standard: 5, Max: 20)

**Beispiel:**
```json
{
  "query": "Python async programming",
  "max_results": 5
}
```

### 3. `get_memory_stats`
Gibt Statistiken über das Memory System zurück.

**Parameter:** Keine

**Beispiel:**
```json
{}
```

### 4. `delete_atomic_note`
Löscht eine Note aus dem Memory System. Entfernt die Note aus Graph und Vector Store sowie alle zugehörigen Verbindungen.

**Parameter:**
- `note_id` (string, required): Die UUID der Note, die gelöscht werden soll

**Beispiel:**
```json
{
  "note_id": "732c8c3b-7c71-42a6-9534-a611b4ffe7bf"
}
```

### 5. `add_file`
Speichert den Inhalt einer Datei (z.B. .md) als Note im Memory System. Unterstützt automatisches Chunking für große Dateien (>16KB).

**Parameter:**
- `file_path` (string, optional): Pfad zur Datei, die gespeichert werden soll (relativ oder absolut)
- `file_content` (string, optional): Alternativ: Direkter Dateiinhalt als String (wenn file_path nicht angegeben)
- `chunk_size` (integer, optional): Maximale Größe pro Chunk in Bytes (Standard: 15000, Max: 16384)

**Hinweis:** Entweder `file_path` ODER `file_content` muss angegeben werden.

**Beispiel:**
```json
{
  "file_path": "documentation.md",
  "chunk_size": 15000
}
```

Oder mit direktem Inhalt:
```json
{
  "file_content": "# Dokumentation\n\nDies ist der Inhalt...",
  "chunk_size": 15000
}
```

### 6. `reset_memory`
Setzt das komplette Memory System zurück (Graph + Vector Store). Löscht alle Notes, Edges und Embeddings.

**⚠️ ACHTUNG:** Diese Aktion kann nicht rückgängig gemacht werden!

**Parameter:** Keine

**Beispiel:**
```json
{}
```

## 🚀 Installation & Start

### 1. Dependencies installieren

```bash
pip install mcp
```

### 2. MCP Server starten

```bash
python mcp_server.py
```

Oder direkt:

```bash
python -m src.a_mem.main
```

## 📝 Cursor/IDE Konfiguration

### Cursor IDE

Füge folgende Konfiguration zu deiner MCP-Konfigurationsdatei hinzu:
- Windows: `%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
- macOS: `~/Library/Application Support/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- Linux: `~/.config/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": ["-m", "src.a_mem.main"],
      "cwd": "/path/to/a-mem-agentic-memory-system"
    }
  }
}
```

**Wichtig:** Passe `cwd` auf den absoluten Pfad zu deinem Projekt-Verzeichnis an!

### Visual Studio Code

Falls eine MCP Extension verfügbar ist, nutze die VSCode Settings (JSON) oder eine `mcp.json` Datei im Projekt-Root:

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": ["-m", "src.a_mem.main"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

## 🔧 Konfiguration

Der Server nutzt die Konfiguration aus `src/a_mem/config.py` und `.env` Datei.

### Environment Variables (.env)

Kopiere `.env.example` zu `.env` und passe die Werte an:

**Ollama (Standard):**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
```

**OpenRouter:**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

### Ollama Setup (bei LLM_PROVIDER=ollama)

Stelle sicher, dass Ollama läuft und beide Modelle installiert sind:
```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text:latest
```

## 📊 Beispiel-Nutzung

### Memory erstellen:
```python
# Via MCP Tool
create_atomic_note(
    content="Python async/await Patterns",
    source="user_input"
)
```

### Memory suchen:
```python
# Via MCP Tool
retrieve_memories(
    query="Python concurrency",
    max_results=5
)
```

### Statistiken abrufen:
```python
# Via MCP Tool
get_memory_stats()
```

### Note löschen:
```python
# Via MCP Tool
delete_atomic_note(
    note_id="732c8c3b-7c71-42a6-9534-a611b4ffe7bf"
)
```

### Datei importieren:
```python
# Via MCP Tool - Automatisches Chunking bei großen Dateien
add_file(
    file_path="documentation.md",
    chunk_size=15000
)
```

### Memory System zurücksetzen:
```python
# Via MCP Tool - ⚠️ LÖSCHT ALLES!
reset_memory()
```

## ✅ Status

Der MCP Server ist vollständig implementiert und nutzt:
- ✅ Lokales Ollama (qwen3:4b für LLM, nomic-embed-text für Embeddings)
- ✅ Async I/O für Performance
- ✅ Memory Evolution im Hintergrund
- ✅ Graph-basierte Verknüpfungen



