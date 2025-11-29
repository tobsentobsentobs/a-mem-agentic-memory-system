# A-MEM: Vollständige Architektur-Darstellung

## System-Architektur (Mermaid Diagram)

```mermaid
graph TB
    subgraph "MCP Server Layer"
        MCP[main.py<br/>MCP Server<br/>15 Tools]
        HTTP[HTTP Server<br/>Optional<br/>Port 42424]
    end

    subgraph "Core Logic Layer"
        MC[MemoryController<br/>core/logic.py]
        MC -->|create_note| CN[Note Creation]
        MC -->|retrieve| RET[Memory Retrieval]
        MC -->|evolve| EVO[Memory Evolution]
        MC -->|link| LINK[Automatic Linking]
    end

    subgraph "Storage Layer - Dual Architecture"
        SM[StorageManager<br/>storage/engine.py]
        SM -->|Vector Search| VS[VectorStore<br/>ChromaDB<br/>Embeddings]
        SM -->|Graph Relations| GS[GraphStore<br/>NetworkX DiGraph<br/>Typed Edges]
    end

    subgraph "LLM Service Layer"
        LLM[LLMService<br/>utils/llm.py]
        LLM -->|Metadata Extraction| META[Type Classification<br/>Keywords, Tags, Summary]
        LLM -->|Embeddings| EMB[Text Embeddings<br/>nomic-embed-text]
        LLM -->|Text Generation| GEN[Text Generation<br/>qwen3:4b / GPT-4o-mini]
    end

    subgraph "Advanced Features"
        PRIO[Priority Scoring<br/>utils/priority.py<br/>Type, Age, Usage, Edges]
        ENZ[Memory Enzymes<br/>utils/enzymes.py]
        ENZ -->|Maintenance| LP[Link Pruner<br/>Old/Weak Links]
        ENZ -->|Discovery| RS[Relation Suggester<br/>New Connections]
        ENZ -->|Compression| SD[Summary Digester<br/>Overcrowded Nodes]
        ENZ -->|Cleanup| ZNR[Zombie Node Remover<br/>Empty Nodes]
        ENZ -->|Validation| NV[Note Validator]
        ENZ -->|Merging| DM[Duplicate Merger]
        ENZ -->|Linking| INL[Isolated Node Linker]
    end

    subgraph "Researcher Agent"
        RA[ResearcherAgent<br/>utils/researcher.py]
        RA -->|Web Search| WS[Web Search Module]
        RA -->|Content Extract| CE[Content Extraction]
        RA -->|Note Creation| NC[Research Note Creation]
        WS -->|Primary| GSAPI[Google Search API]
        WS -->|Fallback| DDG[DuckDuckGo HTTP]
        CE -->|Primary| JINA[Jina Reader<br/>Local Docker / Cloud]
        CE -->|PDF| UNSTR[Unstructured<br/>Library / API]
        CE -->|Fallback| READ[Readability]
    end

    subgraph "Background Processes"
        SCHED[Enzyme Scheduler<br/>Hourly Maintenance]
        AUTO[Auto-Save Task<br/>5min Interval]
        SCHED -->|Trigger| ENZ
        AUTO -->|Save| SM
    end

    subgraph "Event & Logging System"
        EVT[Event Logger<br/>utils/priority.py]
        EVT -->|Append-Only| JSONL[events.jsonl<br/>NOTE_CREATED<br/>RELATION_CREATED<br/>MEMORY_EVOLVED<br/>ENZYME_RUN]
    end

    subgraph "External Providers"
        OLLAMA[Ollama<br/>Local Models<br/>localhost:11434]
        OR[OpenRouter<br/>Cloud LLMs<br/>API Gateway]
        JINA_EXT[Jina Reader<br/>Docker / Cloud]
        UNSTR_EXT[Unstructured<br/>PDF Extraction]
        GOOGLE[Google Search API<br/>Custom Search]
    end

    subgraph "Data Models"
        AN[AtomicNote<br/>models/note.py<br/>Content, Summary, Keywords, Tags, Type, Metadata]
        NR[NoteRelation<br/>models/note.py<br/>Type, Reasoning, Weight]
        NI[NoteInput<br/>models/note.py<br/>Input Data Structure]
        SR[SearchResult<br/>models/note.py<br/>Note + Score + Priority]
    end

    subgraph "File System"
        CHROMA[data/chroma/<br/>Vector Database]
        GRAPH[data/graph/<br/>knowledge_graph.json<br/>NetworkX Format]
        EVENTS[data/events.jsonl<br/>Event Log]
        LOCK[data/graph/graph.lock<br/>Cross-Platform Locking]
    end

    %% MCP Server Connections
    MCP -->|15 Tools| MC
    HTTP -->|Graph Export| SM

    %% Core Logic Connections
    CN -->|Metadata| LLM
    CN -->|Store| SM
    CN -->|Trigger| EVO
    RET -->|Vector Search| VS
    RET -->|Graph Traversal| GS
    RET -->|Low Confidence| RA
    RET -->|Priority| PRIO
    EVO -->|Find Similar| VS
    EVO -->|Create Links| GS
    LINK -->|Semantic Similarity| VS
    LINK -->|Add Edges| GS

    %% Storage Connections
    VS -->|Persist| CHROMA
    GS -->|Persist| GRAPH
    GS -->|Lock| LOCK

    %% LLM Service Connections
    LLM -->|Ollama| OLLAMA
    LLM -->|OpenRouter| OR

    %% Researcher Connections
    RA -->|Search| WS
    RA -->|Extract| CE
    RA -->|Create| NC
    NC -->|Store| MC
    GSAPI -->|API| GOOGLE
    JINA -->|Local| JINA_EXT
    JINA -->|Cloud| JINA_EXT
    UNSTR -->|Library| UNSTR_EXT
    UNSTR -->|API| UNSTR_EXT

    %% Background Processes
    MC -->|Start| SCHED
    MC -->|Start| AUTO

    %% Event Logging
    CN -->|Log| EVT
    EVO -->|Log| EVT
    ENZ -->|Log| EVT
    RA -->|Log| EVT
    EVT -->|Write| EVENTS

    %% Data Models Usage
    CN -->|Create| AN
    RET -->|Return| SR
    EVO -->|Create| NR
    AN -->|Store| SM
    NR -->|Store| GS

    %% Styling
    classDef mcpLayer fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000000
    classDef coreLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000000
    classDef storageLayer fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000000
    classDef llmLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000000
    classDef featureLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000000
    classDef externalLayer fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#000000
    classDef dataLayer fill:#e0f2f1,stroke:#004d40,stroke-width:2px,color:#000000

    class MCP,HTTP mcpLayer
    class MC,CN,RET,EVO,LINK coreLayer
    class SM,VS,GS,CHROMA,GRAPH,LOCK storageLayer
    class LLM,META,EMB,GEN llmLayer
    class PRIO,ENZ,LP,RS,SD,ZNR,NV,DM,INL featureLayer
    class OLLAMA,OR,JINA_EXT,UNSTR_EXT,GOOGLE externalLayer
    class AN,NR,NI,SR dataLayer
```

## Workflow-Diagramm: Note Creation & Retrieval

```mermaid
sequenceDiagram
    participant User
    participant MCP as MCP Server
    participant MC as MemoryController
    participant LLM as LLMService
    participant VS as VectorStore
    participant GS as GraphStore
    participant ENZ as Enzymes
    participant EVT as Event Logger

    Note over User,EVT: Note Creation Workflow
    User->>MCP: create_atomic_note(content)
    MCP->>MC: create_note(NoteInput)
    MC->>LLM: extract_metadata(content)
    LLM-->>MC: {type, keywords, tags, summary}
    MC->>LLM: get_embedding(text)
    LLM-->>MC: embedding vector
    MC->>VS: add(note, embedding)
    MC->>GS: add_node(note)
    MC->>GS: save_snapshot()
    MC->>EVT: log_event(NOTE_CREATED)
    MC->>MC: _evolve_memory() [async]
    MC-->>MCP: note_id
    MCP-->>User: note_id

    Note over MC,EVT: Background Evolution
    MC->>VS: query(embedding, k=5)
    VS-->>MC: similar_notes[]
    MC->>LLM: should_link?(note, similar)
    LLM-->>MC: {should_link, reasoning, weight}
    alt Should Link
        MC->>GS: add_edge(note1, note2, relation)
        MC->>EVT: log_event(RELATION_CREATED)
    end
    MC->>LLM: should_evolve?(existing_note, new_note)
    LLM-->>MC: {should_evolve, updated_content}
    alt Should Evolve
        MC->>GS: update_node(existing_note)
        MC->>VS: update(existing_note, embedding)
        MC->>EVT: log_event(MEMORY_EVOLVED)
    end

    Note over User,EVT: Memory Retrieval Workflow
    User->>MCP: retrieve_memories(query, max_results)
    MCP->>MC: retrieve(query, max_results)
    MC->>LLM: get_embedding(query)
    LLM-->>MC: query_embedding
    MC->>VS: query(query_embedding, k=max_results*2)
    VS-->>MC: candidates[]
    MC->>GS: get_connected_notes(candidates)
    GS-->>MC: linked_contexts[]
    MC->>PRIO: compute_priority(note)
    PRIO-->>MC: priority_score
    MC->>MC: rank_by(similarity + priority)
    alt Low Confidence
        MC->>RA: research(query) [async]
        RA->>RA: Web Search + Extract
        RA->>MC: create_note(research_notes)
    end
    MC-->>MCP: SearchResult[]
    MCP-->>User: ranked_results
```

## Memory Enzymes Workflow

```mermaid
graph LR
    subgraph "Enzyme Scheduler (Hourly)"
        SCHED[Enzyme Scheduler<br/>1h Interval]
    end

    subgraph "Enzyme Execution"
        ENZ[run_memory_enzymes]
        ENZ --> LP[Link Pruner]
        ENZ --> ZNR[Zombie Node Remover]
        ENZ --> RS[Relation Suggester]
        ENZ --> SD[Summary Digester]
        ENZ --> NV[Note Validator]
        ENZ --> DM[Duplicate Merger]
        ENZ --> INL[Isolated Node Linker]
    end

    subgraph "Link Pruner"
        LP -->|Check Age| AGE{Edge > 90 days?}
        LP -->|Check Weight| WEIGHT{Weight < 0.3?}
        LP -->|Check Orphan| ORPHAN{Orphaned Edge?}
        AGE -->|Yes| REMOVE1[Remove Edge]
        WEIGHT -->|Yes| REMOVE2[Remove Edge]
        ORPHAN -->|Yes| REMOVE3[Remove Edge]
    end

    subgraph "Relation Suggester"
        RS -->|Vector Search| VS[Find Similar Notes]
        RS -->|Check Similarity| SIM{Similarity ≥ 0.75?}
        RS -->|Check Existing| EXIST{Link exists?}
        SIM -->|Yes| SUGGEST[Suggest Relation]
        EXIST -->|No| CREATE[Create Link]
    end

    subgraph "Summary Digester"
        SD -->|Check Children| CHILDREN{>8 children?}
        CHILDREN -->|Yes| COMPRESS[Compress Summary]
        COMPRESS -->|Update| GS[GraphStore]
    end

    SCHED -->|Trigger| ENZ
    REMOVE1 --> EVT[Event Logger]
    REMOVE2 --> EVT
    REMOVE3 --> EVT
    CREATE --> EVT
    COMPRESS --> EVT

    style SCHED fill:#e1f5ff,color:#000000
    style ENZ fill:#f3e5f5,color:#000000
    style EVT fill:#fff3e0,color:#000000
```

## Researcher Agent Workflow

```mermaid
graph TD
    subgraph "Trigger Conditions"
        RET[Memory Retrieval]
        RET -->|Confidence < 0.5| TRIGGER[Trigger Researcher]
        MANUAL[MCP Tool Call] -->|research_and_store| TRIGGER
    end

    subgraph "Research Process"
        TRIGGER --> RA[ResearcherAgent]
        RA --> SEARCH[Web Search]
        SEARCH -->|Primary| GSAPI[Google Search API]
        SEARCH -->|Fallback| DDG[DuckDuckGo HTTP]
        GSAPI -->|Results| EXTRACT[Content Extraction]
        DDG -->|Results| EXTRACT
        EXTRACT -->|Primary| JINA[Jina Reader]
        EXTRACT -->|PDF| UNSTR[Unstructured]
        EXTRACT -->|Fallback| READ[Readability]
        JINA -->|Content| CREATE[Note Creation]
        UNSTR -->|Content| CREATE
        READ -->|Content| CREATE
        CREATE -->|LLM Metadata| META[Extract Type, Keywords, Tags]
        META -->|AtomicNote| STORE[Store via MemoryController]
    end

    subgraph "Storage"
        STORE --> MC[MemoryController.create_note]
        MC --> VS[VectorStore]
        MC --> GS[GraphStore]
        MC --> EVO[Memory Evolution]
    end

    style TRIGGER fill:#ffebee,color:#000000
    style RA fill:#e1f5ff,color:#000000
    style STORE fill:#e8f5e9,color:#000000
```

## Storage Architecture Detail

```mermaid
graph TB
    subgraph "Dual-Storage Architecture"
        SM[StorageManager]
        SM --> VS[VectorStore<br/>ChromaDB]
        SM --> GS[GraphStore<br/>NetworkX DiGraph]
    end

    subgraph "VectorStore Operations"
        VS -->|add| VADD[Add Embedding]
        VS -->|query| VQUERY[Similarity Search]
        VS -->|update| VUPDATE[Update Embedding]
        VS -->|delete| VDELETE[Remove Embedding]
        VADD --> CHROMA[(ChromaDB<br/>Persistent Storage)]
        VQUERY --> CHROMA
        VUPDATE --> CHROMA
        VDELETE --> CHROMA
    end

    subgraph "GraphStore Operations"
        GS -->|add_node| GADD[Add Node]
        GS -->|add_edge| GEDGE[Add Typed Edge]
        GS -->|get_connected| GCONN[Get Connected Notes]
        GS -->|save_snapshot| GSAVE[Save to JSON]
        GADD --> GRAPH[(knowledge_graph.json<br/>NetworkX Format)]
        GEDGE --> GRAPH
        GCONN --> GRAPH
        GSAVE --> GRAPH
        GS -->|lock| LOCK[Cross-Platform Lock<br/>graph.lock]
    end

    subgraph "Edge Properties"
        GEDGE --> TYPE[relation_type<br/>related_to, part_of, etc.]
        GEDGE --> REASON[reasoning<br/>Why linked?]
        GEDGE --> WEIGHT[weight<br/>0.0 - 1.0]
    end

    style VS fill:#e8f5e9,color:#000000
    style GS fill:#e1f5ff,color:#000000
    style CHROMA fill:#fff3e0,color:#000000
    style GRAPH fill:#f3e5f5,color:#000000
```

## Type Classification System

```mermaid
graph LR
    subgraph "Note Types"
        INPUT[Note Content] --> LLM[LLM Classification]
        LLM -->|Classify| TYPE{Note Type}
        TYPE -->|Imperative| RULE[rule<br/>Never X, Always Y]
        TYPE -->|Steps| PROC[procedure<br/>Numbered steps]
        TYPE -->|Explanation| CONCEPT[concept<br/>Explanations]
        TYPE -->|Function| TOOL[tool<br/>APIs, Functions]
        TYPE -->|Data| REF[reference<br/>Tables, Lists]
        TYPE -->|Connection| INT[integration<br/>System Links]
    end

    subgraph "Priority Calculation"
        RULE -->|Weight: 1.0| PRIO[Priority Score]
        PROC -->|Weight: 0.9| PRIO
        CONCEPT -->|Weight: 0.7| PRIO
        TOOL -->|Weight: 0.8| PRIO
        REF -->|Weight: 0.6| PRIO
        INT -->|Weight: 0.7| PRIO
        PRIO -->|+ Age Factor| FINAL[Final Priority]
        PRIO -->|+ Usage Count| FINAL
        PRIO -->|+ Edge Count| FINAL
    end

    style TYPE fill:#e1f5ff,color:#000000
    style PRIO fill:#fff3e0,color:#000000
```

## MCP Tools Overview

```mermaid
graph TB
    subgraph "Core Memory Operations"
        T1[create_atomic_note]
        T2[retrieve_memories]
        T3[get_memory_stats]
        T4[add_file]
        T5[reset_memory]
    end

    subgraph "Note Management"
        T6[list_notes]
        T7[get_note]
        T8[update_note]
        T9[delete_atomic_note]
    end

    subgraph "Relation Management"
        T10[list_relations]
        T11[add_relation]
        T12[remove_relation]
    end

    subgraph "Graph Operations"
        T13[get_graph]
    end

    subgraph "Memory Maintenance"
        T14[run_memory_enzymes]
    end

    subgraph "Research & Web"
        T15[research_and_store]
    end

    T1 --> MC[MemoryController]
    T2 --> MC
    T3 --> MC
    T4 --> MC
    T5 --> MC
    T6 --> MC
    T7 --> MC
    T8 --> MC
    T9 --> MC
    T10 --> MC
    T11 --> MC
    T12 --> MC
    T13 --> MC
    T14 --> MC
    T15 --> RA[ResearcherAgent]

    style T1 fill:#e8f5e9,color:#000000
    style T2 fill:#e8f5e9,color:#000000
    style T14 fill:#fff3e0,color:#000000
    style T15 fill:#f3e5f5,color:#000000
```

---

**Erstellt:** 2025-01-XX  
**Version:** 1.0  
**Basierend auf:** A-MEM Agentic Memory System Implementation

