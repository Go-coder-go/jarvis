# services/agent.py
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode

from pymongo import MongoClient

from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.tools import tool as tool_decorator

from .db import get_mongo_client, get_diagram_mongo_client
from .tools import generate_random_name, rotate_strings
from .store_doc import upsert_doc
from .support_store import upsert_support_case_from_json
from .diagram_store import parse_excalidraw_scene, build_vector_docs_from_graph

# ---------- State ----------
class AgentState(TypedDict):
    # messages will be appended via add_messages so tools & LLM calls form a chain
    messages: Annotated[List[BaseMessage], add_messages]


# ---------- Tool: return only doc URL(s) ----------
def build_doc_link_tool(
    client: MongoClient,
    db_name: str = "knowledge_base",
    collection_name: str = "docs",
    index_name: str = "vector_index",
    text_key: str = "embedding_text",
    embedding_key: str = "embedding",
) -> Tool:
    """
    Search docs like doc_search but return only a small JSON array of URLs (and titles).
    This is intended for use-cases where the user only wants the link(s).
    """
    embeddings = OpenAIEmbeddings()
    collection = client[db_name][collection_name]

    from langchain_core.tools import tool as tool_decorator

    @tool_decorator(
        "doc_link",
        description=(
                "You are a helpful Knowledge Base Chatbot Agent for docs, processes and flow."
                "If a user explicitly requests ONLY the doc link (e.g. 'give me the doc link' or 'send link'), "
                "call the doc_link tool and return only the URL(s) with no additional commentary. "
                "return only the relevant doc not all the doc links."
                "Otherwise use doc_search for richer info."
            ),
    )
    def _wrapper(query: str, k: int = 1) -> str:
        vs = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=collection,
            index_name=index_name,
            text_key=text_key,
            embedding_key=embedding_key,
        )
        docs = vs.similarity_search_with_score(query, k=k)
        out = []
        for doc, score in docs:
            meta = doc.metadata or {}
            title = meta.get("title") or (doc.page_content[:120] + "...")
            url = meta.get("url") or meta.get("link") or ""
            out.append({"title": title, "url": url, "score": score})
        # Return compact JSON so model sees only urls
        return json.dumps(out, default=str)

    return _wrapper

# ---------- Tools: Vector search for Docs ----------
def build_doc_search_tool(
    client: MongoClient,
    db_name: str = "knowledge_base",
    collection_name: str = "docs",
    index_name: str = "vector_index",
    text_key: str = "embedding_text",
    embedding_key: str = "embedding",
) -> Tool:
    embeddings = OpenAIEmbeddings()
    collection = client[db_name][collection_name]

    from langchain_core.tools import tool as tool_decorator

    @tool_decorator(
        "doc_search",
        description=(
                "You are a helpful Knowledge Base Chatbot Agent for docs, processes and flow."
                "When using doc_search, ALWAYS take ONLY the top result."
                "Do NOT list multiple documents unless the user explicitly asks for 'all', 'list', 'multiple', or 'show more'."
                "From doc_search results: ONLY use the most relevant document."
                "If the user asks for a link, respond ONLY with the link associated with the top document."
                "Your answers must remain short, direct, and relevant."
        ),
    )
    def _wrapper(query: str, k: int = 1) -> str:
        vs = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=collection,
            index_name=index_name,
            text_key=text_key,
            embedding_key=embedding_key,
        )
        docs = vs.similarity_search_with_score(query, k=k)

        results = []
        for doc, score in docs:
            # metadata may contain _id, created_at, etc.
            metadata = dict(doc.metadata) if doc.metadata else {}
            # You can optionally drop raw _id if you don't care:
            # metadata.pop("_id", None)

            results.append(
                {
                    "page_content": doc.page_content,
                    "metadata": metadata,
                    "score": score,
                }
            )

        return json.dumps(results, default=str)

    return _wrapper


def build_support_insert_tool(
    client: MongoClient,
    db_name: str = "support_knowledge_base",
    collection_name: str = "support_cases",
):
    """
    Tool for inserting/updating support/error cases in the support knowledge base vector DB.

    This behaves similarly to doc_insert, but is dedicated to SUPPORT / INCIDENT data:
      - support chat threads
      - incident reports
      - on-call tickets
      - debugging sessions
      - production issues
      - outage investigations

    The LLM should call this tool when the user provides:
      - a support thread
      - a troubleshooting conversation
      - any problem investigation context
      - a JSON-like object describing an issue
      - a ticket they want to save for future retrieval

    This tool stores:
      title, summary, content, metadata, file, date, tags, doc_link, AND resolution.
    Resolution is optional and stored separately (not part of the embedding).
    """

    @tool_decorator(
        "support_case_insert",
        description=(
            "Insert or update a SUPPORT / INCIDENT case in the support knowledge base.\n\n"
            "Use this tool ONLY for support issues, production incidents, outage investigations, "
            "debugging threads, or on-call problems. "
            "Do NOT use this for normal documentation (use doc_insert for that).\n\n"

            "Arguments:\n"
            "- case_id: unique identifier for this support case (file name, slug, ticket id, etc.)\n"
            "- title: short human-readable title of the issue\n"
            "- summary: 1–3 sentence summary of what happened or what the issue is\n"
            "- content (optional): longer extracted text or conversation from the support thread\n"
            "- resolution (optional): how the issue was ultimately fixed (stored separately; NOT used in embeddings)\n"
            "- doc_link (optional): link to wiki or documentation related to this case\n"
            "- tags (optional): list of tags describing the issue (e.g. ['whatsapp','pinjam','timeouts'])\n"
            "- file (optional): source file or attachment name (e.g. '4.pdf')\n"
            "- date (optional): ISO timestamp string describing when the issue occurred\n\n"

            "The tool will store all of this in the support knowledge base for future similarity search "
            "via support_case_search."
        ),
    )
    def _wrapper(
        case_id: str,
        title: str,
        summary: str,
        content: Optional[str] = None,
        resolution: Optional[str] = None,
        doc_link: Optional[str] = None,
        tags: Optional[List[str]] = None,
        file: Optional[str] = None,
        date: Optional[str] = None,
    ) -> str:
        """
        Upsert a support case into MongoDB + vector DB.
        Works similarly to doc_insert but specifically for support/incident data.
        """
        try:
            metadata: Dict[str, Any] = {}
            if file:
                metadata["file"] = file
            if date:
                metadata["date"] = date
            if doc_link:
                metadata["doc_link"] = doc_link
            if tags:
                metadata["tags"] = tags

            case_dict = {
                "case_id": case_id,
                "title": title,
                "summary": summary,
                "content": content,
                "resolution": resolution,   # NEW FIELD STORED
                "metadata": metadata,
                "tags": tags,
            }

            doc = upsert_support_case_from_json(
                client=client,
                case_json=case_dict,
                db_name=db_name,
                collection_name=collection_name,
            )

            return json.dumps(
                {
                    "status": "ok",
                    "case_id": doc.get("case_id"),
                    "title": doc.get("title"),
                    "tags": doc.get("tags"),
                    "resolution": doc.get("resolution"),
                },
                default=str,
            )
        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                }
            )

    return _wrapper



def build_support_search_tool(
    client: MongoClient,
    db_name: str = "support_knowledge_base",
    collection_name: str = "support_cases",
    index_name: str = "vector_index",   # Atlas index name
    text_key: str = "embedding_text",
    embedding_key: str = "embedding",
):
    """
    Tool for searching similar past support cases based on a natural language query.
    """

    embeddings = OpenAIEmbeddings()
    collection = client[db_name][collection_name]

    @tool_decorator(
        "support_case_search",
        description=(
            "Searches past support queries / tickets in the support knowledge base. "
            "Input is a natural language description of the issue; output is a list "
            "of matching cases with title, summary, resolution (if present), doc_link, "
            "tags, and similarity score."
        ),
    )
    def _wrapper(query: str, k: int = 5) -> str:
        vs = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=collection,
            index_name=index_name,
            text_key=text_key,
            embedding_key=embedding_key,
        )

        docs = vs.similarity_search_with_score(query, k=k)
        results: List[Dict[str, Any]] = []

        for doc, score in docs:
            # All non-embedding fields land in metadata
            meta = dict(doc.metadata) if doc.metadata else {}

            title = meta.get("title")
            summary = meta.get("summary")
            tags = meta.get("tags")
            resolution = meta.get("resolution")          # <-- NEW
            case_metadata = meta.get("metadata") or {}
            doc_link = case_metadata.get("doc_link")

            results.append(
                {
                    "title": title,
                    "summary": summary,
                    "resolution": resolution,           # <-- NEW
                    "doc_link": doc_link,
                    "tags": tags,
                    "score": score,
                }
            )

        return json.dumps(results, default=str)

    return _wrapper


# ---------- NEW: Tool to INSERT/UPDATE docs in DB ----------
def build_doc_insert_tool(
    client: MongoClient,
    db_name: str = "knowledge_base",
    collection_name: str = "docs",
) -> Tool:
    """
    Build a LangChain tool that inserts/updates documents in the knowledge base.
    The LLM can call this to register new docs (with title, url, description, tags, etc.)
    into the vector DB.
    """
    from langchain_core.tools import tool as tool_decorator

    @tool_decorator(
        "doc_insert",
        description=(
            "Insert or update a document in the internal document knowledge base vector DB for document related data "
            "Use this when the user provides a new doc/link that should be remembered. "
            "Provide a meaningful doc_id (slug), title, url, description, tags, and optional content."
        ),
    )
    def _wrapper(
        doc_id: str,
        title: str,
        url: str,
        description: str,
        tags: Optional[List[str]] = None,
        content: Optional[str] = None,
    ) -> str:
        """
        Upsert a knowledge base document and its embedding into MongoDB.
        """
        try:
            print("running upsert doc or save doc function:")
            doc = upsert_doc(
                client=client,
                doc_id=doc_id,
                title=title,
                url=url,
                description=description,
                tags=tags,
                content=content,
                db_name=db_name,
                collection_name=collection_name,
            )

            # Make datetimes JSON-serializable
            if isinstance(doc.get("created_at"), datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            if isinstance(doc.get("updated_at"), datetime):
                doc["updated_at"] = doc["updated_at"].isoformat()

            return json.dumps(
                {
                    "status": "ok",
                    "doc": doc,
                }
            )
        except Exception as e:
            print("eeror while saving doc - " , e)
            return json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                }
            )

    return _wrapper


def build_diagram_insert_tool(
    client: MongoClient,
    db_name: str = "diagram_knowledge_base",
    collection_name: str = "diagram_vectors",
) -> Tool:
    """
    Insert / update an architecture diagram derived from Excalidraw JSON into a vector DB.
    Stores each component & relationship as a separate vector document.
    """
    print("inserting diagram 1 ", client)
    @tool_decorator(
        "diagram_insert",
        description=(
            "Parse and store an architecture diagram from Excalidraw JSON into the diagram vector DB.\n\n"
            "Use this tool when the user provides an Excalidraw JSON (the full diagram), a title for the diagram, "
            "and a short summary of what the architecture represents.\n\n"
            "Arguments:\n"
            "- diagram_id: a stable unique id for this diagram (e.g. 'payments-arch-v1')\n"
            "- title: human-readable title of the diagram\n"
            "- summary: 1–3 sentence summary of what the diagram represents\n"
            "- excalidraw_json: raw string containing the Excalidraw JSON for the diagram\n"
            "- tags (optional): list of tags describing the system (e.g. ['payments','whatsapp','service-mesh'])\n\n"
            "The tool will parse nodes and edges, generate natural-language descriptions for each component "
            "and relationship, embed them, and store them in the 'diagram_vectors' collection. "
            "This enables later semantic search via diagram_search."
        ),
    )
    def _wrapper(
        diagram_id: str,
        title: str,
        summary: str,
        excalidraw_json: str,
        tags: Optional[List[str]] = None,
    ) -> str:
        try:
            
            scene = json.loads(excalidraw_json)
            print("diagram json loaded")
            
        except Exception as e:
            print("error while loading jsong")
            
            return json.dumps(
                {"status": "error", "error": f"Invalid Excalidraw JSON: {e}"}
            )

        try:
            parsed = parse_excalidraw_scene(scene, diagram_id=diagram_id)
            print("done parsing diagram json")
            
            vector_docs = build_vector_docs_from_graph(
                parsed,
                diagram_id=diagram_id,
                diagram_title=title,
                diagram_summary=summary,
                tags=tags,
            )

            if not vector_docs:
                return json.dumps(
                    {
                        "status": "error",
                        "error": "Parsed diagram produced no nodes/edges to store.",
                    }
                )

            embeddings = OpenAIEmbeddings()
            texts = [d["embedding_text"] for d in vector_docs]
            vectors = embeddings.embed_documents(texts)

            collection = client[db_name][collection_name]

            # Optional: store/refresh high-level metadata for the diagram
            meta_coll = client[db_name]["diagrams_meta"]
            now = datetime.utcnow()
            meta_coll.update_one(
                {"_id": diagram_id},
                {
                    "$set": {
                        "title": title,
                        "summary": summary,
                        "tags": tags or [],
                        "updated_at": now,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )

            # Clear old docs for this diagram to avoid duplicates
            collection.delete_many({"diagram_id": diagram_id})

            docs_to_insert = []
            for doc, vec in zip(vector_docs, vectors):
                doc_record = {
                    **doc,
                    "embedding": vec,
                    "created_at": now,
                    "updated_at": now,
                }
                docs_to_insert.append(doc_record)

            if docs_to_insert:
                collection.insert_many(docs_to_insert)

            return json.dumps(
                {
                    "status": "ok",
                    "diagram_id": diagram_id,
                    "title": title,
                    "summary": summary,
                    "stored_chunks": len(docs_to_insert),
                },
                default=str,
            )
        except Exception as e:
            print("error while inserting diagram")
            
            return json.dumps({"status": "error", "error": str(e)})

    return _wrapper


def build_diagram_search_tool(
    client: MongoClient,
    db_name: str = "diagram_knowledge_base",
    collection_name: str = "diagram_vectors",
    index_name: str = "vector_index",
    text_key: str = "embedding_text",
    embedding_key: str = "embedding",
) -> Tool:
    """
    Search stored architecture diagrams (components and relationships) using vector similarity.
    """

    embeddings = OpenAIEmbeddings()
    collection = client[db_name][collection_name]

    @tool_decorator(
        "diagram_search",
        description=(
            "Semantic search over stored architecture diagrams parsed from Excalidraw.\n\n"
            "Use this when the user asks questions about how a system is wired, which components "
            "talk to which, data flow paths, or responsibilities of components, and you know "
            "there are diagrams stored in the diagram knowledge base.\n\n"
            "Arguments:\n"
            "- query: natural language question about the architecture\n"
            "- diagram_id (optional): if provided, restrict search to a specific diagram id\n"
            "- k (optional, default 5): number of results\n\n"
            "The tool returns a list of chunks describing nodes, their neighbourhood, and edges. "
            "Use these chunks to answer the user's question in your own words."
        ),
    )
    def _wrapper(
        query: str,
        diagram_id: Optional[str] = None,
        k: int = 5,
    ) -> str:
        vs = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=collection,
            index_name=index_name,
            text_key=text_key,
            embedding_key=embedding_key,
        )

        # Basic similarity search
        docs = vs.similarity_search_with_score(query, k=k)

        results: List[Dict[str, Any]] = []
        for doc, score in docs:
            meta = dict(doc.metadata) if doc.metadata else {}
            # Filter client-side by diagram_id if provided
            d_id = getattr(doc, "diagram_id", None) or meta.get("diagram_id")
            if diagram_id and d_id != diagram_id:
                # skip mismatched diagrams
                continue

            results.append(
                {
                    "diagram_id": d_id,
                    "diagram_title": meta.get("diagram_title"),
                    "object_type": meta.get("object_type") or doc.metadata.get("object_type") if doc.metadata else None,
                    "text": doc.page_content if hasattr(doc, "page_content") else doc.page_content,
                    "metadata": meta,
                    "score": score,
                }
            )

        return json.dumps(results, default=str)

    return _wrapper


# ---------- Agent core ----------
def build_app(client: MongoClient, diagram_client:  MongoClient):
    tools: List[Tool] = [
        build_doc_search_tool(client),
        build_doc_insert_tool(client),
        build_doc_link_tool(client),
        build_support_search_tool(client),
        build_support_insert_tool(client),
        build_diagram_insert_tool(diagram_client),
        build_diagram_search_tool(diagram_client),
        generate_random_name,
        rotate_strings,
    ]

    tool_node = ToolNode(tools)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    async def call_model(state: AgentState) -> AgentState:
        """
        Node that calls the LLM.
        State shape: {"messages": [BaseMessage, ...]}.
        """
        messages = state.get("messages", [])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful Knowledge Base Chatbot Agent. "
                    "You can:\n"
                    "- Search internal docs from a vector DB (doc_search)\n"
                    "- Insert or update docs in the documentation Knowledge base (doc_insert)\n"
                    "- Insert or update SUPPORT / INCIDENT cases in the support KB (support_case_insert)\n"
                    "- if a insert or update support request contains Doc then saves the support issue in support knowledge base and save Doc in documents knowledge base with relevant details"
                    "- Search past support cases / tickets (support_case_search) and provide support based on the previous support and provide steps to resolve the issue. if multiple ways are present then tell all possible approaches\n"
                    "- If user keeps saying that this method for resolving the support or error issue then provide some other approach if present in support issue other guide him to contact comms team oncall"
                    "- Parse and store Excalidraw architecture diagrams (diagram_insert)\n"
                    "- Search stored architecture diagrams to explain system behaviour (diagram_search)\n"
                    "- Generate random names (generate_random_name)\n"
                    "- Rotate strings (rotate_strings)\n\n"
                    "When the user provides a support chat thread / incident JSON and asks you to "
                    "store or save it as a support case, ALWAYS call support_case_insert with that JSON. "
                    "When they provide a normal documentation link or doc, use doc_insert instead.\n\n"
                    "Use tools when useful. If you have the final answer, "
                    "If you do not have any information for Doc or support or error issue then ask user to contact communication platform team"
                    "Tools available: {tool_names}. "
                    "Current time: {time}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        formatted = await prompt.ainvoke(
            {
                "tool_names": ", ".join([t.name for t in tools]),
                "time": datetime.utcnow().isoformat(),
                "messages": messages,
            }
        )

        result: AIMessage = await model.ainvoke(formatted)
        # LangGraph expects a partial state update
        return {"messages": [result]}

    def should_continue(state: AgentState) -> str:
        """
        Decide whether to call tools again or end the graph.
        """
        messages = state.get("messages", [])
        if not messages:
            return "__end__"

        last = messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge("__start__", "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app


# Build app once so thread_id memory works across calls
_mongo_client = get_mongo_client()
_diagram_mongo_client = get_diagram_mongo_client()
_app = build_app(_mongo_client, _diagram_mongo_client)




async def call_agent(query: str, thread_id: str) -> str:
    """
    Entrypoint used by Django service layer or CLI.
    Returns the final assistant text.
    """
    final_state: AgentState = await _app.ainvoke(
        {"messages": [HumanMessage(query)]},
        {"configurable": {"thread_id": thread_id}, "recursion_limit": 15},
    )

    messages = final_state.get("messages", [])
    if not messages:
        return ""

    last: BaseMessage = messages[-1]
    content = getattr(last, "content", "")
    if isinstance(content, list):
        # Sometimes content can be a list of chunks; flatten to string
        return "".join(str(c) for c in content)
    return str(content)
