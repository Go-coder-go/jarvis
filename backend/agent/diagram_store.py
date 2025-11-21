# ---------- Excalidraw helpers for diagrams ----------

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

NODE_TYPES = {"rectangle", "diamond", "ellipse", "roundRectangle"}
DEFAULT_TEXT_COLOR = "#1e1e1e"


@dataclass
class DiagramNode:
    diagram_id: str
    excalidraw_id: str
    name: str
    label: str
    shape: str
    x: float
    y: float
    width: float
    height: float
    style: Dict[str, Any]
    link: Optional[str] = None
    parent_id: Optional[str] = None


@dataclass
class DiagramEdge:
    diagram_id: str
    excalidraw_id: str
    from_excalidraw_id: str
    to_excalidraw_id: str
    from_name: str
    to_name: str
    label: str
    arrow: str
    style: Dict[str, Any]


def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    s = text.replace("\n", " ")
    s = " ".join(s.split())
    return s


def _collect_container_and_link_text(
    elements: List[Dict[str, Any]]
) -> tuple[Dict[str, str], Dict[str, str], Dict[str, float], Dict[str, str]]:
    id_to_el = {el.get("id"): el for el in elements}
    container_text: Dict[str, str] = {}
    container_text_color: Dict[str, str] = {}
    container_font_size: Dict[str, float] = {}
    link_text: Dict[str, str] = {}

    for el in elements:
        if el.get("isDeleted"):
            continue
        if el.get("type") == "text" and el.get("containerId"):
            container_id = el["containerId"]
            parent = id_to_el.get(container_id)
            if not parent:
                continue
            parent_type = parent.get("type")

            # Text inside nodes
            if parent_type in NODE_TYPES:
                if container_id not in container_text:
                    container_text[container_id] = ""
                if container_text[container_id]:
                    container_text[container_id] += " "
                container_text[container_id] += _sanitize_text(el.get("text", ""))

                stroke_color = el.get("strokeColor", DEFAULT_TEXT_COLOR)
                if stroke_color and stroke_color not in (DEFAULT_TEXT_COLOR, "black"):
                    container_text_color[container_id] = stroke_color

                if container_id not in container_font_size:
                    container_font_size[container_id] = el.get("fontSize", 16.0)

            # Text attached to line/arrow (edge label)
            elif parent_type in ("arrow", "line"):
                if container_id not in link_text:
                    link_text[container_id] = ""
                if link_text[container_id]:
                    link_text[container_id] += " "
                link_text[container_id] += _sanitize_text(el.get("text", ""))

    return container_text, container_text_color, container_font_size, link_text


def _get_node_shape_type(el_type: str) -> str:
    if el_type == "rectangle":
        return "rectangle"
    if el_type == "roundRectangle":
        return "roundRectangle"
    if el_type == "ellipse":
        return "ellipse"
    if el_type == "diamond":
        return "diamond"
    return el_type


def _get_node_style(
    el: Dict[str, Any],
    container_font_size: Dict[str, float],
    container_text_color: Dict[str, str],
) -> Dict[str, Any]:
    style: Dict[str, Any] = {}

    stroke_style = el.get("strokeStyle")
    if stroke_style == "dashed":
        style["strokeDasharray"] = "5 5"
    elif stroke_style == "dotted":
        style["strokeDasharray"] = "2 2"

    stroke_color = el.get("strokeColor")
    if stroke_color:
        style["stroke"] = stroke_color

    stroke_width = el.get("strokeWidth")
    if stroke_width == 4:
        style["strokeWidth"] = 2
    elif stroke_width == 1:
        style["strokeWidth"] = 0.5

    bg_color = el.get("backgroundColor")
    if bg_color:
        style["fill"] = bg_color

    opacity = el.get("opacity", 100.0)
    if opacity < 100:
        style["opacity"] = opacity / 100.0

    node_id = el.get("id")
    if node_id in container_font_size:
        style["fontSize"] = container_font_size[node_id]

    if node_id in container_text_color:
        color = container_text_color[node_id]
        if not color.startswith("#") and color not in ("black", "white"):
            color = "#" + color
        style["textColor"] = color

    return style


def _detect_spatial_containment(
    elements: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    childNodeExcalidrawId -> parentNodeExcalidrawId
    """
    node_elements = [
        el for el in elements
        if not el.get("isDeleted") and el.get("type") in NODE_TYPES
    ]

    contained_by: Dict[str, str] = {}

    def box_and_area(el: Dict[str, Any]):
        x1 = el.get("x", 0.0)
        y1 = el.get("y", 0.0)
        x2 = x1 + el.get("width", 0.0)
        y2 = y1 + el.get("height", 0.0)
        area = el.get("width", 0.0) * el.get("height", 0.0)
        return x1, y1, x2, y2, area

    id_to_el = {el["id"]: el for el in node_elements}
    boxes = {el["id"]: box_and_area(el) for el in node_elements}

    for child_id in id_to_el.keys():
        cx1, cy1, cx2, cy2, _ = boxes[child_id]
        best_parent = None
        best_area = None

        for parent_id in id_to_el.keys():
            if parent_id == child_id:
                continue
            px1, py1, px2, py2, p_area = boxes[parent_id]
            if cx1 >= px1 and cy1 >= py1 and cx2 <= px2 and cy2 <= py2:
                if best_parent is None or p_area < best_area:
                    best_parent = parent_id
                    best_area = p_area

        if best_parent:
            contained_by[child_id] = best_parent

    return contained_by


def _construct_edge_arrow(el_type: str, end_arrowhead: str, stroke_style: str) -> str:
    arrow = "--"

    if end_arrowhead == "arrow":
        arrow = "-->"
    elif end_arrowhead in ("circle_outline", "circle"):
        arrow = "--o"
    elif end_arrowhead == "arrow_bidirectional":
        arrow = "<-->"
    elif end_arrowhead == "circle_outline_bidirectional":
        arrow = "o--o"
    else:
        if el_type == "arrow":
            arrow = "-->"

    if stroke_style in ("dashed", "dotted"):
        arrow = f"dotted:{arrow}"

    return arrow


def parse_excalidraw_scene(scene: Dict[str, Any], diagram_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse Excalidraw JSON into node + edge structures.
    """
    elements = scene.get("elements", [])
    (
        container_text,
        container_text_color,
        container_font_size,
        link_text,
    ) = _collect_container_and_link_text(elements)

    nodes: Dict[str, DiagramNode] = {}
    edges: List[DiagramEdge] = []
    node_count = 0

    contained_by = _detect_spatial_containment(elements)

    # Nodes
    for el in elements:
        if el.get("isDeleted"):
            continue
        el_type = el.get("type")
        if el_type not in NODE_TYPES:
            continue

        excalidraw_id = el["id"]
        name = f"N{node_count}"
        node_count += 1

        label = container_text.get(excalidraw_id, "").strip() or " "
        shape = _get_node_shape_type(el_type)

        node = DiagramNode(
            diagram_id=diagram_id,
            excalidraw_id=excalidraw_id,
            name=name,
            label=label,
            shape=shape,
            x=el.get("x", 0.0),
            y=el.get("y", 0.0),
            width=el.get("width", 0.0),
            height=el.get("height", 0.0),
            style=_get_node_style(el, container_font_size, container_text_color),
            link=(el.get("link") or None) if el.get("link") != "null" else None,
            parent_id=contained_by.get(excalidraw_id),
        )

        nodes[excalidraw_id] = node

    # Edges
    for el in elements:
        if el.get("isDeleted"):
            continue
        el_type = el.get("type")
        if el_type not in ("arrow", "line"):
            continue

        excalidraw_id = el["id"]
        start_binding = el.get("startBinding") or {}
        end_binding = el.get("endBinding") or {}
        start_id = start_binding.get("elementId")
        end_id = end_binding.get("elementId")
        if not start_id or not end_id:
            continue
        if start_id not in nodes or end_id not in nodes:
            continue

        stroke_style = el.get("strokeStyle", "")
        end_arrowhead = el.get("endArrowhead", "")
        arrow = _construct_edge_arrow(el_type, end_arrowhead, stroke_style)

        label = link_text.get(excalidraw_id, "").strip()
        if not label and el.get("text"):
            label = _sanitize_text(el["text"])

        style: Dict[str, Any] = {}
        if stroke_style:
            style["strokeStyle"] = stroke_style
        stroke_color = el.get("strokeColor")
        if stroke_color:
            style["stroke"] = stroke_color
        stroke_width = el.get("strokeWidth")
        if stroke_width == 4:
            style["strokeWidth"] = 2
        elif stroke_width == 1:
            style["strokeWidth"] = 0.5
        opacity = el.get("opacity", 100.0)
        if opacity < 100:
            style["opacity"] = opacity / 100.0

        edge = DiagramEdge(
            diagram_id=diagram_id,
            excalidraw_id=excalidraw_id,
            from_excalidraw_id=start_id,
            to_excalidraw_id=end_id,
            from_name=nodes[start_id].name,
            to_name=nodes[end_id].name,
            label=label,
            arrow=arrow,
            style=style,
        )
        edges.append(edge)

    node_docs = [asdict(n) for n in nodes.values()]
    edge_docs = [asdict(e) for e in edges]
    return {"nodes": node_docs, "edges": edge_docs}


def build_vector_docs_from_graph(
    parsed: Dict[str, List[Dict[str, Any]]],
    diagram_id: str,
    diagram_title: str,
    diagram_summary: str,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Turn parsed nodes + edges into natural language vector docs.
    """
    nodes = parsed["nodes"]
    edges = parsed["edges"]
    node_by_id = {n["excalidraw_id"]: n for n in nodes}

    outgoing: Dict[str, List[Dict[str, Any]]] = {n["excalidraw_id"]: [] for n in nodes}
    incoming: Dict[str, List[Dict[str, Any]]] = {n["excalidraw_id"]: [] for n in nodes}
    for e in edges:
        s = e["from_excalidraw_id"]
        t = e["to_excalidraw_id"]
        if s in outgoing:
            outgoing[s].append(e)
        if t in incoming:
            incoming[t].append(e)

    docs: List[Dict[str, Any]] = []

    # 1. Node docs
    for n in nodes:
        parent_text = ""
        if n.get("parent_id"):
            parent = node_by_id.get(n["parent_id"])
            if parent:
                parent_text = f" It is contained inside '{parent['label']}' ({parent['name']})."

        link_text = ""
        if n.get("link"):
            link_text = f" It has an associated link: {n['link']}."

        text = (
            f"In diagram '{diagram_title}', node {n['name']} represents '{n['label']}'. "
            f"It is a {n['shape']} component in the system architecture.{parent_text}{link_text}"
        )

        docs.append(
            {
                "diagram_id": diagram_id,
                "diagram_title": diagram_title,
                "diagram_summary": diagram_summary,
                "object_type": "node",
                "object_id": n["excalidraw_id"],
                "object_name": n["name"],
                "embedding_text": text,
                "metadata": {
                    "node_label": n["label"],
                    "shape": n["shape"],
                    "parent_id": n.get("parent_id"),
                    "link": n.get("link"),
                    "tags": tags or [],
                },
            }
        )

    # 2. Node + neighbourhood docs
    for n in nodes:
        nid = n["excalidraw_id"]
        label = n["label"]
        out_edges = outgoing.get(nid, [])
        in_edges = incoming.get(nid, [])

        out_desc_parts: List[str] = []
        for e in out_edges:
            target = node_by_id.get(e["to_excalidraw_id"])
            if not target:
                continue
            rel = e.get("label") or ""
            seg = f"{label} sends data to {target['label']}"
            if rel:
                seg += f" via '{rel}'"
            out_desc_parts.append(seg)

        in_desc_parts: List[str] = []
        for e in in_edges:
            src = node_by_id.get(e["from_excalidraw_id"])
            if not src:
                continue
            rel = e.get("label") or ""
            seg = f"{src['label']} sends data to {label}"
            if rel:
                seg += f" via '{rel}'"
            in_desc_parts.append(seg)

        if not out_desc_parts and not in_desc_parts:
            continue

        text = (
            f"In diagram '{diagram_title}', component '{label}' (node {n['name']}) participates in these relationships. "
        )
        if out_desc_parts:
            text += "Outgoing connections: " + "; ".join(out_desc_parts) + ". "
        if in_desc_parts:
            text += "Incoming connections: " + "; ".join(in_desc_parts) + "."

        docs.append(
            {
                "diagram_id": diagram_id,
                "diagram_title": diagram_title,
                "diagram_summary": diagram_summary,
                "object_type": "node_context",
                "object_id": nid,
                "object_name": n["name"],
                "embedding_text": text,
                "metadata": {
                    "node_label": label,
                    "outgoing_count": len(out_desc_parts),
                    "incoming_count": len(in_desc_parts),
                    "tags": tags or [],
                },
            }
        )

    # 3. Edge docs
    for e in edges:
        src = node_by_id.get(e["from_excalidraw_id"])
        dst = node_by_id.get(e["to_excalidraw_id"])
        if not src or not dst:
            continue
        label = e.get("label") or ""
        arrow = e.get("arrow") or ""

        text = (
            f"In diagram '{diagram_title}', there is a connection from '{src['label']}' "
            f"(node {src['name']}) to '{dst['label']}' (node {dst['name']}). "
        )
        if label:
            text += f"The relationship is described as '{label}'. "
        if arrow:
            text += f"The arrow type is '{arrow}'."

        docs.append(
            {
                "diagram_id": diagram_id,
                "diagram_title": diagram_title,
                "diagram_summary": diagram_summary,
                "object_type": "edge",
                "object_id": e["excalidraw_id"],
                "embedding_text": text,
                "metadata": {
                    "from_node": src["name"],
                    "to_node": dst["name"],
                    "from_label": src["label"],
                    "to_label": dst["label"],
                    "edge_label": label,
                    "arrow": arrow,
                    "tags": tags or [],
                },
            }
        )

    return docs
