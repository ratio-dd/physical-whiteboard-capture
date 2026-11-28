"""Small public native SVG serializer used by the live runtime."""
from __future__ import annotations
import html

def _path(points, closed=False):
    if not isinstance(points, (list, tuple)) or len(points) < (3 if closed else 2): return ""
    d = " ".join(("M" if i == 0 else "L") + f"{float(p[0]):g} {float(p[1]):g}" for i, p in enumerate(points))
    return d + (" Z" if closed else "")

def geometry_svg(paths, width: int, height: int) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">']
    for index, stroke in enumerate(paths or []):
        color = stroke.get("color", "#202824") if isinstance(stroke, dict) else "#202824"
        if not isinstance(color, str) or not color.startswith("#"): color = "#202824"
        color_attr = html.escape(color, quote=True)
        rings = stroke.get("outline", []) if isinstance(stroke, dict) else []
        layers = stroke.get("layers", []) if isinstance(stroke, dict) else []
        center = stroke.get("centerlines") or [] if isinstance(stroke, dict) else []
        primary = stroke.get("points", []) if isinstance(stroke, dict) else []
        group = []
        for layer in layers:
            d = " ".join(filter(None, (_path(r, True) for r in layer.get("outline", []))))
            layer_color = layer.get("color", color)
            if not isinstance(layer_color, str) or not layer_color.startswith("#"): layer_color = color
            if d: group.append(f'<path d="{html.escape(d, quote=True)}" fill="{html.escape(layer_color, quote=True)}" fill-rule="evenodd" data-geometry="outline"/>')
        if not layers and rings:
            d = " ".join(filter(None, (_path(r, True) for r in rings)))
            if d: group.append(f'<path d="{html.escape(d, quote=True)}" fill="{color_attr}" fill-rule="evenodd" data-geometry="outline"/>')
        lines = " ".join(filter(None, (_path(b.get("points", b), bool(b.get("closed"))) for b in center))) or _path(primary, bool(stroke.get("closed", False)))
        if lines:
            width_px = max(1.0, float(stroke.get("width", 1)))
            group.append(f'<path d="{html.escape(lines, quote=True)}" fill="none" stroke="{color_attr}" stroke-width="{width_px:g}" stroke-linecap="round" stroke-linejoin="round" data-geometry="line"/>')
        if group:
            ident = html.escape(str(stroke.get("id", f"stroke-{index:04d}")), quote=True)
            appearance = stroke.get("appearance", {}) if isinstance(stroke, dict) else {}
            opacity = appearance.get("opacity") if isinstance(appearance, dict) else None
            opacity_attr = f' opacity="{float(opacity):g}"' if isinstance(opacity, (int, float)) and 0 <= opacity <= 1 else ""
            out.append(f'<g id="{ident}" data-stroke-id="{ident}"{opacity_attr}>' + "".join(group) + "</g>")
    out.append("</svg>")
    return "".join(out)
