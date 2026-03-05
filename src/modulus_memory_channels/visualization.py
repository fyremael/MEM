from __future__ import annotations

from pathlib import Path
import csv


def write_csv(
    rows: list[dict[str, object]],
    path: str | Path,
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path = Path(path)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    start = points[0]
    segments = [f"M {start[0]:.2f} {start[1]:.2f}"]
    segments.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return " ".join(segments)


def write_line_svg(
    values: list[float],
    path: str | Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 640,
    height: int = 320,
) -> None:
    path = Path(path)
    if not values:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='320'/>", encoding="ascii")
        return

    margin_left = 56
    margin_right = 24
    margin_top = 32
    margin_bottom = 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    def scale_x(index: int) -> float:
        if len(values) == 1:
            return margin_left + (plot_width / 2.0)
        return margin_left + (plot_width * index / (len(values) - 1))

    def scale_y(value: float) -> float:
        normalized = (value - min_value) / (max_value - min_value)
        return margin_top + plot_height * (1.0 - normalized)

    points = [(scale_x(idx), scale_y(value)) for idx, value in enumerate(values)]
    axis_y = margin_top + plot_height
    labels = [
        f"<text x='{width / 2:.0f}' y='20' text-anchor='middle' font-size='16'>{title}</text>",
        f"<text x='{width / 2:.0f}' y='{height - 6}' text-anchor='middle' font-size='12'>{x_label}</text>",
        f"<text x='14' y='{height / 2:.0f}' text-anchor='middle' font-size='12' transform='rotate(-90 14 {height / 2:.0f})'>{y_label}</text>",
    ]
    ticks = [
        f"<text x='{margin_left - 8}' y='{scale_y(max_value) + 4:.2f}' text-anchor='end' font-size='10'>{max_value:.3f}</text>",
        f"<text x='{margin_left - 8}' y='{scale_y(min_value) + 4:.2f}' text-anchor='end' font-size='10'>{min_value:.3f}</text>",
        f"<text x='{margin_left:.2f}' y='{axis_y + 16:.2f}' text-anchor='middle' font-size='10'>0</text>",
        f"<text x='{margin_left + plot_width:.2f}' y='{axis_y + 16:.2f}' text-anchor='middle' font-size='10'>{len(values) - 1}</text>",
    ]
    circles = "\n".join(
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='2.5' fill='#0b6e4f'/>" for x, y in points
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#fbfbf8"/>
  <line x1="{margin_left}" y1="{axis_y}" x2="{margin_left + plot_width}" y2="{axis_y}" stroke="#333" stroke-width="1"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_y}" stroke="#333" stroke-width="1"/>
  <path d="{_line_path(points)}" fill="none" stroke="#0b6e4f" stroke-width="2"/>
  {circles}
  {' '.join(labels)}
  {' '.join(ticks)}
</svg>
"""
    path.write_text(svg, encoding="ascii")


def write_bar_svg(
    labels: list[str],
    values: list[float],
    path: str | Path,
    *,
    title: str,
    y_label: str,
    width: int = 640,
    height: int = 320,
) -> None:
    path = Path(path)
    if not values:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='320'/>", encoding="ascii")
        return

    margin_left = 56
    margin_right = 24
    margin_top = 32
    margin_bottom = 56
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(max(values), 1e-6)
    bar_width = plot_width / max(len(values), 1)

    rects = []
    texts = []
    for idx, value in enumerate(values):
        scaled_height = plot_height * (value / max_value)
        x = margin_left + idx * bar_width + (bar_width * 0.15)
        y = margin_top + plot_height - scaled_height
        width_px = bar_width * 0.7
        rects.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{width_px:.2f}' height='{scaled_height:.2f}' fill='#c84c09'/>"
        )
        texts.append(
            f"<text x='{x + (width_px / 2):.2f}' y='{height - 22:.2f}' text-anchor='middle' font-size='10'>{labels[idx]}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#fbfbf8"/>
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
  {' '.join(rects)}
  {' '.join(texts)}
  <text x='{width / 2:.0f}' y='20' text-anchor='middle' font-size='16'>{title}</text>
  <text x='14' y='{height / 2:.0f}' text-anchor='middle' font-size='12' transform='rotate(-90 14 {height / 2:.0f})'>{y_label}</text>
  <text x='{margin_left - 8}' y='{margin_top + 4:.2f}' text-anchor='end' font-size='10'>{max_value:.3f}</text>
  <text x='{margin_left - 8}' y='{margin_top + plot_height + 4:.2f}' text-anchor='end' font-size='10'>0.000</text>
</svg>
"""
    path.write_text(svg, encoding="ascii")


def write_multi_line_svg(
    series: list[tuple[str, list[float], str]],
    path: str | Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 640,
    height: int = 320,
) -> None:
    path = Path(path)
    all_values = [value for _, values, _ in series for value in values]
    if not all_values:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='320'/>", encoding="ascii")
        return

    margin_left = 56
    margin_right = 24
    margin_top = 32
    margin_bottom = 48
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    max_len = max(len(values) for _, values, _ in series)

    def scale_x(index: int) -> float:
        if max_len <= 1:
            return margin_left + (plot_width / 2.0)
        return margin_left + (plot_width * index / (max_len - 1))

    def scale_y(value: float) -> float:
        normalized = (value - min_value) / (max_value - min_value)
        return margin_top + plot_height * (1.0 - normalized)

    paths = []
    legend = []
    for idx, (label, values, color) in enumerate(series):
        points = [(scale_x(i), scale_y(value)) for i, value in enumerate(values)]
        paths.append(f"<path d='{_line_path(points)}' fill='none' stroke='{color}' stroke-width='2'/>")
        legend_y = margin_top + 14 + (idx * 16)
        legend.append(f"<rect x='{width - 160}' y='{legend_y - 9}' width='10' height='10' fill='{color}'/>")
        legend.append(f"<text x='{width - 144}' y='{legend_y}' font-size='11'>{label}</text>")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#fbfbf8"/>
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
  {' '.join(paths)}
  {' '.join(legend)}
  <text x='{width / 2:.0f}' y='20' text-anchor='middle' font-size='16'>{title}</text>
  <text x='{width / 2:.0f}' y='{height - 6}' text-anchor='middle' font-size='12'>{x_label}</text>
  <text x='14' y='{height / 2:.0f}' text-anchor='middle' font-size='12' transform='rotate(-90 14 {height / 2:.0f})'>{y_label}</text>
  <text x='{margin_left - 8}' y='{margin_top + 4:.2f}' text-anchor='end' font-size='10'>{max_value:.3f}</text>
  <text x='{margin_left - 8}' y='{margin_top + plot_height + 4:.2f}' text-anchor='end' font-size='10'>{min_value:.3f}</text>
</svg>
"""
    path.write_text(svg, encoding="ascii")


def write_grouped_bar_svg(
    categories: list[str],
    left_values: list[float],
    right_values: list[float],
    path: str | Path,
    *,
    title: str,
    y_label: str,
    left_label: str,
    right_label: str,
    width: int = 640,
    height: int = 320,
) -> None:
    path = Path(path)
    if not categories:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='320'/>", encoding="ascii")
        return

    margin_left = 56
    margin_right = 24
    margin_top = 32
    margin_bottom = 56
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(max(left_values, default=0.0), max(right_values, default=0.0), 1e-6)
    group_width = plot_width / len(categories)

    rects = []
    labels = []
    for idx, category in enumerate(categories):
        base_x = margin_left + idx * group_width
        bar_width = group_width * 0.28
        left_height = plot_height * (left_values[idx] / max_value)
        right_height = plot_height * (right_values[idx] / max_value)
        rects.append(
            f"<rect x='{base_x + group_width * 0.15:.2f}' y='{margin_top + plot_height - left_height:.2f}' width='{bar_width:.2f}' height='{left_height:.2f}' fill='#0b6e4f'/>"
        )
        rects.append(
            f"<rect x='{base_x + group_width * 0.57:.2f}' y='{margin_top + plot_height - right_height:.2f}' width='{bar_width:.2f}' height='{right_height:.2f}' fill='#c84c09'/>"
        )
        labels.append(
            f"<text x='{base_x + group_width / 2:.2f}' y='{height - 22:.2f}' text-anchor='middle' font-size='10'>{category}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#fbfbf8"/>
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>
  {' '.join(rects)}
  {' '.join(labels)}
  <rect x='{width - 170}' y='{margin_top}' width='10' height='10' fill='#0b6e4f'/>
  <text x='{width - 154}' y='{margin_top + 9}' font-size='11'>{left_label}</text>
  <rect x='{width - 170}' y='{margin_top + 16}' width='10' height='10' fill='#c84c09'/>
  <text x='{width - 154}' y='{margin_top + 25}' font-size='11'>{right_label}</text>
  <text x='{width / 2:.0f}' y='20' text-anchor='middle' font-size='16'>{title}</text>
  <text x='14' y='{height / 2:.0f}' text-anchor='middle' font-size='12' transform='rotate(-90 14 {height / 2:.0f})'>{y_label}</text>
  <text x='{margin_left - 8}' y='{margin_top + 4:.2f}' text-anchor='end' font-size='10'>{max_value:.3f}</text>
  <text x='{margin_left - 8}' y='{margin_top + plot_height + 4:.2f}' text-anchor='end' font-size='10'>0.000</text>
</svg>
"""
    path.write_text(svg, encoding="ascii")
