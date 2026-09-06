#!/usr/bin/env python3
"""Build a presentation-oriented TP2 report from completed study phases."""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from sia_tp2.study import read_records, select_conditions, summarize_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT.parent / ".context" / "tp2-colombia-final-study"
DEFAULT_RESULTS = PROJECT_ROOT / "experiments" / "results"
PHASES = (
    "resolution",
    "capacity",
    "selection",
    "crossover",
    "mutation",
    "survival",
    "validation",
    "showcase",
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the TP2 comparative report.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    completed: Dict[str, List[Mapping[str, str]]] = {}
    for phase in PHASES:
        path = args.output / "records" / f"{phase}.csv"
        if path.is_file():
            completed[phase] = list(read_records(path))

    lines = _header(completed)
    for phase in PHASES:
        records = completed.get(phase)
        if not records:
            lines.extend(_pending_section(phase))
            continue
        summaries = summarize_records(records)
        lines.extend(_phase_section(phase, summaries, records, args.results))

    lines.extend(_limitations())
    report = args.results / "COMPARATIVE-REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {report}")
    return 0


def _header(completed: Mapping[str, Sequence[Mapping[str, str]]]) -> List[str]:
    phases = ", ".join(completed) if completed else "ninguna"
    if all(phase in completed for phase in PHASES):
        state = "El protocolo planificado se completó en su totalidad."
    else:
        state = (
            "Las secciones pendientes se mantienen visibles para no presentar "
            "conclusiones antes de tener las cinco semillas por condición."
        )
    return [
        "# TP2 — Informe comparativo de operadores",
        "",
        "## Estado",
        "",
        f"Fases con resultados completos: **{phases}**. {state}",
        "",
        "## Protocolo controlado",
        "",
        "Todas las condiciones usan las semillas `101, 202, 303, 404, 505`. Se compara "
        "dentro del mismo objetivo y presupuesto; los NMSE absolutos de objetivos distintos "
        "no se comparan entre sí. Las curvas usan el mejor histórico.",
    ]


def _phase_section(
    phase: str,
    summaries: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, str]],
    results: Path,
) -> List[str]:
    title = {
        "resolution": "Resolución de trabajo",
        "capacity": "Cantidad de triángulos",
        "selection": "Selección de padres",
        "crossover": "Cruza",
        "mutation": "Mutación",
        "survival": "Supervivencia",
        "validation": "Validación final",
        "showcase": "Demostraciones visuales extendidas",
    }[phase]
    lines = ["", f"## {title}", "", _observation(phase, summaries), ""]
    lines.extend(_summary_table(summaries))
    figure_root = results / "figures" / phase
    for target in sorted({str(row["target"]) for row in summaries}):
        figure = figure_root / f"{target}-curves.svg"
        if figure.is_file():
            lines.extend(["", f"![Curvas de {target}](figures/{phase}/{figure.name})"])
    lines.extend(_representative_images(phase, records, results))
    return lines


def _summary_table(summaries: Sequence[Mapping[str, object]]) -> List[str]:
    lines = [
        "| Objetivo | Condición | NMSE mediano | IQR NMSE | AUC normalizada | Diversidad final | Éxitos 90 % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        iqr = float(item["q3_best_error"]) - float(item["q1_best_error"])
        lines.append(
            "| {target} | {condition} | {error:.6f} | {iqr:.6f} | {auc:.6f} | {diversity:.6f} | {successes}/{runs} |".format(
                target=item["target"],
                condition=item["condition"],
                error=float(item["median_best_error"]),
                iqr=iqr,
                auc=float(item["median_normalized_auc"]),
                diversity=float(item["median_final_diversity"]),
                successes=item["successes_90_percent_reduction"],
                runs=item["runs"],
            )
        )
    return lines


def _representative_images(
    phase: str, records: Sequence[Mapping[str, str]], results: Path
) -> List[str]:
    grouped: Dict[tuple, List[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        grouped[(record["target"], record["condition"])].append(record)
    lines = ["", "### Imágenes representativas", ""]
    for (target, condition), rows in sorted(grouped.items()):
        median = sorted(rows, key=lambda row: float(row["best_error"]))[len(rows) // 2]
        source = Path(median["run_directory"]) / "best.png"
        filename = f"{target}-{condition}-seed-{median['seed']}.png"
        destination = results / "images" / phase / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        lines.append(
            f"- `{target}` — `{condition}`, semilla mediana `{median['seed']}`: "
            f"![resultado](images/{phase}/{filename})"
        )
    return lines


def _observation(phase: str, summaries: Sequence[Mapping[str, object]]) -> str:
    if phase == "validation":
        return (
            "**Resultado:** la configuración elegida se evaluó sobre los tres niveles de "
            "dificultad. Sus NMSE se interpretan dentro de cada objetivo, no como un ranking "
            "directo entre imágenes."
        )
    if phase == "showcase":
        return (
            "**Resultado:** se extendió la semilla mediana de cada objetivo para obtener "
            "evidencia visual representativa; estas tres ejecuciones no forman una nueva "
            "comparación estadística."
        )
    if phase in ("crossover", "mutation", "survival"):
        winner = select_conditions(summaries, phase=phase, count=1)[0]
        return (
            f"**Resultado:** `{winner}` obtuvo el mejor ranking agregado después de ordenar "
            "las condiciones dentro de cada objetivo por calidad, AUC y diversidad."
        )
    winner = min(summaries, key=lambda item: float(item["median_best_error"]))
    if phase == "selection":
        return (
            f"**Resultado:** `{winner['condition']}` obtuvo el menor NMSE mediano "
            f"({float(winner['median_best_error']):.6f}). La interpretación debe contrastarlo "
            "con AUC y diversidad: presión alta no implica necesariamente mejor calidad final."
        )
    return (
        f"**Resultado:** la mejor condición de esta etapa fue `{winner['condition']}` "
        f"(NMSE mediano {float(winner['median_best_error']):.6f})."
    )


def _pending_section(phase: str) -> List[str]:
    labels = {
        "resolution": "Resolución de trabajo",
        "capacity": "Cantidad de triángulos",
        "selection": "Selección de padres",
        "crossover": "Cruza",
        "mutation": "Mutación",
        "survival": "Supervivencia",
        "validation": "Validación final",
        "showcase": "Demostraciones visuales extendidas",
    }
    return ["", f"## {labels[phase]}", "", "**Pendiente:** esta fase aún no terminó sus cinco semillas por condición."]


def _limitations() -> List[str]:
    return [
        "",
        "## Limitaciones y lectura para la exposición",
        "",
        "El NMSE se mide en la resolución de trabajo, por lo que una mejora numérica no "
        "demuestra fidelidad perfecta a tamaño original. Las imágenes representativas son la "
        "semilla mediana, no la mejor: muestran un caso típico. Las conclusiones finales deben "
        "distinguir velocidad (AUC), calidad (NMSE) y riesgo de convergencia prematura "
        "(diversidad).",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
