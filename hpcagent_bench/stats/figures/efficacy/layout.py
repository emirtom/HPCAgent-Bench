# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The efficacy figure's LAYOUT: its config, category positions, labels and the key's columns."""

import dataclasses
import enum
import functools
import itertools
import math
import textwrap
from collections.abc import Sequence

import numpy as np
import pandas as pd
from hpcagent_agent.coded import Coded
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.typing import LineStyleType

from hpcagent_bench import packets, study_tags
from hpcagent_bench.stats import cost as cost_models
from hpcagent_bench.stats import palette, population, rules, style, summary
from hpcagent_bench.stats.figures.efficacy.reduce import (
    COST_SIG_MARK,
    EMPTY_POINT,
    NO_SIGNIFICANCE,
    SPEEDUP_OVER,
    UNCHANGED_POINT,
    Series,
    SetupPoint,
    Significance,
    absolute_columns,
    axis_significance,
    drawn_symbols,
    setup_groups,
    setup_point,
    skill_halves,
    solved_flags,
    speedup_mask,
)
from hpcagent_bench.study_tags import Marker

__all__ = [
    "BASELINE_BAND_FRACTION",
    "BASELINE_BAND_TYPE",
    "BASELINE_HEADROOM_FRACTION",
    "CATEGORY_TICK_PT",
    "CHAR_EM",
    "CONTROL_MARKER",
    "Comparator",
    "DEFAULT_CONFIG",
    "DOT_ROW_HEIGHT_IN",
    "DifferenceKey",
    "DotColumn",
    "EXTRA_MARKERS",
    "FEW_KERNELS_NOTE",
    "FigureConfig",
    "GROUPED_BY_DELIVERY",
    "GROUP_STEP",
    "LABEL_CHARS_PER_PT",
    "LINE_BAND",
    "MAX_X_TICKS",
    "Measure",
    "Mode",
    "MEASURES",
    "MEASURE_HEIGHT",
    "MEASURE_LABELS",
    "MEASURE_PAD_IN",
    "MIN_COLUMN_CATEGORIES",
    "NAME_CHAR_EM",
    "NAME_CLEARANCE_IN",
    "NARROW_FIGURE_IN",
    "PAIRED_COLUMNS",
    "PANEL_LETTERS",
    "PANEL_NAME_GID",
    "PAPER_CONFIG",
    "PER_COLUMN_TREATMENTS",
    "Panel",
    "RELATIVE_LABELS",
    "SERVED_SPEEDUP_LABEL",
    "SHORT_NAMES",
    "SHORT_ROW_IN",
    "SHORT_ROW_TOKEN_SUBS",
    "SNAP_PAD",
    "SNAP_REACH",
    "STAGGER_TICK_SHARE",
    "SUCCESS_HEADROOM",
    "SUCCESS_TICKS",
    "SetupRow",
    "TICK_ALIASES",
    "TICK_LABEL_CLEARANCE_IN",
    "TITLE_ACRONYMS",
    "TITLE_SMALL_WORDS",
    "TREATMENT_NAMES",
    "VERDICT_MARKS",
    "alias_footnotes",
    "baseline_text",
    "category_tick",
    "column_limits",
    "column_order",
    "column_shape",
    "column_treatment_name",
    "column_treatment_shape",
    "column_x",
    "comma_list",
    "comparator_point",
    "comparator_rows",
    "comparator_shapes",
    "comparator_short_name",
    "comparator_table",
    "comparators_from_table",
    "cost_axis_label",
    "cost_card_name",
    "cost_row_label",
    "delivery_markers",
    "delivery_order",
    "difference_factor",
    "difference_middle",
    "dot_columns",
    "dot_row_grid",
    "dot_row_heights",
    "dot_row_widths",
    "first_per_label",
    "flat_treatments",
    "fold_width",
    "group_rules",
    "interval_bounds",
    "interval_kernels",
    "is_pending",
    "key_columns",
    "key_filler",
    "label_wrap",
    "leg_centres",
    "leg_rank",
    "leg_runs",
    "measure_row_config",
    "measure_value",
    "mode_labels",
    "name_layout",
    "name_line_width",
    "name_spans",
    "near_ticks",
    "nth",
    "nth_list",
    "paired_kernels",
    "pairs_table",
    "panel_name_wrap",
    "panel_rows",
    "panel_tag",
    "parse_differences",
    "pending_rows",
    "per_column_rows",
    "placed_spans",
    "placeholder_rows",
    "reduce_pair",
    "relative_point",
    "relative_rows",
    "retyped",
    "row_label",
    "row_label_fit",
    "row_verdict",
    "setup_points",
    "setup_rows",
    "speedup_row_label",
    "success_rate",
    "text_band",
    "tick_alias",
    "title_case",
    "treatment_marker",
    "two_line_fold",
    "verdicts_of",
    "with_comparators",
    "wrapped_label",
    "x_tick_step",
]


class Mode(Coded):
    """Where a mark sits (:attr:`FigureConfig.mode`), spelled as the ``--mode`` argument:
    :attr:`ABSOLUTE` against the experiment baseline, :attr:`RELATIVE` as the change over the LLM's own
    no-packet control."""

    ABSOLUTE = enum.auto()
    RELATIVE = enum.auto()


class Measure(enum.Enum):
    """One stacked row of a dot-row figure: what each setup REACHED over the experiment baseline
    (``SPEEDUP``), how many of its kernels it got RIGHT (``SUCCESS``), and what it SPENT (``COST``)."""

    SPEEDUP = enum.auto()
    SUCCESS = enum.auto()
    COST = enum.auto()


@dataclasses.dataclass(frozen=True, slots=True)
class FigureConfig:
    """Every knob a reader of this figure might want to change by hand, in one place. Pass a
    replacement instance to a drawing function's ``config`` argument rather than editing a constant
    inside it. The row width stays the caller's (``row_width_in``/``--row-width``).
    """

    #: Every type size: ticks and category names (``tick_pt``), axis labels (``label_pt``), panel
    #: names (``title_pt``), point notes and superscripts (``annotation_pt``) and the key
    #: (``legend_pt``). One of :mod:`~hpcagent_bench.stats.style`'s scales, so two figures of one
    #: page print at one size.
    type_: style.TypeScale = style.AUTHOR_SCALE
    #: Where a mark sits (:class:`Mode`); relative mode prints the control's own level in a baseline band
    #: (:func:`draw_baseline_band`): speedup as a geomean of paired ratios, tokens as a ratio of arithmetic means.
    mode: Mode = Mode.ABSOLUTE
    #: A legend SWATCH's own size, points, and how far matplotlib scales it past that. At paper
    #: type a swatch drawn for authoring scale is taller than the row it sits in and the rows
    #: collide.
    legend_marker_pt: float = 9.0
    legend_marker_scale: float = 1.4
    #: The legend's column count ceiling (:func:`~hpcagent_bench.stats.style.legend_below` wraps a
    #: row that does not fit the canvas onto fewer columns, never more than this).
    legend_ncol: int = 4
    #: Draw a "?" in a category that holds no measurement yet (a ``pending=`` model, a
    #: ``placeholders=`` delivery), so a comparison still running reads as pending, not as absent.
    mark_pending: bool = False
    #: A summary mark's own size (points^2, matplotlib's ``s=``).
    mark_size: float = 74.0
    #: How far a point's label sits from its mark, in points. SMALL: a label further from its mark
    #: than from its neighbour's is a label a reader has to guess the owner of.
    label_offset_pt: float = 9.0
    #: How far a bare significance superscript sits from ITS mark, in points. Smaller than a
    #: label's: a lone ``*`` has to read as belonging to the mark beside it, and at the label's own
    #: offset it floated between two columns.
    symbol_offset_pt: float = 4.0
    #: The TOKEN-COST interval's line style. Dashed, so the two axes' intervals cannot be read as
    #: one quantity: they are a speedup and a spend, on their own scales (SC15 Rule 4). The
    #: speedup interval stays solid.
    cost_linestyle: LineStyleType = "--"
    #: Most labelled ticks an axis may carry. Raising it thins the spacing between whole ratios.
    max_ticks: int = 13
    #: Where a labelled tick lands within each decade of a TOKEN axis. A count axis spends most of
    #: an experiment inside one decade, so 1-2-5 leaves it three ticks -- but a short row cannot carry
    #: six either, and a caller sizing one replaces this.
    token_subs: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    #: Fractional padding ``ax.margins`` adds around the plotted extent on each axis -- SMALL: the
    #: panel fits the data and its 95% intervals plus a little air, not a fixed wide window.
    margin: float = 0.15
    #: The narrowest total span EITHER axis is ever drawn at, in octaves (log2 units): a panel whose
    #: every setup moved a kernel by a few percent still gets more than the one tick a degenerate
    #: sub-octave window would leave (:func:`~hpcagent_bench.stats.style.value_axis`'s own log2
    #: branch). Small on purpose -- large enough for >=2 ticks, not so large it reopens the "huge
    #: empty area" a wider floor left around a tightly clustered result.
    min_span: float = 1.0
    #: The minor grid's own line weight and colour (:func:`minor_grid`), lighter than the major grid
    #: (:data:`~hpcagent_bench.stats.style.RULE`) so it reads as a finer ruling under the marks, not
    #: a second reference. Default to :mod:`~hpcagent_bench.stats.style`'s, the one source.
    minor_grid_width: float = style.MINOR_GRID_WIDTH
    minor_grid_color: str = style.MINOR_RULE
    #: An interval's own line weight, the major grid's, and the panel frame's. Separate knobs
    #: because a figure drawn at its FINAL printed size needs all three thinner: a 1.2pt whisker
    #: that reads as a line at authoring scale reproduces as a bar at 8pt type.
    interval_width: float = 1.2
    #: Half the width of the cap on a capped interval bar, points.
    interval_cap_pt: float = 2.0
    #: How far past the outermost MARK of a dot-row panel an interval is drawn, as a factor (4 = two
    #: octaves). A few-kernel interval reaching 0.004x stretched its panel over twenty octaves, the
    #: ticks read 0.00391x and every mark sat in a sliver; the interval is cut at this reach instead,
    #: with an arrowhead where it continues (:func:`draw_interval`).
    interval_reach: float = 4.0
    #: The fewest kernels a dot-row mark's interval is drawn from. Three kernels put the 95% log-t
    #: critical value at 4.3 and the interval over three decades, cut at both ends; below this the
    #: mark stands alone and the key says why (:data:`FEW_KERNELS_NOTE`).
    min_interval_kernels: int = summary.MIN_PAIRS_FOR_INTERVAL
    grid_width: float = 0.7
    spine_width: float = 0.8
    #: Join a setup to its own no-packet twin with a faint segment (:func:`draw_measure_row`). OFF: at
    #: paper scale the segment reads as a third mark between the two;
    #: :func:`draw_difference_arrow` draws the displacement, labelled, on request.
    link_pairs: bool = False
    link_width: float = 0.9
    link_alpha: float = 0.35
    #: How far either side of its category's own position each of the two setups is drawn, in
    #: category units. Half of it is the gap between the pair; 0.5 would put one pair's mark on top
    #: of the next pair's.
    dodge: float = 0.16
    #: The gap between the two measure rows, as a fraction of a row's own height.
    row_gap: float = 0.2
    #: The gap between two columns of a stacked row, as a fraction of a column's own width. Wide
    #: enough that each column reads as its own panel, since they share no Y scale.
    column_gap: float = 0.3
    #: What the WIDEST column keeps of its share. A column with three times the categories does not
    #: need three times the width -- its marks are already the densest on the page -- and what it
    #: gives back is width the narrow columns have nothing else to take from.
    wide_column_scale: float = 0.85
    #: The chrome LEFT of the data box, in inches: the first estimate :func:`figure_dot_row` lays
    #: out with before it measures the Y labels and takes exactly what they need.
    left_chrome_in: float = 1.0
    #: The band BELOW the category names the key is fitted into, in inches (:func:`fit_legend`).
    legend_chrome_in: float = 0.48
    #: How far :func:`fit_legend` shrinks the key's type, as a fraction of its ``legend_pt``, before
    #: it grows the band instead.
    legend_min_scale: float = 0.7
    #: How far :func:`stagger_crowded_ticks` shrinks the category names, as a fraction of
    #: the tick size, when two staggered lines still leave them touching.
    category_min_scale: float = 0.7
    #: The most lines a panel's name, or a rotated axis label, may fold onto. A third line comes out
    #: of the panel.
    max_name_lines: int = 2
    #: The widest a delivery's tick text runs before it folds.
    tick_wrap: int = 8
    #: The category names' size against the tick size: under nine columns of a text-width row the
    #: names are the densest text on the figure (20% smaller than the ticks).
    category_scale: float = 1.0
    #: Put the key's text-only notes in the key: abbreviations, the few-kernels note, and one row per
    #: significance superscript. A paper figure turns it off: the notes go to the caption and the two
    #: superscripts share one row, so the key fits four columns.
    key_notes: bool = True
    #: Set the key with :data:`~hpcagent_bench.stats.style.COMPACT_KEY` spacing, so four columns of
    #: a paper figure fit the plot body.
    compact_key: bool = False


#: The default a caller draws with unless it hands a replacement in: a figure AUTHORED large and
#: reproduced at roughly half its width (:data:`~hpcagent_bench.stats.style.AUTHOR_SCALE`).
DEFAULT_CONFIG = FigureConfig()

#: For a figure drawn at the size it will be PRINTED at: a row of panels budgeted to a paper's own
#: text width, which goes into the page at scale 1.0 and is never shrunk.
#:
#: Figure size and type size are one decision, not two: the type is
#: :data:`~hpcagent_bench.stats.style.PRINT_SCALE`, and the marks and rules follow it thinner.
PAPER_CONFIG = dataclasses.replace(
    DEFAULT_CONFIG,
    type_=style.PRINT_SCALE,
    legend_ncol=5,
    key_notes=False,
    compact_key=True,
    # 19% over the earlier 0.8: the category names read small beside the marks.
    category_scale=0.95,
    legend_min_scale=1.0,
    category_min_scale=style.PRINT_MIN_PT / style.PRINT_SCALE.tick_pt,
    legend_marker_pt=5.5,
    legend_marker_scale=1.0,
    # 15% under the earlier 19 pt^2: neighbouring marks of one delivery overlapped.
    mark_size=13.3,
    label_offset_pt=6.0,
    symbol_offset_pt=3.0,
    interval_width=0.9,
    interval_cap_pt=1.5,
    grid_width=0.5,
    spine_width=0.5,
)


def retyped(config: FigureConfig, **sizes: float) -> FigureConfig:
    """``config`` with the named :attr:`FigureConfig.type_` sizes replaced."""
    return dataclasses.replace(config, type_=dataclasses.replace(config.type_, **sizes))


#: Columns of :func:`paired_kernels`' frame: everything a drawn interval or a significance test on
#: one comparison needs, and the raw costs Rule 4 requires travel with it.
PAIRED_COLUMNS: tuple[str, ...] = (
    "control_speedup",
    "treated_speedup",
    "control_tokens",
    "treated_tokens",
    "baseline_ns",
    "native_ns",
    "delivered",
    "control_solved",
    "treated_solved",
)


def paired_kernels(
    control: pd.DataFrame,
    treated: pd.DataFrame,
    card: cost_models.CostModel | None = None,
) -> pd.DataFrame:
    """One row per kernel BOTH sides cover on speedup; its tokens are NaN where either side has no
    task total.

    The speedup leg is paired over every such kernel, the same population ``paired_setups.py``'s
    score leg (and so the family's corrected test) is taken over; the token leg over the subset
    with a total on both sides (:func:`reduce_pair`). Intersecting the two would move the speedup
    coordinate off the table's value whenever a token record is missing. ``delivered`` is True only
    when BOTH sides verified an answer there; a kernel either side only served
    (:data:`~hpcagent_bench.stats.population.NOT_DELIVERED`) is a placeholder ratio, not a
    measurement.

    Tokens are priced with ``card`` (:func:`~hpcagent_bench.stats.cost.priced`), the ``billed`` card
    by default: a frame handed in unpriced carries effective tokens, and the cost row is labelled
    with the card. Pricing is idempotent, so a frame the caller already priced with ``card`` is
    unchanged.
    """
    card = card or cost_models.resolve()
    control, treated = cost_models.priced(control, card), cost_models.priced(treated, card)
    control_answers = population.kernel_answers(control)
    treated_answers = population.kernel_answers(treated)
    control_tokens = population.kernel_tokens(control)
    treated_tokens = population.kernel_tokens(treated)
    kernels = control_answers.index.intersection(treated_answers.index)
    if len(kernels) == 0:
        return pd.DataFrame(columns=PAIRED_COLUMNS)
    has_delivered = population.DELIVERED_COLUMN in control_answers and population.DELIVERED_COLUMN in treated_answers
    frame = pd.DataFrame(
        {
            "control_speedup": control_answers.loc[kernels, "speedup"].astype(float),
            "treated_speedup": treated_answers.loc[kernels, "speedup"].astype(float),
            "control_tokens": control_tokens.reindex(kernels).astype(float),
            "treated_tokens": treated_tokens.reindex(kernels).astype(float),
            "baseline_ns": control_answers.loc[kernels, "baseline_ns"].astype(float)
            if "baseline_ns" in control_answers
            else math.nan,
            "native_ns": control_answers.loc[kernels, "native_ns"].astype(float)
            if "native_ns" in control_answers
            else math.nan,
            "delivered": (
                control_answers.loc[kernels, population.DELIVERED_COLUMN].astype(bool)
                & treated_answers.loc[kernels, population.DELIVERED_COLUMN].astype(bool)
            )
            if has_delivered
            else True,
            "control_solved": solved_flags(control_answers, kernels),
            "treated_solved": solved_flags(treated_answers, kernels),
        },
        index=kernels,
    )
    usable = (
        np.isfinite(frame.control_speedup)
        & (frame.control_speedup > 0.0)
        & np.isfinite(frame.treated_speedup)
        & (frame.treated_speedup > 0.0)
    )
    return frame.loc[usable]


def reduce_pair(
    control: pd.DataFrame,
    treated: pd.DataFrame,
    over: population.KernelPolicy = SPEEDUP_OVER,
    card: cost_models.CostModel | None = None,
) -> Series | None:
    """``(control, treated)`` as a :class:`Series`; ``None`` when they share no usable kernel or no
    kernel has a token total on both sides.

    ``x`` is over every paired kernel, ``y`` over the ones with both token totals
    (:func:`paired_kernels`); ``token_kernels`` says how many that is.
    """
    paired = paired_kernels(control, treated, card)
    if paired.empty:
        return None
    score_ratio = (paired.treated_speedup / paired.control_speedup).to_numpy(dtype=float)
    cost_ratio = (paired.treated_tokens / paired.control_tokens).to_numpy(dtype=float)
    priced = np.isfinite(cost_ratio) & (cost_ratio > 0.0)
    timed = speedup_mask(paired, over)
    if not priced.any() or not timed.any():
        return None
    score = summary.geomean_ci(score_ratio[timed])
    # A cost change is a ratio of arithmetic means (SC15: costs summarize with the arithmetic mean).
    cost = summary.mean_ratio_ci(
        paired.treated_tokens.to_numpy(dtype=float)[priced], paired.control_tokens.to_numpy(dtype=float)[priced]
    )
    return Series(
        x=summary.log2_change(score.point),
        x_low=summary.log2_change(score.low),
        x_high=summary.log2_change(score.high),
        y=cost.point,
        y_low=cost.low,
        y_high=cost.high,
        kernels=int(timed.sum()),
        delivered=int(paired.delivered.sum()),
        baseline_ns=float(paired.baseline_ns.median()),
        native_ns=float(paired.native_ns.median()),
        control_tokens=float(paired.control_tokens[priced].median()),
        treated_tokens=float(paired.treated_tokens[priced].median()),
        token_kernels=int(priced.sum()),
    )


def setup_points(
    control: pd.DataFrame,
    treated: pd.DataFrame,
    over: population.KernelPolicy = SPEEDUP_OVER,
    card: cost_models.CostModel | None = None,
) -> tuple[SetupPoint, SetupPoint] | None:
    """``(control, treated)`` as the two marks a dot-row column draws: where each setup sits against
    the STUDY BASELINE, not where one sits against the other.

    Both are taken over the kernels the two setups SHARE (:func:`paired_kernels`), so the pair is
    comparable and the displacement between the two speedups is EXACTLY :func:`reduce_pair`'s
    ``x`` -- a geomean of ratios is the ratio of the geomeans. That is the reading "HIP reached
    3.2x" needs and a ratio alone cannot give.
    """
    paired = paired_kernels(control, treated, card)
    if paired.empty:
        return None
    control_tokens = paired.control_tokens.to_numpy(dtype=float)
    treated_tokens = paired.treated_tokens.to_numpy(dtype=float)
    priced = np.isfinite(control_tokens) & (control_tokens > 0.0) & np.isfinite(treated_tokens) & (treated_tokens > 0.0)
    if not priced.any():
        return None
    timed = speedup_mask(paired, over)
    return (
        setup_point(paired.control_speedup, paired.control_tokens, priced, timed, paired.control_solved),
        setup_point(paired.treated_speedup, paired.treated_tokens, priced, timed, paired.treated_solved),
    )


#: The one shape a CONTROL mark ever wears, hollow: the palette's, so every figure agrees. A hollow
#: copy of the treatment's own shape reads as "the same thing, lighter" at print size; a different
#: outline reads as a different thing, which is what it is. No packet is ever given this shape
#: (:func:`treatment_marker`).
CONTROL_MARKER: str = palette.CONTROL_MARKER


def treatment_marker(treatment: str) -> Marker:
    """The packet's own shape, never :data:`CONTROL_MARKER`: a treatment drawn as a filled circle
    beside a hollow one is the ambiguity the control shape exists to remove."""
    shape = palette.packet_marker(treatment)
    if shape != CONTROL_MARKER:
        return shape
    shapes = delivery_markers()
    return shapes[(shapes.index(shape) + 1) % len(shapes)] if shape in shapes else "s"


#: Key text of a panel whose treated side is not one packet: a harness comparison's columns each name
#: their own harness, so its one treated shape is "the other harness", whichever the column says.
TREATMENT_NAMES: dict[str, str] = {"harness": "Other Harness (Column)"}

#: Panel treatments whose columns each change a DIFFERENT registered treatment (a harness, or a
#: packet on the control's harness): every column wears that treatment's own registered shape.
PER_COLUMN_TREATMENTS: frozenset[str] = frozenset({"harness", "packets"})

#: Of those, the panels whose columns sit under their DELIVERY's tick ("C-CPF" under "C"): the packet
#: is an intervention, told by shape and key, not a category of its own on the axis.
GROUPED_BY_DELIVERY: frozenset[str] = frozenset({"packets"})


def column_treatment_shape(leg: str) -> Marker:
    """The registered shape of the treatment a column is named after: a harness by its display name,
    else a packet by its display name; "" when neither registry names it."""
    for harness in study_tags.order("harnesses"):
        if harness and study_tags.harness_name(harness) == leg:
            return palette.harness_marker(harness)
    suffix = leg.rsplit("-", 1)[-1]
    for packet in study_tags.order("packets"):
        if packet and leg in (study_tags.packet_name(packet), study_tags.packet_short_name(packet)):
            return palette.packet_marker(packet)
    for packet in study_tags.order("packets"):
        if packet and suffix == study_tags.packet_short_name(packet):
            return palette.packet_marker(packet)
    return ""


def column_treatment_name(leg: str) -> str:
    """The key text of a per-column treatment: a "<delivery>-<packet>" column's packet, else the leg."""
    suffix = leg.rsplit("-", 1)[-1]
    for packet in study_tags.order("packets"):
        if packet and suffix == study_tags.packet_short_name(packet):
            return study_tags.packet_name(packet)
    return leg


#: A key column's filler: an empty row that keeps the next group at the top of its own column.
def key_filler() -> Line2D:
    """One invisible, unlabelled key row."""
    return Line2D([], [], linestyle="none", marker="none", label=" ")


def key_columns(groups: Sequence[Sequence[Line2D]], ncol: int) -> tuple[list[Line2D], int]:
    """The key laid out BY MEANING, one group per column (who drew it, what was compared, how it is
    summarized, what is significant), as matplotlib's column-major fill reads it: each group padded with
    :func:`key_filler` rows to the tallest one. More groups than ``ncol`` columns merge the two shortest
    neighbours until they fit. Returns the flat handle list and its column count."""
    columns = [list(group) for group in groups if group]
    if sum(len(column) for column in columns) <= max(1, ncol):
        # The whole key fits ONE row: one entry per column, no grouping to show.
        columns = [[handle] for column in columns for handle in column]
    while len(columns) > max(1, ncol):
        sizes = [len(a) + len(b) for a, b in itertools.pairwise(columns)]
        merge = sizes.index(min(sizes))
        columns[merge : merge + 2] = [columns[merge] + columns[merge + 1]]
    height = max((len(column) for column in columns), default=0)
    for column in columns:
        for handle in column:
            handle.set_label(title_case(str(handle.get_label())))
    flat = [handle for column in columns for handle in [*column, *[key_filler()] * (height - len(column))]]
    return flat, len(columns)


#: The most labelled ticks the X axis draws before its whole-ratio spacing widens. A few outlier
#: kernels (one crashed to 1/512x, another ran away to 256x) autoscale the window past twenty
#: octaves, and :data:`MultipleLocator(1.0)` -- a tick at EVERY power of 2 -- smears that many labels
#: into one panel's width until they overlap into a solid bar.
MAX_X_TICKS: int = 9


def x_tick_step(span: float, max_ticks: int = MAX_X_TICKS) -> int:
    """The whole-ratio spacing (in log2 units: 1 is every power of 2, 2 every power of 4, ...) that
    keeps the X axis under :data:`MAX_X_TICKS` labelled ticks for a window ``span`` wide. Doubled
    rather than picked from an arbitrary "nice number" table, so a tick always lands on an INTEGER
    log2 value -- the only kind :func:`~hpcagent_bench.stats.style.log2_ratio_tick` spells as a clean ratio."""
    step = 1
    while span / step > max(max_ticks - 1, 1):
        step *= 2
    return step


def cost_card_name(name: str) -> str:
    """A cost card's KEY as a label word: ``billed`` -> ``Billed`` (:data:`~hpcagent_bench.stats.
    style.INK`'s own module docstring carries this repo's Title Case rule for figure text)."""
    return str(name).replace("-", " ").replace("_", " ").title()


def cost_row_label(name: str) -> str:
    """The one-line token-cost Y title of a dot row: the card's name (``billed`` -> ``Billed Tokens``).
    The weight vector goes into the caption; spelled on the axis it folds the title onto three lines."""
    card = cost_card_name(name)
    return f"{card} Tokens" if card else "Tokens"


def cost_axis_label(card: cost_models.CostModel) -> str:
    """The token-cost Y title of a panel figure (per-kernel, temperature): the card's name over its (fresh input,
    cached input, output) weights; no-break spaces keep each line whole when the axis folds a tall label."""
    weights = f"({card.fresh_input:g}, {card.cached_input:g}, {card.output:g})"
    return f"{cost_row_label(card.key)}\n{weights}".replace(" ", "\N{NO-BREAK SPACE}")


@functools.lru_cache(maxsize=1, typed=True)
def delivery_order() -> tuple[str, ...]:
    """Every delivery a figure can draw, in SHAPE-assignment order: the registry's languages in
    their own order, then the offload delivery, which is a device plus a language
    (:data:`~hpcagent_bench.study_tags.OFFLOAD_DELIVERY_NAME`) and so has no tag of its own.

    Keyed on the DISPLAY name rather than the language tag on purpose: an offload setup's tag is
    plain ``c``, and sharing C's shape would give a joined row's CPU panel and its GPU panel the
    same mark for two different things.
    """
    names = study_tags.names("languages")
    ordered = [names[tag] for tag in study_tags.order("languages") if tag in names]
    return (*ordered, study_tags.OFFLOAD_DELIVERY_NAME)


#: Shapes for deliveries past the registry's own eight-marker table. Appended HERE rather than to
#: the registry because :func:`~hpcagent_bench.stats.palette.marker` hands standalone optimizers
#: shapes from the BACK of that table -- growing it would repaint every DaCe and CPF mark already
#: in the paper. Nine deliveries against eight markers is what made "OpenMP Offload" and "C" the
#: same circle.
EXTRA_MARKERS: tuple[str, ...] = ("<", ">", "p", "h", "8")


@functools.lru_cache(maxsize=1, typed=True)
def delivery_markers() -> tuple[Marker, ...]:
    """The shape table deliveries draw from: the registry's, then :data:`EXTRA_MARKERS`."""
    return (*palette.markers(), *EXTRA_MARKERS)


#: Clearance added past a measurement, inches -- the same margin :func:`~hpcagent_bench.stats.style.
#: title` and :func:`~hpcagent_bench.stats.style.legend_below` leave past their own measured boxes.
MEASURE_PAD_IN: float = 0.08


#: One character of panel name, as a fraction of the type's own point size. The sans face this
#: repo sets averages a little over half an em across mixed case; measured rather than assumed
#: would need a renderer, and the fold only has to be close.
NAME_CHAR_EM: float = 0.52


def panel_name_wrap(side: float, points: float, em: float = NAME_CHAR_EM) -> int:
    """How many characters of a ``points``-sized name fit a span ``side`` inches wide."""
    return max(4, int(side * style.POINTS_PER_INCH / (points * em)))


#: The gid a panel's own name is drawn under, so a later pass can find it, measure it and replace
#: it. It is an annotation rather than a title because a left-aligned title is not the artist
#: ``ax.title`` returns -- measuring that one measures the empty centre title instead.
PANEL_NAME_GID: str = "efficacy-panel-name"


#: One panel of a joined row: ``title`` (what the panel is CALLED) plus either the SINGLE-treatment
#: shape (``treatment: str``, ``stats``/``frame`` each one table) or the MULTI-treatment one
#: (``treatments: Sequence[str]``, ``stats``/``frame`` each a ``{treatment: table}`` dict).
Panel = (
    tuple[str, str, pd.DataFrame, pd.DataFrame]
    | tuple[str, Sequence[str], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
)


def flat_treatments(spec: str | Sequence[str]) -> list[str]:
    """``spec`` as a flat list, whether it names one treatment or several."""
    return [spec] if isinstance(spec, str) else list(spec)


#: The measures a dot-row figure stacks, top to bottom: what each setup REACHED over the experiment
#: baseline, how many of its kernels it got RIGHT, and what it SPENT. One row each, over one shared
#: categorical X.
MEASURES: tuple[Measure, ...] = tuple(Measure)

#: Each measure's default axis label.
MEASURE_LABELS: dict[Measure, str] = {
    Measure.SPEEDUP: "Speedup",
    Measure.SUCCESS: "Solved Rate",
    Measure.COST: "Token Cost",
}

#: Each measure's row height as a fraction of ``row_height_in``. A count out of N needs no ladder of
#: ratios, so the success row is the shortest; the speedup and cost rows are 0.7 of one and the
#: success row 0.45 (15% and 10% below the earlier 0.82 and 0.5); the speedup row
#: 25% taller, 0.875 (speedup differences were hard to see).
MEASURE_HEIGHT: dict[Measure, float] = {Measure.SPEEDUP: 0.875, Measure.SUCCESS: 0.45, Measure.COST: 0.7}

#: The height ``MEASURE_HEIGHT`` is a fraction of, inches: a dot row placed at a paper's text width.
DOT_ROW_HEIGHT_IN: float = 0.98

#: Headroom above N on the success row, as a fraction of N, so the dashed ceiling at N is not the frame.
SUCCESS_HEADROOM: float = 0.05

#: The speedup row's label when failures enter at 1x instead of being left out.
SERVED_SPEEDUP_LABEL: str = "Speedup (1x Fallback)"


def speedup_row_label(over: population.KernelPolicy) -> str:
    """The speedup row's Y label under ``over``."""
    return SERVED_SPEEDUP_LABEL if over == population.KernelPolicy.SERVED else MEASURE_LABELS[Measure.SPEEDUP]


@dataclasses.dataclass(frozen=True, slots=True)
class SetupRow:
    """One CATEGORY of a dot-row figure: an (LLM, delivery) pair and its two setups."""

    model: str
    leg: str
    colour: str
    control: SetupPoint
    treated: SetupPoint
    #: The treated mark's own shape where a column's treatment is not the panel's one packet (a
    #: harness comparison: each column a different harness); "" wears the panel's shape.
    shape: Marker = ""
    #: The axis group the column sits in when that is not its leg: a several-packet panel's
    #: "C-CPF" column sits under the "C" tick, its packet told by its shape and the key.
    group: str = ""
    #: A compiler or framework COMPARATOR (:class:`Comparator`), not an LLM pair: ``model`` is its
    #: comparator key, ``treated`` its one mark, ``control`` empty. Drawn on the speedup and solved
    #: rows only, after the models of its group.
    comparator: bool = False
    #: Relative mode only: the control's ABSOLUTE point (its speedup over the experiment baseline, its mean tokens
    #: per kernel), which the baseline band prints; ``control`` is then the 1x reference and ``treated`` the change.
    baseline: SetupPoint | None = None

    @property
    def axis_group(self) -> str:
        """The tick this column sits under: :attr:`group`, else its leg."""
        return self.group or self.leg

    @property
    def label(self) -> str:
        """The category's own tick text: what it DELIVERED, or the MODEL where the comparison has
        no delivery of its own (gitscicomp10's repository against the bare kernel). The model is
        otherwise drawn once per run of columns instead (:func:`draw_category_axis`) -- spelled on
        every column, "Qwen3.8-27B" three times over collides with itself long before nine
        categories."""
        return self.leg or study_tags.model_name(self.model)


def setup_rows(
    frame: pd.DataFrame,
    over: population.KernelPolicy = SPEEDUP_OVER,
    card: cost_models.CostModel | None = None,
) -> list[SetupRow]:
    """Every (model, leg) of ``frame`` as a category, in the order the categorical axis draws them.

    Sorted by DELIVERY first, then model: one tick names a language once, its
    models stand side by side in their colours, and a light rule separates the languages
    (:func:`column_x`, :func:`leg_runs`). The model is the colour; repeating "C, Fortran" under every
    model named the same thing three times.
    """
    rows: list[SetupRow] = []
    for model, leg, pair in setup_groups(frame):
        points = setup_points(*skill_halves(pair), over, card)
        if points is None:
            continue
        rows.append(SetupRow(model, leg, palette.model_color(model), points[0], points[1]))
    return column_order(rows)


def relative_point(
    control: pd.DataFrame,
    treated: pd.DataFrame,
    over: population.KernelPolicy = SPEEDUP_OVER,
    card: cost_models.CostModel | None = None,
) -> SetupPoint | None:
    """The treated setup's CHANGE over its own control, both in ``log2(ratio)``: ``x`` the geomean of the paired
    per-kernel speedup ratios, ``y`` the ratio of the arithmetic means of their token spend (SC15: costs summarize
    with the arithmetic mean), each with its 95% bootstrap interval; ``solved``/``served`` the treated setup's.
    The same kernels as :func:`setup_points`. None when the pair shares no usable kernel."""
    pair = reduce_pair(control, treated, over, card)
    points = setup_points(control, treated, over, card)
    if pair is None or points is None:
        return None
    return SetupPoint(
        x=pair.x, x_low=pair.x_low, x_high=pair.x_high,
        y=math.log2(pair.y), y_low=math.log2(pair.y_low), y_high=math.log2(pair.y_high),
        kernels=pair.kernels, token_kernels=pair.token_kernels, solved=points[1].solved, served=points[1].served,
    )  # fmt: skip


def relative_rows(
    frame: pd.DataFrame,
    over: population.KernelPolicy = SPEEDUP_OVER,
    card: cost_models.CostModel | None = None,
) -> list[SetupRow]:
    """:func:`setup_rows` for :attr:`Mode.RELATIVE`: each LLM's control is its own 1x (:data:`UNCHANGED_POINT`, keeping
    its solved count), the treated point its change over it (:func:`relative_point`), and :attr:`SetupRow.baseline`
    the control's absolute point for the baseline band."""
    rows: list[SetupRow] = []
    for model, leg, pair in setup_groups(frame):
        control, treated = skill_halves(pair)
        points = setup_points(control, treated, over, card)
        change = relative_point(control, treated, over, card)
        if points is None or change is None:
            continue
        unchanged = dataclasses.replace(UNCHANGED_POINT, solved=points[0].solved, served=points[0].served)
        rows.append(SetupRow(model, leg, palette.model_color(model), unchanged, change, baseline=points[0]))
    return column_order(rows)


def leg_rank(leg: str) -> tuple[int, str]:
    """A delivery's place on the axis: :func:`delivery_order`, unregistered ones last, alphabetical."""
    order = delivery_order()
    return (order.index(leg), "") if leg in order else (len(order), leg)


def column_order(rows: Sequence[SetupRow]) -> list[SetupRow]:
    """``rows`` in axis order: delivery first (:func:`leg_rank`), its comparators after its models,
    then model in registry order."""
    order = {name: index for index, name in enumerate(palette.in_order([row.model for row in rows]))}
    return sorted(
        rows, key=lambda row: (leg_rank(row.axis_group), row.comparator, row.leg, order.get(row.model, len(order)))
    )


#: The spacing of two columns of ONE delivery (its models side by side), against 1.0 between two
#: deliveries: the models of a language read as one group.
GROUP_STEP: float = 0.6


def column_x(rows: Sequence[SetupRow]) -> list[float]:
    """Each column's x: :data:`GROUP_STEP` apart within a delivery, a whole step between deliveries."""
    xs: list[float] = []
    for index, row in enumerate(rows):
        same = row.axis_group == rows[index - 1].axis_group if index else False
        xs.append(0.0 if index == 0 else xs[-1] + (GROUP_STEP if same else 1.0))
    return xs


def leg_runs(rows: Sequence[SetupRow]) -> list[tuple[str, int, int]]:
    """Each contiguous run of one delivery as ``(leg, first index, last index)``."""
    runs: list[tuple[str, int, int]] = []
    for index, row in enumerate(rows):
        if runs and runs[-1][0] == row.axis_group:
            runs[-1] = (row.axis_group, runs[-1][1], index)
        else:
            runs.append((row.axis_group, index, index))
    return runs


def leg_centres(rows: Sequence[SetupRow]) -> list[float]:
    """The x of each delivery's tick: the middle of its run of columns."""
    xs = column_x(rows)
    return [(xs[first] + xs[last]) / 2.0 for _, first, last in leg_runs(rows)]


def column_limits(rows: Sequence[SetupRow]) -> tuple[float, float]:
    """The X limits of a row of columns: 0.6 past the outer columns, never a degenerate axis."""
    xs = column_x(rows) or [0.0]
    return -0.6, max(xs[-1] + 0.6, 0.6)


#: Short tick spellings for deliveries whose display name does not fit a column of a joined row,
#: with the footnote the key carries for each. The tick is the abbreviation, so the column stays
#: readable; the key is where the reader finds out what it stands for.
TICK_ALIASES: dict[str, tuple[str, str]] = {
    study_tags.OFFLOAD_DELIVERY_NAME: ("OpenMP", "OpenMP = OpenMP offload"),
}


#: Words :func:`title_case` leaves lower case inside a title.
TITLE_SMALL_WORDS: frozenset[str] = frozenset({"vs", "of", "and", "per", "on", "in", "to", "the"})
#: Registry-key words :func:`title_case` spells as acronyms.
TITLE_ACRONYMS: frozenset[str] = frozenset(
    {"cpf", "cpu", "gpu", "hip", "mpi", "omp", "llr", "llm", "blas", "fft", "sdfg", "ml", "ai", "api", "abi"}
)


def title_case(text: str) -> str:
    """``text`` as a title for a tick or a key row: a lower-case word or registry key is capitalized
    (``lang-skills`` -> ``Lang Skills``, a leading ``+`` set apart: ``C +lang-skills`` -> ``C + Lang Skills``);
    a word with any capital (``HIP``, ``Qwen3.8-27B``, ``DaCe+CPF``), a number or a unit (``1x``, ``95%``), a single
    letter and the small words of :data:`TITLE_SMALL_WORDS` are kept as written."""
    words: list[str] = []
    for index, word in enumerate(text.split(" ")):
        lead = "+" if word.startswith("+") and len(word) > 1 else ""
        body = word[len(lead) :]
        core = body.rstrip(",:;)")
        if (
            len(core) > 1
            and core.replace("-", "").isalpha()
            and core.islower()
            and (index == 0 or core not in TITLE_SMALL_WORDS)
        ):
            body = (
                " ".join(part.upper() if part in TITLE_ACRONYMS else part.capitalize() for part in core.split("-"))
                + body[len(core) :]
            )
        words.append(f"{lead} {body}" if lead else body)
    return " ".join(words)


def category_tick(leg: str, config: FigureConfig) -> str:
    """A category's tick text in title case: a delivery with a packet breaks before its ``+`` (``C`` over
    ``+ Lang Skills``), so the language stays one word on top; anything else wraps at ``config.tick_wrap``."""
    text = title_case(tick_alias(leg))
    head, plus, packet = text.partition(" + ")
    return f"{head}\n+ {packet}" if plus else wrapped_label(text, config.tick_wrap)


def tick_alias(leg: str) -> str:
    """``leg``'s tick spelling: its abbreviation where it has one."""
    alias = TICK_ALIASES.get(leg)
    return alias[0] if alias else leg


def alias_footnotes(legs: Sequence[str]) -> list[Line2D]:
    """One text-only key row per abbreviation the figure actually drew."""
    return [
        Line2D([], [], linestyle="none", marker="none", label=TICK_ALIASES[leg][1])
        for leg in dict.fromkeys(legs)
        if leg in TICK_ALIASES
    ]


#: How much taller one line of drawn text makes a band than the type itself: leading plus the gap
#: to whatever sits under it. Bands are DERIVED from the type scale rather than fixed in inches --
#: a 0.22in band is right above a 13.5pt name and half empty above an 8pt one, and that empty half
#: is what makes a page-budgeted row's panels look small.
LINE_BAND: float = 1.5


def text_band(points: float, lines: int = 1) -> float:
    """``lines`` of ``points``-sized text as a band height, inches."""
    return points / style.POINTS_PER_INCH * LINE_BAND * lines


#: Below this width (inches) a figure's key goes compact and two columns wide: a wrap figure.
NARROW_FIGURE_IN: float = 3.0

#: The length of a category tick mark, points.
CATEGORY_TICK_PT: float = 2.5

#: How much of a staggered name's drop its tick mark grows by.
STAGGER_TICK_SHARE: float = 0.25


def row_label(rows: Sequence[SetupRow], index: int) -> str:
    """A column's label when its delivery is blank (a stub's pending model): the row's own label."""
    return rows[index].label


def group_rules(ax: Axes, rows: Sequence[SetupRow]) -> None:
    """A light rule between one delivery's group of columns and the next, on every row of the figure,
    so the groups read as groups without a box around each."""
    xs = column_x(rows)
    for _, first, _ in leg_runs(rows)[1:]:
        ax.axvline((xs[first - 1] + xs[first]) / 2.0, color=style.RULE, linewidth=0.8, zorder=0)


#: Characters of a rotated Y label per point of its type: 18 at 13.5pt.
LABEL_CHARS_PER_PT: float = 18.0 / 13.5


def label_wrap(config: FigureConfig) -> int:
    """How many characters of a ROTATED Y label fit the row it labels, at ``config``'s own type."""
    return max(8, int(config.type_.label_pt * LABEL_CHARS_PER_PT))


def wrapped_label(text: str, width: int = 18, hyphens: bool = False) -> str:
    """A label folded onto as many lines as it needs, never INSIDE a word. A dot-row panel is about
    two inches tall and its Y label is rotated, so "Geomean Speedup Over Numba" on one line runs
    off both ends of the row -- but a fold at the hyphen gives "Geomean Speed-" over "Up", which is
    worse than the overflow.

    ``hyphens`` allows the fold at a hyphen, for a TICK: a model name has no spaces to fold at
    ("Qwen3.8-27B"), so without it three of them under three columns print on top of one another.
    """
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=hyphens)


#: How a figure names its own panels, so a caption can refer to one of them (:func:`panel_tag`).
PANEL_LETTERS: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g", "h")


def panel_tag(index: int) -> str:
    """``(a)``, ``(b)``, ... for panel ``index``: LaTeX's subfigure tag, so a figure's panels read as the
    caption names them."""
    return f"({PANEL_LETTERS[index]})" if index < len(PANEL_LETTERS) else f"({index + 1})"


def fold_width(text: str, wrap: int, lines: int) -> int:
    """The narrowest width at or above ``wrap`` that folds ``text`` onto at most ``lines``.

    Searched rather than estimated: the first line carries the tag as well as its first word
    ("iii) Repository"), which no per-word or per-character rule predicts -- estimating it is what
    put one panel name on three lines.
    """
    for width in range(max(wrap, -(-len(text) // lines)), max(len(text), wrap) + 1):
        if wrapped_label(text, width).count("\n") < lines:
            return width
    return max(len(text), wrap)


def name_line_width(text: str, wrap: int, lines: int = 2) -> int:
    """:func:`fold_width` at ``lines`` (:attr:`FigureConfig.max_name_lines`)."""
    return fold_width(text, wrap, lines)


def name_layout(
    names: Sequence[str], spans: Sequence[float], points: float, lines: int = 2, em: float = NAME_CHAR_EM
) -> tuple[float, list[int]]:
    """``(one type size, one fold width per panel)`` for a row of panel names.

    ONE size, the smallest any name needs: set at its own size each, a row whose third name is
    longer than its column reads as three headings rather than one row of them. Each fold width is
    then that panel's own span measured at THAT size, so every name folds onto at most ``lines`` --
    and onto ONE where its span holds it.
    """
    size, folds = points, [len(name) for name in names]
    # The fold decides the size and the size decides the fold, so it is iterated to a fixed point
    # (a long word folds wider than ceil(len/lines)).
    for _ in range(4):
        folds = [
            name_line_width(name, panel_name_wrap(span, size, em), lines)
            for name, span in zip(names, spans, strict=True)
        ]  # fmt: skip
        settled = min(
            (min(points, span * style.POINTS_PER_INCH / (fold * em)) for span, fold in zip(spans, folds, strict=True)), default=points
        )  # fmt: skip
        if abs(settled - size) < 0.05:
            break
        size = settled
    return size, folds


#: One comparison an arrow is drawn across, as ``(model tag, delivery)``.
DifferenceKey = tuple[str, str]


def parse_differences(spec: str) -> frozenset[DifferenceKey]:
    """``HIP:qwen38,Triton:kimi27sglang`` as the set of comparisons to draw an arrow across.

    Asked for BY NAME rather than drawn everywhere: an arrow on every column is a second grid, and
    the ones worth drawing are the ones a caption is going to quote.
    """
    keys: set[DifferenceKey] = set()
    for token in str(spec).split(","):
        leg, _, model = token.strip().partition(":")
        if leg.strip() and model.strip():
            keys.add((model.strip(), leg.strip()))
    return frozenset(keys)


def difference_factor(control_value: float, treated_value: float, measure: Measure) -> float:
    """The factor between one comparison's two marks, in the measure's own units: the speedup row
    holds ``log2(ratio)``, so its factor is a power of two, while the cost row holds counts."""
    if measure is Measure.COST:
        return treated_value / control_value if control_value > 0.0 else math.nan
    return 2.0 ** (treated_value - control_value)


def difference_middle(control_value: float, treated_value: float, measure: Measure) -> float:
    """Where the arrow's label sits: halfway along the arrow AS DRAWN, which is the geometric
    middle on the cost row's log axis and the arithmetic one on the log2 speedup row."""
    if measure is Measure.COST:
        return math.sqrt(control_value * treated_value) if control_value > 0.0 else math.nan
    return (control_value + treated_value) / 2.0


def interval_bounds(values: Sequence[float], cost: bool, config: FigureConfig) -> tuple[float, float]:
    """How far a panel's intervals are drawn: :data:`FigureConfig.interval_reach` past its lowest and
    highest mark, in the row's own units (tokens on the cost row, ``log2(ratio)`` on the speedup
    row). No finite mark leaves nothing to cut against."""
    marks = [value for value in values if math.isfinite(value)]
    if not marks:
        return -math.inf, math.inf
    if cost:
        return min(marks) / config.interval_reach, max(marks) * config.interval_reach
    reach = math.log2(config.interval_reach)
    # A speedup row whose marks all sit at or above 1x is floored there: an interval reaching below
    # is cut at 1x with an arrowhead, so the axis never opens below the baseline.
    low = max(min(marks) - reach, 0.0) if min(marks) >= 0.0 else min(marks) - reach
    return low, max(marks) + reach


def interval_kernels(point: SetupPoint, measure: Measure) -> int:
    """How many kernels one setup's interval on ``measure`` is taken over: every served kernel for cost,
    the kernels both setups solved for speedup."""
    return point.token_kernels if measure is Measure.COST else point.kernels


#: The key's note for a mark drawn without its interval (:attr:`FigureConfig.min_interval_kernels`).
FEW_KERNELS_NOTE: str = "No interval: fewer than {} kernels"


def measure_value(point: SetupPoint, measure: Measure) -> tuple[float, float, float]:
    """``(value, low, high)`` of one setup on one measure: the speedup in ``log2(ratio)``, or the
    token count as a count. The success row draws its count alone (:func:`draw_success_row`)."""
    if measure is Measure.COST:
        return point.y, point.y_low, point.y_high
    return point.x, point.x_low, point.x_high


#: How far past the data a border may snap to the next tick, as a fraction of one tick step, and the
#: pad a border keeps from the data when it does not snap.
SNAP_REACH: float = 0.5
SNAP_PAD: float = 0.1


def near_ticks(ticks: tuple[float, float], data: tuple[float, float], steps: int, log: bool) -> tuple[float, float]:
    """The snapped ``ticks`` either side of ``data``, each pulled in to the data (with
    :data:`SNAP_PAD`) where it sits more than :data:`SNAP_REACH` of a tick step away: a 0.94x
    interval end would otherwise open the axis down to 0.5x, a whole empty step."""
    to_axis = math.log if log else (lambda value: value)
    from_axis = math.exp if log else (lambda value: value)
    (low, high), (data_low, data_high) = (tuple(map(to_axis, pair)) for pair in (ticks, data))
    step = (high - low) / steps
    if data_low - low > SNAP_REACH * step:
        low = data_low - SNAP_PAD * step
    if high - data_high > SNAP_REACH * step:
        high = data_high + SNAP_PAD * step
    return from_axis(low), from_axis(high)


def is_pending(row: SetupRow) -> bool:
    """Whether a category holds no measurement on either side: a slot kept for data still running."""
    return all(point.kernels == 0 and point.served == 0 for point in (row.control, row.treated))


def verdicts_of(significance: dict[tuple[str, str], Significance], row: SetupRow) -> Significance:
    """``row``'s two verdicts. The test table names a column by its LANGUAGE ("C") where the figure's column also
    names the packet it delivered ("C +lang-skills"): an exact key first, else the language before the ``+``."""
    exact = significance.get((row.model, row.leg))
    if exact is not None:
        return exact
    language = row.leg.split(" +", 1)[0].strip()
    return significance.get((row.model, language), NO_SIGNIFICANCE)


def row_verdict(row: SetupRow, measure: Measure, significance: dict[tuple[str, str], Significance]) -> bool:
    """Whether ``row`` is starred on ``measure``'s row. Each row carries only ITS OWN verdict: a star
    on the cost row would test the speedup."""
    score_sig, cost_sig = verdicts_of(significance, row)
    return {Measure.COST: cost_sig, Measure.SUCCESS: False}.get(measure, score_sig)


#: Each measure's superscript (:func:`draw_verdict`); anything else is the speedup's.
VERDICT_MARKS: dict[Measure, str] = {Measure.COST: COST_SIG_MARK}


#: How much a relative row grows at its foot for the baseline band, and above its highest value, as shares of
#: the data's own span (:func:`draw_baseline_band`).
BASELINE_BAND_FRACTION: float = 0.45
BASELINE_HEADROOM_FRACTION: float = 0.12
#: The baseline band's type size as a share of the key's: it is a footnote to the marks above it.
BASELINE_BAND_TYPE: float = 0.85


#: Relative mode's row names: each row is a change over the LLM's own baseline, not a level.
RELATIVE_LABELS: dict[Measure, str] = {Measure.SPEEDUP: "Speedup Change", Measure.COST: "Token Change"}


def mode_labels(config: FigureConfig) -> dict[Measure, str]:
    """The row names ``config.mode`` overrides: :data:`RELATIVE_LABELS` in relative mode, none otherwise."""
    return dict(RELATIVE_LABELS) if config.mode is Mode.RELATIVE else {}


def baseline_text(point: SetupPoint, measure: Measure) -> str:
    """A control's ABSOLUTE level as the baseline band prints it: its geomean speedup over the experiment
    baseline (``5.4x``) on the speedup row, its arithmetic-mean tokens per kernel (``1.15M``, ``820K``) on the
    token row; "" where it has none."""
    if measure is Measure.COST:
        tokens = point.y
        if not math.isfinite(tokens):
            return ""
        return f"{tokens / 1e6:.2f}M" if tokens >= 1e6 else f"{tokens / 1e3:.0f}K"
    if not math.isfinite(point.x):
        return ""
    ratio = 2.0**point.x
    return f"{ratio:.0f}x" if ratio >= 10.0 else f"{ratio:.1f}x"


#: The success row's labelled rates: none, half, all of the kernels served.
SUCCESS_TICKS: tuple[float, ...] = (0.0, 0.5, 1.0)


def success_rate(solved: int, served: int) -> float:
    """The fraction of its served kernels a setup solved."""
    return solved / served if served else 0.0


#: A character's advance as a share of its point size, for fitting a rotated row name to its row's height.
CHAR_EM: float = 0.5


def two_line_fold(text: str) -> str:
    """``text`` on at most two lines, broken at the word boundary that makes its longest line shortest: the
    rotated name's length is what bounds its size (:func:`even_row_labels`)."""
    words = text.split()
    splits = [(" ".join(words[:cut]), " ".join(words[cut:])) for cut in range(1, len(words))]
    best = min(splits, key=lambda pair: max(len(pair[0]), len(pair[1])), default=(text, ""))
    return "\n".join(line for line in best if line) if max(map(len, best)) < len(text) else text


def row_label_fit(ax: Axes, ceiling: float) -> float:
    """The largest size up to ``ceiling`` at which ``ax``'s row name's longest line fits the row's height."""
    fig = ax.get_figure(root=True)
    lines = ax.get_ylabel().splitlines()
    if not isinstance(fig, Figure) or not lines:
        return ceiling
    height_pt = ax.get_position().height * float(fig.get_size_inches()[1]) * style.POINTS_PER_INCH
    return min(ceiling, height_pt / (CHAR_EM * max(len(line) for line in lines)))


@dataclasses.dataclass(frozen=True, slots=True)
class DotColumn:
    """ONE study's column of a stacked row figure: its categories, its packet's shape and the
    verdicts its marks wear. An empty ``rows`` is a STUB column -- the axes and the name, nothing
    plotted -- which holds a slot for a comparison that has not finished running."""

    title: str
    treatment: str
    shape: Marker
    rows: tuple[SetupRow, ...]
    significance: dict[tuple[str, str], Significance]
    symbols: Significance
    reference: str
    control: str
    differences: frozenset[DifferenceKey]


def placeholder_rows(rows: Sequence[SetupRow], deliveries: Sequence[str]) -> list[SetupRow]:
    """``rows`` plus one empty category per (model already drawn, delivery in ``deliveries``)."""
    if not deliveries:
        return list(rows)
    extra = [
        SetupRow(model, leg, palette.model_color(model), EMPTY_POINT, EMPTY_POINT)
        for model in dict.fromkeys(row.model for row in rows)
        for leg in deliveries
        if (model, leg) not in {(r.model, r.leg) for r in rows}
    ]
    return column_order([*rows, *extra])


def pending_rows(rows: Sequence[SetupRow], models: Sequence[str], leg: str = "") -> list[SetupRow]:
    """``rows`` plus one empty category per model of ``models`` not drawn yet, under the delivery the
    drawn models use, or ``leg`` in a stub panel (its placeholder delivery), in the registry's model
    order. A stub category with no delivery would be named by its model, which the colour already
    says and which overprints at a stub's column width."""
    drawn = {row.model for row in rows}
    leg = rows[0].leg if rows else leg
    extra = [
        SetupRow(model, leg, palette.model_color(model), EMPTY_POINT, EMPTY_POINT)
        for model in dict.fromkeys(models)
        if model not in drawn
    ]
    if not extra:
        return list(rows)
    return column_order([*rows, *extra])


def nth(values: Sequence[str], index: int) -> str:
    """``values[index]``, or "" past the end: a per-column list may stop short of the row."""
    return values[index] if index < len(values) else ""


def comma_list(spec: str) -> list[str]:
    """``"a, b,,c"`` -> ``["a", "b", "c"]``."""
    return [part.strip() for part in spec.split(",") if part.strip()]


def panel_rows(
    frame: pd.DataFrame | dict[str, pd.DataFrame],
    over: population.KernelPolicy,
    card: cost_models.CostModel | None = None,
    mode: Mode = Mode.ABSOLUTE,
) -> list[SetupRow]:
    """A panel's drawn categories (:func:`relative_rows` in relative ``mode``); none for a STUB panel."""
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        return relative_rows(frame, over, card) if mode is Mode.RELATIVE else setup_rows(frame, over, card)
    return []


def per_column_rows(rows: Sequence[SetupRow], treatment: str) -> list[SetupRow]:
    """``rows`` of a :data:`PER_COLUMN_TREATMENTS` panel, each wearing its own column's registered
    shape, and, under :data:`GROUPED_BY_DELIVERY`, sitting under its delivery's tick."""
    grouped = treatment in GROUPED_BY_DELIVERY
    return column_order(
        [
            dataclasses.replace(
                row, shape=column_treatment_shape(row.leg), group=row.leg.rsplit("-", 1)[0] if grouped else ""
            )
            for row in rows
        ]
    )


def column_shape(treatment: str) -> Marker:
    """A panel's treated shape. A :data:`PER_COLUMN_TREATMENTS` panel has none of its own -- each
    column wears its own packet's or harness's (:func:`per_column_rows`) -- so the registry is never
    asked for the pseudo-intervention's name."""
    return "" if treatment in PER_COLUMN_TREATMENTS else treatment_marker(treatment)


def dot_columns(
    panels: Sequence[Panel],
    references: Sequence[str],
    control_names: Sequence[str] = (),
    differences: Sequence[str] = (),
    placeholders: Sequence[str] = (),
    over: population.KernelPolicy = SPEEDUP_OVER,
    pending: Sequence[str] = (),
    card: cost_models.CostModel | None = None,
    mode: Mode = Mode.ABSOLUTE,
) -> list[DotColumn]:
    """Each panel of a joined row reduced to its own :class:`DotColumn`. ``pending[i]`` names, comma
    separated, the models panel ``i`` keeps an empty category for (:func:`pending_rows`)."""
    columns: list[DotColumn] = []
    for index, (title, treatment, stats, frame) in enumerate(panels):
        key = str(treatment if isinstance(treatment, str) else (flat_treatments(treatment) or [""])[0])
        table = stats if isinstance(stats, pd.DataFrame) else pd.concat(stats.values(), ignore_index=True)
        rows = panel_rows(frame, over, card, mode)
        named = comma_list(str(nth(placeholders, index)))
        stub_leg = named[0] if named and not rows else ""
        rows = placeholder_rows(rows, named) if rows else rows
        rows = pending_rows(rows, comma_list(str(nth(pending, index))), stub_leg)
        if key in PER_COLUMN_TREATMENTS:
            rows = per_column_rows(rows, key)
        columns.append(
            DotColumn(
                title=str(title),
                treatment=key,
                shape=column_shape(key),
                rows=tuple(rows),
                significance=axis_significance(table),
                symbols=drawn_symbols(table),
                reference=nth(references, index),
                control=nth(control_names, index) or packets.control_label([key]),
                differences=parse_differences(nth(differences, index)),
            )
        )
    return columns


@dataclasses.dataclass(frozen=True, slots=True)
class Comparator:
    """A compiler or framework drawn beside the models of one delivery group (Pluto beside the C
    answers, PPCG beside HIP): ``name`` is its key (``pluto``, ``ppcg_hip``, ``jax_cpu``), ``group``
    the delivery tick it sits under, ``speedups`` the baseline-over-comparator ratio of each kernel
    it ran validly, and ``served`` the tag it was asked for."""

    name: str
    group: str
    speedups: tuple[float, ...]
    served: int


def comparator_point(comparator: Comparator) -> SetupPoint:
    """The comparator's one mark: the geomean of its valid kernels' speedups with its 95% log-t
    interval (:func:`~hpcagent_bench.stats.summary.geomean_interval`, none below
    :data:`~hpcagent_bench.stats.summary.MIN_PAIRS_FOR_INTERVAL` kernels), and its solved share."""
    values = np.asarray(comparator.speedups, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    speed = summary.geomean_interval(values)
    return SetupPoint(
        x=summary.log2_change(speed.point) if values.size else math.nan,
        x_low=summary.log2_change(speed.low) if math.isfinite(speed.low) else math.nan,
        x_high=summary.log2_change(speed.high) if math.isfinite(speed.high) else math.nan,
        y=math.nan, y_low=math.nan, y_high=math.nan, kernels=int(values.size), token_kernels=0,
        solved=int(values.size), served=comparator.served,
    )  # fmt: skip


def comparators_from_table(table: pd.DataFrame, entries: Sequence[tuple[str, str]]) -> list[Comparator]:
    """``entries`` (comparator key, delivery group) read off a comparator CSV (``kernel``,
    ``comparator``, ``speedup``; one row per tag kernel, ``speedup`` blank where the comparator
    has no valid run). A key the table does not name is skipped."""
    out: list[Comparator] = []
    for name, group in entries:
        selected = table.loc[table["comparator"].astype(str) == name]
        if selected.empty:
            continue
        speedups = np.asarray(pd.to_numeric(selected["speedup"], errors="coerce"), dtype=float)
        valid = [float(v) for v in speedups if math.isfinite(v) and v > 0.0]
        out.append(Comparator(name, group, tuple(valid), len(set(selected["kernel"].astype(str).tolist()))))
    return out


def comparator_table(comparators: Sequence[Comparator]) -> pd.DataFrame:
    """One row per drawn comparator: its geomean speedup, interval and solved share (SC15 Rule 4)."""
    rows = []
    for comparator in comparators:
        point = comparator_point(comparator)
        rows.append({
            "comparator": comparator.name, "group": comparator.group, "geomean_speedup": 2.0 ** point.x,
            "speedup_low": 2.0 ** point.x_low, "speedup_high": 2.0 ** point.x_high, "kernels": point.kernels,
            "served": point.served,
        })  # fmt: skip
    return pd.DataFrame(rows)


def comparator_shapes(names: Sequence[str]) -> dict[str, str]:
    """Each comparator's shape: its registered optimizer shape (:func:`~hpcagent_bench.stats.palette.
    marker`) unless a registered packet or the control wears it, else the first of the registry's
    ``markers`` no packet, control or earlier comparator (``frameworks`` order) wears. A comparator
    never reads as a packet, two never share a shape, and the same set of comparators gets the same
    shapes in every figure. Keyed by FRAMEWORK: ``jax_cpu`` and ``jax_gpu`` are one JAX, one shape."""
    taken: list[object] = [CONTROL_MARKER, *(palette.packet_marker(key) for key in palette.hue_order("packets"))]
    shapes: dict[str, str] = {}
    optimizers = study_tags.order("optimizers")
    for name in palette.in_order([study_tags.canonical("frameworks", name) for name in names], "frameworks"):
        own = palette.marker(name) if study_tags.canonical("optimizers", name) in optimizers else None
        free = [shape for shape in palette.markers() if shape not in taken]
        shape = own if own is not None and own not in taken else (free[0] if free else CONTROL_MARKER)
        shapes[name] = shape
        taken.append(shape)
    return shapes


def comparator_rows(comparators: Sequence[Comparator], shapes: dict[str, str]) -> list[SetupRow]:
    """``comparators`` as categories: the framework's own colour
    (:func:`~hpcagent_bench.stats.palette.framework_color`, never a model's), ``shapes``' shape."""
    return [
        SetupRow(
            comparator.name,
            comparator.group,
            palette.framework_color(comparator.name),
            EMPTY_POINT,
            comparator_point(comparator),
            shape=shapes[study_tags.canonical("frameworks", comparator.name)],
            group=comparator.group,
            comparator=True,
        )
        for comparator in comparators
        if comparator.speedups
    ]


def with_comparators(columns: Sequence[DotColumn], comparators: Sequence[Sequence[Comparator]]) -> list[DotColumn]:
    """``columns`` with ``comparators[i]`` added to column ``i`` after the models of their group,
    each in one shape for the whole figure (:func:`comparator_shapes`)."""
    if not any(comparators):
        return list(columns)
    shapes = comparator_shapes([one.name for group in comparators for one in group])
    placed: list[DotColumn] = []
    for index, column in enumerate(columns):
        # A comparator named without a group sits under the column's first delivery.
        first = column.rows[0].axis_group if column.rows else ""
        grouped = [dataclasses.replace(one, group=one.group or first) for one in nth_list(comparators, index)]
        placed.append(
            dataclasses.replace(column, rows=tuple(column_order([*column.rows, *comparator_rows(grouped, shapes)])))
        )
    return placed


#: Short category names for the X ticks. The legend carries the full name; a tick has about 30pt at
#: five categories across a column, and "Qwen3.8" beside "GPT-OSS" already overprinted there.
SHORT_NAMES: dict[str, str] = {
    "qwen38": "Qwen",
    "oss120b": "OSS",
    "kimi27sglang": "Kimi",
    "glm53": "GLM",
    "dace_cpu_canon": "DaCe+CPF",
    "dace_gpu_canon": "DaCe+CPF",
    "dace_cpu_parallel": "DaCe+AOpt",
    "pluto": "Pluto",
    "ppcg_hip": "PPCG",
}


def nth_list(values: Sequence[Sequence[Comparator]], index: int) -> Sequence[Comparator]:
    """``values[index]``, or none past the end: the comparator list may stop short of the row."""
    return values[index] if index < len(values) else ()


def comparator_short_name(name: str) -> str:
    """A comparator's key text: its short name where it has one (``PPCG``; the caption says it is
    CUDA through hipify), else its ``frameworks`` name."""
    resolved = study_tags.canonical("frameworks", name)
    return SHORT_NAMES.get(resolved, study_tags.framework_name(resolved))


def first_per_label(handles: Sequence[Line2D]) -> list[Line2D]:
    """``handles`` keeping only the first of each label."""
    kept: dict[str, Line2D] = {}
    for handle in handles:
        kept.setdefault(str(handle.get_label()), handle)
    return list(kept.values())


#: The narrowest a column may be, in categories. A stub column has none, and at a width ratio of
#: one beside a nine-category neighbour it collapsed to a sliver its own name could not sit over.
MIN_COLUMN_CATEGORIES: int = 2


def dot_row_widths(columns: Sequence[DotColumn], config: FigureConfig = DEFAULT_CONFIG) -> list[float]:
    """Each column's share of the row's width: its own category count, floored at
    :data:`MIN_COLUMN_CATEGORIES`."""
    widths = [float(max(len(column.rows), MIN_COLUMN_CATEGORIES)) for column in columns]
    if widths:
        widest = widths.index(max(widths))
        widths[widest] *= config.wide_column_scale
    return widths


#: A short row's token ticks: 1-2-5 per decade, each one labelled. 1 and 3 left a decade two grid
#: lines, too few to read a mark's cost off.
SHORT_ROW_TOKEN_SUBS: tuple[float, ...] = (1.0, 2.0, 5.0)
#: A row lower than this (inches) is short: it takes :data:`SHORT_ROW_TOKEN_SUBS`.
SHORT_ROW_IN: float = 1.6


def measure_row_config(config: FigureConfig, row_height_in: float) -> FigureConfig:
    """``config`` sized for a row this tall. A rotated Y label and a tick ladder are both bounded by
    the row's HEIGHT, not by the figure's width: the same 8pt label and thirteen ratios that fit a
    1.5in row overprint each other in half of one.
    """
    return dataclasses.replace(
        retyped(config, label_pt=min(config.type_.label_pt, max(6.0, row_height_in * 8.0))),
        max_ticks=max(4, int(row_height_in * 7.0)),
        token_subs=config.token_subs if row_height_in >= SHORT_ROW_IN else SHORT_ROW_TOKEN_SUBS,
    )


#: Clearance kept between the end of one panel name and the start of the next, inches.
NAME_CLEARANCE_IN: float = 0.1


def dot_row_heights(
    measures: Sequence[Measure], row_height_in: float, config: FigureConfig
) -> tuple[list[float], float, float]:
    """``(height ratios, data box height in inches, hspace)`` of a dot-row grid.

    The DATA box's height is the fixed quantity: rows of a stated height plus the gaps between
    them. The chrome is measured and added OUTSIDE it, so a taller key grows the canvas instead of
    shrinking the rows, and every dot-row figure of a paper draws rows of one height. hspace is a
    fraction of the MEAN axes height, so it is rescaled to keep the gaps ``row_gap`` rows."""
    heights = [MEASURE_HEIGHT[measure] for measure in measures]
    data_height = row_height_in * (sum(heights) + config.row_gap * (len(measures) - 1))
    return heights, data_height, config.row_gap * len(measures) / sum(heights)


def dot_row_grid(
    columns: Sequence[DotColumn], row_width_in: float, config: FigureConfig
) -> tuple[list[float], list[float]]:
    """``(each column's width in inches, the grid's width ratios)``, a spacer column between panels.

    The gaps come OUT of the data width: matplotlib's wspace is a fraction of the mean axes width,
    so n panels and n-1 gaps share it. Ignoring that overstated every span by about a fifth, which
    is what let two panel names overlap. Every gap is a spacer column of its own, so each can be
    widened to what its neighbour's tick labels need (:func:`fit_column_gaps`) without narrowing
    the gaps that need nothing."""
    n = len(columns)
    ratios = dot_row_widths(columns, config)
    data_width = row_width_in - config.left_chrome_in
    axes_total = data_width / (1.0 + (n - 1) * config.column_gap / n)
    widths = [axes_total * ratio / sum(ratios) for ratio in ratios]
    spacer = config.column_gap * sum(ratios) / n
    return widths, [width for ratio in ratios for width in (ratio, spacer)][:-1]


def name_spans(widths: Sequence[float], config: FigureConfig) -> list[float]:
    """The width each panel name may fold against: its column PLUS the gap to the next one. The
    space beside a left-aligned name is empty until the next name starts, and refusing to use it
    forced two lines onto names that fit on one. The LAST name has no next one to run into, but no
    canvas either -- its own width and nothing more."""
    gap = config.column_gap * (sum(widths) / max(len(widths), 1))
    return [width + gap for width in widths[:-1]] + list(widths[-1:])


def placed_spans(axes: np.ndarray, row_width_in: float) -> list[float]:
    """:func:`name_spans` measured off the columns as placed: each to the next one's left edge, the
    last to the canvas's right edge (nothing follows it to run into)."""
    boxes = [ax.get_position() for ax in axes[0]]
    return [(nxt.x0 - box.x0) * row_width_in for box, nxt in itertools.pairwise(boxes)] + [
        (1.0 - boxes[-1].x0) * row_width_in
    ]


#: Clearance between an inner column's widest Y tick label and the panel to its left, in inches.
TICK_LABEL_CLEARANCE_IN: float = 0.05


def pairs_table(
    frame: pd.DataFrame,
    over: population.KernelPolicy = SPEEDUP_OVER,
    card: cost_models.CostModel | None = None,
) -> pd.DataFrame:
    """One row per (model, leg): the paired comparison behind a column's marks, as the CSV record
    beside the figure (SC15 Rule 4: the costs a ratio was taken over travel with it).

    An empty frame is a STUB panel, which draws nothing and therefore records nothing."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for model, leg, pair in setup_groups(frame):
        control, treated = skill_halves(pair)
        series = reduce_pair(control, treated, over, card)
        setups = setup_points(control, treated, over, card)
        if series is None or setups is None:
            continue
        rows.append(
            {
                "model": model,
                "leg": leg,
                "score_change": series.x,
                "score_change_low": series.x_low,
                "score_change_high": series.x_high,
                "cost_ratio": series.y,
                "cost_ratio_low": series.y_low,
                "cost_ratio_high": series.y_high,
                "kernels": series.kernels,
                "token_kernels": series.token_kernels,
                "delivered": series.delivered,
                "baseline_ns": series.baseline_ns,
                "native_ns": series.native_ns,
                "control_tokens": series.control_tokens,
                "treated_tokens": series.treated_tokens,
                "speedup_over": over.text,
                "served": setups[0].served,
                "control_solved": setups[0].solved,
                "treated_solved": setups[1].solved,
                # The baseline band's absolutes and the absolute mode's marks, with their 95% intervals: each
                # setup's geomean speedup over the experiment baseline and its arithmetic-mean tokens per kernel.
                **absolute_columns("control", setups[0]),
                **absolute_columns("treated", setups[1]),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    rules.require_costs(table, "score_change", ["baseline_ns", "native_ns"])
    rules.require_costs(table, "cost_ratio", ["control_tokens", "treated_tokens"])
    rules.require_interval(table, "score_change", "score_change_low", "score_change_high")
    return rules.require_interval(table, "cost_ratio", "cost_ratio_low", "cost_ratio_high")
