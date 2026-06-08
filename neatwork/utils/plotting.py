from __future__ import annotations

import matplotlib.artist
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import json
import os
import seaborn as sns
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# NOTE: `SweepConfigResult` / `TrialResult` live in `testing/data_model.py`, which is
# not shipped with the package. They are only referenced in type annotations of the
# paper-figure helpers below; `from __future__ import annotations` keeps those
# annotations lazy so this module imports cleanly without the testing harness.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data_model import SweepConfigResult, TrialResult

style = 'seaborn-v0_8-whitegrid'
plt.style.use(style)

# --- Metadata & Saving Utilities ---

def _read_png_metadata(path: str) -> dict:
    with Image.open(path) as img:
        if "custom_json" not in img.info:
            return {}
        return json.loads(img.info["custom_json"])

def _write_png_metadata(path: str, data: Union[str, dict, list]):
    img = Image.open(path)
    metadata = PngInfo()
    if isinstance(data, (dict, list)):
        data = json.dumps(data)
    metadata.add_text("custom_json", data)
    img.save(path, pnginfo=metadata)

def _save_figure_with_metadata(
    fig: plt.Figure, 
    path: str, 
    dpi: int = 300, 
    metadata: Optional[Any] = None
):
    """
    Saves the figure to the specified path. 
    If format is PNG and metadata is provided, embeds it.
    """
    root, ext = os.path.splitext(path)
    if ext == "":
        path = f"{root}.pdf"

    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    
    if metadata is not None and path.lower().endswith('.png'):
        _write_png_metadata(path, metadata)

@dataclass
class PlotGraph:
    """
    A generic, decoupled data structure for graph visualization.
    All coordinates are (row, col).
    """
    cost_map: NDArray  # 2D matrix of cost values
    fixed_nodes: List[Tuple[int, int]] = field(default_factory=list)
    dynamic_nodes: List[Tuple[int, int]] = field(default_factory=list)
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    highlight: Optional[NDArray] = None # 2D matrix (int >= 0) of highlighted cells
    fitness: Optional[float] = None
    fixed_node_ids: Optional[List[int]] = None  # IDs corresponding to fixed_nodes
    dynamic_node_ids: Optional[List[int]] = None  # IDs corresponding to dynamic_nodes
    edge_ids: Optional[List[int]] = None  # IDs corresponding to edges

def network_to_plot_graph(network, cost_map) -> PlotGraph:
    """Convert a Network to a PlotGraph for visualization."""
    from neatwork.types import NodeType
    from neatwork.neatwork_py_eval import matrix_trajectories_mask
    pos_by_id = {n.id: (int(n.pos[0]), int(n.pos[1])) for n in network.nodes}
    fixed = [(n.id, pos_by_id[n.id]) for n in network.nodes if n.type == NodeType.Fixed]
    dynamic = [(n.id, pos_by_id[n.id]) for n in network.nodes if n.type == NodeType.Flexible]
    edges = []
    edge_ids = []
    for e in network.edges:
        if e.enabled and e.node1 in pos_by_id and e.node2 in pos_by_id:
            edges.append((pos_by_id[e.node1], pos_by_id[e.node2]))
            edge_ids.append(e.id)
    return PlotGraph(
        cost_map=cost_map,
        fixed_nodes=[p for _, p in fixed],
        dynamic_nodes=[p for _, p in dynamic],
        edges=edges,
        highlight=matrix_trajectories_mask(cost_map, edges),
        fitness=network.fitness,
        fixed_node_ids=[i for i, _ in fixed],
        dynamic_node_ids=[i for i, _ in dynamic],
        edge_ids=edge_ids,
    )


def plot_graph_on_ax(
    ax: plt.Axes, 
    graph: PlotGraph, 
    title: str,
    setup_axis: bool = True,
    show_ids: bool = False,
    show_blobs: bool = True,
    show_legend: bool = True,
    show_grid: bool = False,
    show_coordinates: bool = True
) -> List[matplotlib.artist.Artist]:
    """
    Core utility function to draw a single PlotGraph onto a provided matplotlib axis.
    
    Args:
        ax: The matplotlib axis to draw on.
        graph: The PlotGraph data object.
        title: The title for the plot.
        setup_axis: If True (default), clear and format the axis. 
                    If False, only plot data artists (for faster updates).
        show_ids: If True, display node and edge IDs on the plot.
        show_blobs: If True (default), display flexible nodes as green circles.
        show_legend: If True (default), add a legend to the axis (only if setup_axis is True).
                    
    Returns:
        A list of the artists added to the plot.
    """
    all_artists: List[matplotlib.artist.Artist] = []

    # 1. Get data from the PlotGraph object
    cost_map = graph.cost_map
    highlight_map = graph.highlight
    if cost_map is None:
        print("Warning: cost_map is None. Cannot plot.")
        return []
    H, W = cost_map.shape
    
    # 2. Clear and format the axis (if requested)
    if setup_axis:
        ax.clear()

    # 3. Display the cost map
    im = ax.imshow(cost_map, cmap='binary', origin='upper', interpolation='nearest', 
                   vmin=0, vmax=float(np.max(cost_map)), zorder=1)
    all_artists.append(im)

    # 4. Plot nodes (Data is already prepared)
    if graph.fixed_nodes:
        fx, fy = np.array(graph.fixed_nodes)[:, 1], np.array(graph.fixed_nodes)[:, 0]
        # Changed marker to 'D' (diamond) and size to 100 (was 150)
        sc_fixed = ax.scatter(fx, fy, c='blue', marker='D', s=100, label='Fixed Nodes', zorder=10)
        all_artists.append(sc_fixed)
    
    if show_blobs and graph.dynamic_nodes:
        flx, fly = np.array(graph.dynamic_nodes)[:, 1], np.array(graph.dynamic_nodes)[:, 0]
        # Changed size to 70 (was 100)
        sc_flex = ax.scatter(flx, fly, c='green', marker='o', s=70, label='Flexible Nodes', zorder=10)
        all_artists.append(sc_flex)

    # 5. Plot edges and highlights
    
    # --- Plot gray edge lines ---
    for p1, p2 in graph.edges:
        line, = ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'gray', linestyle='-', linewidth=1.5, alpha=0.7, zorder=5)
        all_artists.append(line)

    # --- Plot highlighted cells ---
    if highlight_map is not None:
        # Create an RGBA image instead of patches
        # 0.0, 0.3, 0.9 is the highlight color
        highlight_color = np.array([0.0, 0.3, 0.9, 1.0])
        img_highlight = np.zeros((*highlight_map.shape, 4))
        
        # Normalized highlighting 
        mask = highlight_map > 0
        if np.any(mask):
            hl_values = highlight_map[mask].astype(float)
            v_min, v_max = hl_values.min(), hl_values.max()
            
            # Map values to alpha range [0.3, 0.6]
            if v_max > v_min:
                alphas = 0.3 + 0.3 * (hl_values - v_min) / (v_max - v_min)
            else:
                alphas = np.full_like(hl_values, 0.45)
                
            img_highlight[mask, :3] = highlight_color[:3]
            img_highlight[mask, 3] = alphas
            
            im_hl = ax.imshow(img_highlight, origin='upper', interpolation='nearest', zorder=2)
            all_artists.append(im_hl)

            # Add dummy artist for legend
            sc_dummy = ax.scatter([], [], c=[highlight_color[:3]], marker='s', s=100, 
                                   alpha=0.6, label='Activated Cells', zorder=-1)
            all_artists.append(sc_dummy)

    # 6. Plot formatting (if requested)
    if setup_axis:
        if show_coordinates:
            ax.set_xlabel('Column Index (x)')
            ax.set_ylabel('Row Index (y)')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            
        if show_legend:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Apply grid if requested - draw lines manually to ensure visibility
        if show_grid:
            # Draw vertical lines
            for x in range(W):
                ax.axvline(x - 0.5, color='cyan', linestyle='-', linewidth=0.8, alpha=0.7, zorder=3)
            # Draw horizontal lines
            for y in range(H):
                ax.axhline(y - 0.5, color='cyan', linestyle='-', linewidth=0.8, alpha=0.7, zorder=3)
        else:
            ax.grid(False)

    # 7. Set title (Always do this, as title changes)
    ax.set_title(title)
    
    # 8. Display IDs if requested
    if show_ids:
        # Display fixed node IDs
        # Positions are stored as (row, col), so we need pos[1] for x and pos[0] for y
        if graph.fixed_node_ids and len(graph.fixed_node_ids) == len(graph.fixed_nodes):
            for node_id, pos in zip(graph.fixed_node_ids, graph.fixed_nodes):
                text = ax.text(pos[1], pos[0], str(node_id), fontsize=8, ha='center', va='center',
                              color='white', weight='bold', zorder=20,
                              bbox=dict(boxstyle='circle,pad=0.2', facecolor='darkblue', alpha=0.7, edgecolor='none'))
                all_artists.append(text)
        
        # Display dynamic node IDs
        if graph.dynamic_node_ids and len(graph.dynamic_node_ids) == len(graph.dynamic_nodes):
            for node_id, pos in zip(graph.dynamic_node_ids, graph.dynamic_nodes):
                text = ax.text(pos[1], pos[0], str(node_id), fontsize=8, ha='center', va='center',
                              color='white', weight='bold', zorder=20,
                              bbox=dict(boxstyle='circle,pad=0.2', facecolor='darkgreen', alpha=0.7, edgecolor='none'))
                all_artists.append(text)
        
        # Display edge IDs at midpoints
        if graph.edge_ids and len(graph.edge_ids) == len(graph.edges):
            for edge_id, (p1, p2) in zip(graph.edge_ids, graph.edges):
                mid_y = (p1[0] + p2[0]) / 2
                mid_x = (p1[1] + p2[1]) / 2
                text = ax.text(mid_x, mid_y, str(edge_id), fontsize=7, ha='center', va='center',
                              color='black', weight='bold', zorder=15,
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8, edgecolor='gray', linewidth=0.5))
                all_artists.append(text)
    
    return all_artists


def plot_graph(
    graph: PlotGraph,
    title: str = "Graph Visualization",
    show_ids: bool = False,
    show_blobs: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    metadata: Optional[Any] = None,
    show_grid: bool = False,
    show_coordinates: bool = True,
    show_legend: bool = False,
):
    """
    Visualizes a single PlotGraph on top of its cost map.
    This is a standalone function for plotting one graph.

    Args:
        graph: The PlotGraph object to visualize.
        title: Title for the plot.
        show_ids: If True, display node and edge IDs on the plot.
        show_blobs: If True (default), display flexible nodes as green circles.
        save_path: If provided, save the figure to this path.
        dpi: Resolution for saved figure.
        metadata: Optional data to embed in the saved image (PNG only).
        show_grid: If True, display the grid.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    plot_graph_on_ax(ax, graph, title, show_ids=show_ids, show_blobs=show_blobs, show_legend=False, show_grid=show_grid, show_coordinates=show_coordinates)

    legend_by_label: Dict[str, matplotlib.artist.Artist] = {}
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label and not label.startswith('_') and label not in legend_by_label:
            legend_by_label[label] = handle

    if legend_by_label and show_legend:
        fig.legend(
            legend_by_label.values(),
            legend_by_label.keys(),
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            bbox_transform=ax.transAxes,
            frameon=True,
        )

    plt.tight_layout()

    if save_path:
        _save_figure_with_metadata(fig, save_path, dpi, metadata)

    plt.show()


def plot_graphs(
    graph_list: List[PlotGraph],
    titles: Optional[List[str]] = None,
    cols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    show_ids: bool = False,
    show_blobs: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    metadata: Optional[Any] = None,
    show_grid: bool = False,
    show_coordinates: bool = True,
    show_legend: bool = False,
):
    """
    Visualizes multiple PlotGraphs side-by-side or in a grid.
    
    Args:
        graph_list: List of PlotGraph objects to visualize.
        titles: Optional list of titles for each graph.
        cols: Number of columns in the grid.
        figsize: Figure size (width, height). Auto-calculated if None.
        show_ids: If True, display node and edge IDs.
        show_blobs: If True, display flexible nodes.
        save_path: If provided, save the figure.
        dpi: Resolution for saved figure.
        metadata: Optional data to embed in the saved image (PNG only).
    """
    n_graphs = len(graph_list)
    if n_graphs == 0:
        print("No graphs to plot.")
        return

    rows = int(np.ceil(n_graphs / cols))
    
    if figsize is None:
        # Scale figure size based on grid (approx 6x4 per graph)
        figsize = (6 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    # Handle titles
    if titles is None:
        current_titles = [f"Graph {i}" for i in range(n_graphs)]
    elif len(titles) != n_graphs:
        current_titles = [f"Graph {i}" for i in range(n_graphs)]
    else:
        current_titles = titles

    for i, ax in enumerate(axes_flat):
        if i < n_graphs:
            plot_graph_on_ax(ax, graph_list[i], current_titles[i], show_ids=show_ids, show_blobs=show_blobs, show_legend=False, show_grid=show_grid, show_coordinates=show_coordinates)
        else:
            ax.axis('off')

    # Let matplotlib manage legend placement outside the subplot area.
    legend_by_label: Dict[str, matplotlib.artist.Artist] = {}
    for i in range(n_graphs):
        handles, labels = axes_flat[i].get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith('_') and label not in legend_by_label:
                legend_by_label[label] = handle

    if legend_by_label and show_legend:
        fig.legend(
            legend_by_label.values(),
            legend_by_label.keys(),
            loc='center left',
            bbox_to_anchor=(0.84, 0.5),
            bbox_transform=fig.transFigure,
            frameon=True,
        )

    # Use a uniform layout policy across all combined plots to keep
    # subplot spacing consistent in paper figures.
    if legend_by_label:
        fig.subplots_adjust(left=0.08, right=0.82, bottom=0.10, top=0.92, wspace=0.25, hspace=0.25)
    else:
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92, wspace=0.25, hspace=0.25)
    
    if save_path:
         _save_figure_with_metadata(fig, save_path, dpi, metadata)
         
    plt.show()


def plot_pareto_scatter(
    scatter_groups: List[Tuple[List[Tuple[float, float]], str, str, str]],
    pareto_points: List[Tuple[float, float]],
    reference_points: Optional[List[Tuple[Tuple[float, float], str, str, str]]] = None,
    title: str = 'Pareto Front Scatter',
    xlabel: str = 'Objective 1',
    ylabel: str = 'Objective 2',
    figsize: Tuple[int, int] = (6, 4),
    save_path: Optional[str] = None,
    dpi: int = 300,
    metadata: Optional[Any] = None,
):
    """
    Visualizes Pareto front against sets of multi-objective points.
    
    Args:
        scatter_groups: List of (points, label, marker, color) tuples for plotting result clouds.
        pareto_points: List of (x, y) coordinates on the Pareto front.
        reference_points: Optional list of tuples formatted as (point, label, marker, color).
        title: Plot title.
        xlabel: Label for X-axis.
        ylabel: Label for Y-axis.
        save_path: If provided, save figure to this path.
        dpi: Resolution for saving.
        metadata: Optional metadata to embed for PNG saves.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for points, label, marker, color in scatter_groups:
        if points:
            xs, ys = zip(*points)
            ax.scatter(xs, ys, label=label, marker=marker, color=color, alpha=0.7, s=60)

    # Pareto front
    if pareto_points:
        # Sort points by X coordinate to draw a coherent line
        sorted_pareto = sorted(pareto_points, key=lambda p: p[0])
        pareto_xs, pareto_ys = zip(*sorted_pareto)
        ax.plot(pareto_xs, pareto_ys, color='red', linestyle='--', linewidth=1.5, zorder=1)
        ax.scatter(pareto_xs, pareto_ys, edgecolors='red', facecolors='none', s=120, linewidth=2, label='Pareto Front', zorder=2)

    # Optional reference points
    if reference_points:
        for pt, lbl, marker, color in reference_points:
            ax.scatter([pt[0]], [pt[1]], marker=marker, color=color, s=150, label=lbl, zorder=5)

    ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        _save_figure_with_metadata(fig, save_path, dpi, metadata)
        
    plt.show()


def plot_graph_evolution(graph_list: List[PlotGraph], 
                           titles: Optional[List[str]] = None,
                           show_ids: bool = False,
                           show_blobs: bool = True,
                           show_grid: bool = False,
                           show_coordinates: bool = True):
    """
    Visualizes a list of PlotGraph objects with an interactive slider.
    
    Optimized to only redraw data artists for fast updates.
    
    Args:
        graph_list: List of PlotGraph objects to visualize.
        titles: Optional list of titles for each graph.
        show_ids: If True, display node and edge IDs on the plot.
        show_blobs: If True (default), display flexible nodes as green circles.
    """
    if not graph_list:
        print("Graph list is empty. Nothing to plot.")
        return

    # --- Validate or create titles ---
    n_graphs = len(graph_list)
    if titles is None:
        plot_titles = [f"Graph {i}" for i in range(n_graphs)]
    elif len(titles) != n_graphs:
        print(f"Warning: Mismatch between number of graphs ({n_graphs}) "
              f"and titles ({len(titles)}). Using default titles.")
        plot_titles = [f"Graph {i}" for i in range(n_graphs)]
    else:
        plot_titles = titles

    # --- Create the Figure and Main Axes ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    
    # --- Artist Cache ---
    # This list will hold the artists from the *previous* plot
    current_artists: List[matplotlib.artist.Artist] = []
    
    # --- Define the Update Function (Optimized) ---
    def update(val):
        nonlocal current_artists
        
        index = int(val)
        graph = graph_list[index]
        title = plot_titles[index]
        
        # 1. REMOVE old artists
        for artist in current_artists:
            artist.remove()
        
        # 2. PLOT new artists (using setup_axis=False)
        # This only plots data, no clearing or re-formatting
        current_artists = plot_graph_on_ax(ax, graph, title, setup_axis=False, show_ids=show_ids, show_blobs=show_blobs, show_grid=show_grid, show_coordinates=show_coordinates)
        
        # 3. Redraw the canvas
        fig.canvas.draw_idle()

    # --- Create the Slider ---
    max_index = n_graphs - 1
    slider = Slider(
        ax=ax_slider,
        label='Index',
        valmin=0,
        valmax=max_index,
        valinit=max_index, # Default to the last item
        valstep=1
    )

    # --- Attach the update function ---
    slider.on_changed(update)
    fig.slider = slider # Store reference

    # --- Initial Plot ---
    # We must do a *full* setup for the first plot.
    # We use the graph at the initial index (max_index).
    initial_graph = graph_list[max_index]
    initial_title = plot_titles[max_index]
    
    # Call with setup_axis=True (the default) to clear, format,
    # and plot the first frame. Store the artists.
    current_artists = plot_graph_on_ax(ax, initial_graph, initial_title, show_ids=show_ids, show_blobs=show_blobs, show_grid=show_grid, show_coordinates=show_coordinates)
    
    plt.show()

@dataclass
class ConfigComparisonExperimentRun:
    successes: List[int]
    failure_prob: float
    label: str
    meta: Optional[str] = None

def _plot_config_performance_comparison_single(
    ax: plt.Axes,
    ax2: plt.Axes,
    runs: List[SweepConfigResult],
    show_means: bool = True,
    show_std: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> List[matplotlib.artist.Artist]:
    """
    Core plotting function for a single experiment result on given axes.
    Returns legend elements for later use.
    """

    # Extract data from SweepConfigResult objects
    success_results = [np.array(r.successes) for r in runs]
    failure_probabilities = [r.failure_prob for r in runs]
    labels = [r.label for r in runs]

    x = np.arange(len(success_results))
    
    # Colors
    COLOR_DOTS = '#1f77b4'
    COLOR_MEAN = '#d62728'
    COLOR_MEDIAN = '#2ca02c'
    COLOR_FAIL = '#555555'
    
    # --- Plot Data using Seaborn (Primary Axis) ---
    import pandas as pd
    
    # Prepare data for seaborn
    plot_data = []
    for i, r in enumerate(success_results):
        for val in r:
             plot_data.append({'Config': labels[i], 'Generations': val, 'x_pos': x[i]})
    
    if plot_data:
        df = pd.DataFrame(plot_data)
        sns.boxplot(
            data=df, x='x_pos', y='Generations', order=x.tolist(),
            ax=ax, color='#EBF5FB', boxprops=dict(alpha=0.5, edgecolor=COLOR_DOTS), 
            medianprops=dict(color=COLOR_MEDIAN, linewidth=2.5),
            whiskerprops=dict(visible=False), capprops=dict(visible=False),
            showfliers=False, width=0.6, zorder=1
        )
        sns.stripplot(
            data=df, x='x_pos', y='Generations', order=x.tolist(),
            ax=ax, color=COLOR_DOTS, size=5, alpha=0.6, 
            edgecolor='white', linewidth=0.8, jitter=True, zorder=3
        )
    
    # C. Mean and Standard Deviation
    if show_means or show_std:
        means = [float(np.mean(r)) if len(r) > 0 else np.nan for r in success_results]
        
        if show_std:
            stds = [float(np.std(r)) if len(r) > 0 else 0.0 for r in success_results]
            lower_errors = [min(std, mean) if not np.isnan(mean) else 0.0 
                           for mean, std in zip(means, stds)]
            upper_errors = stds
            
            ax.errorbar(
                x, means, 
                yerr=[lower_errors, upper_errors],
                fmt='none', 
                ecolor=COLOR_MEAN, 
                elinewidth=2.5, 
                capsize=6, 
                markeredgewidth=2,
                zorder=4, 
                alpha=0.9
            )
            
        if show_means:
            ax.scatter(
                x, means, 
                color='white',
                edgecolors=COLOR_MEAN,
                marker='D', 
                s=50, 
                linewidth=2,
                zorder=5
            )
    
    ax.set_ylabel('Number of generations', fontsize=11, fontweight='bold', color='#333333')
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Algorithm Configurations', fontsize=11)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # --- Secondary Axis (Failure Probabilities) ---
    ax2.bar(
        x, failure_probabilities, 
        color=COLOR_FAIL, alpha=0.4, width=0.8, 
        zorder=0, edgecolor='none'
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Failure Probability', color=COLOR_FAIL, fontsize=11)
    ax2.tick_params(axis='y', colors=COLOR_FAIL)
    
    # --- Grid & Spines ---
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.grid(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(COLOR_FAIL)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    
    # --- X-axis labels ---
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', labelsize=10)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    
    # Return legend elements
    legend_elements = [
        Line2D([0], [0], color=COLOR_MEDIAN, lw=2.5, label='Median'),
        Line2D([0], [0], marker='D', color='w', markeredgecolor=COLOR_MEAN, 
               markerfacecolor='w', markersize=6, lw=0, label='Mean'),
        Line2D([0], [0], color=COLOR_MEAN, lw=2.5, label='Std Dev'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_DOTS, 
               markeredgecolor='white', markersize=8, label='Raw Data'),
        mpatches.Patch(facecolor=COLOR_FAIL, alpha=0.4, label='Failure Prob')
    ]
    
    return legend_elements

def plot_config_performance_comparison(
    runs: List[SweepConfigResult],
    title: str = 'Experiment Results',
    show_means: bool = True,
    show_std: bool = True,
    figsize: Tuple[float, float] = (8, 6),
    ylim: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """Plot results from a single experiment."""
    try:
        style = 'seaborn-v0_8-whitegrid'
        plt.style.use(style)
    except OSError:
        style = 'ggplot'
        
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)
        ax2 = ax.twinx()
        
        legend_elements = _plot_config_performance_comparison_single(
            ax, ax2, runs, show_means, show_std, ylim, title
        )
        
        ax.legend(
            handles=legend_elements, 
            loc='upper left', 
            bbox_to_anchor=(1.08, 1), 
            frameon=False,
            title="Legend"
        )
        
        plt.tight_layout()
        if save_path:
            metadata = [r.config_meta for r in runs]
            _save_figure_with_metadata(fig, save_path, dpi, metadata)
        plt.show()

def plot_config_performance_comparison_multi(
    runs_grid: Optional[Any] = None,
    titles_grid: Optional[Any] = None,
    list_of_runs: Optional[List[List[SweepConfigResult]]] = None,
    list_of_titles: Optional[List[str]] = None,
    horizontal: Optional[bool] = None,
    show_means: bool = True,
    show_std: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    y_max: Optional[float] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    legend_pos: str = 'inside',
):
    """
    Plot multiple experiment results in a 2D subplot grid with a shared legend.
    
    Args:
        runs_grid: 2D array-like where each entry is a list of
            ConfigComparisonExperimentRun objects for one subplot.
        titles_grid: Optional 2D array-like of subplot titles with same shape as runs_grid.
        list_of_runs: Legacy 1D layout input (deprecated). If provided and
            runs_grid is None, it will be converted into a 2D grid.
        list_of_titles: Legacy 1D titles input (deprecated).
        horizontal: Legacy layout switch for 1D input conversion.
        show_means: Show mean markers
        show_std: Show standard deviation error bars
        figsize: Figure size tuple (width, height). If None, auto-calculated
        y_max: Maximum y-axis value. If None, auto-calculated from data
        save_path: Path to save the plot to
    """
    try:
        style = 'seaborn-v0_8-whitegrid'
        plt.style.use(style)
    except OSError:
        style = 'ggplot'
    
    if runs_grid is None and list_of_runs is not None:
        if horizontal is False:
            runs_grid = [[runs] for runs in list_of_runs]
            if list_of_titles is not None:
                titles_grid = [[title] for title in list_of_titles]
        else:
            runs_grid = [list_of_runs]
            if list_of_titles is not None:
                titles_grid = [list_of_titles]

    if runs_grid is None:
        raise ValueError("runs_grid is required")

    if isinstance(runs_grid, np.ndarray):
        if runs_grid.ndim != 2:
            raise ValueError("runs_grid must be a 2D array-like structure")
        normalized_runs_grid = runs_grid.astype(object, copy=False)
    else:
        rows = list(runs_grid)
        if len(rows) == 0:
            raise ValueError("runs_grid must not be empty")
        row_lengths = [len(row) for row in rows]
        if any(length != row_lengths[0] for length in row_lengths):
            raise ValueError("runs_grid rows must have the same length")
        normalized_runs_grid = np.empty((len(rows), row_lengths[0]), dtype=object)
        for row_idx, row in enumerate(rows):
            for col_idx, runs in enumerate(row):
                normalized_runs_grid[row_idx, col_idx] = runs

    runs_grid = normalized_runs_grid

    n_rows, n_cols = runs_grid.shape

    if titles_grid is not None:
        if isinstance(titles_grid, np.ndarray):
            titles_grid = titles_grid.astype(object, copy=False)
        else:
            title_rows = list(titles_grid)
            titles_grid_arr = np.empty((len(title_rows), len(title_rows[0]) if title_rows else 0), dtype=object)
            for row_idx, row in enumerate(title_rows):
                if len(row) != titles_grid_arr.shape[1]:
                    raise ValueError("titles_grid rows must have the same length")
                for col_idx, title in enumerate(row):
                    titles_grid_arr[row_idx, col_idx] = title
            titles_grid = titles_grid_arr
        if titles_grid.shape != runs_grid.shape:
            raise ValueError("titles_grid must have the same shape as runs_grid")
    
    # Calculate shared y-axis limits across all result sets to ensure comparable scales
    all_success_data = []
    for runs in runs_grid.flat:
        for r in runs:
            if len(r.successes) > 0:
                all_success_data.extend(r.successes)
    
    if len(all_success_data) > 0:
        y_min = 0  # Always start at 0 for generation count
        if y_max is None:
            y_max = max(all_success_data) * 1.1  # Add 10% padding at top
        shared_ylim = (y_min, y_max)
    else:
        shared_ylim = None
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (6 * n_cols, 4.5 * n_rows)
    
    with plt.style.context(style):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        
        legend_elements = None

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax2 = ax.twinx()
                runs = runs_grid[row_idx, col_idx]

                title = titles_grid[row_idx, col_idx] if titles_grid is not None else None

                legend_elements = _plot_config_performance_comparison_single(
                    ax, ax2, runs, show_means, show_std, shared_ylim, title
                )
        
        # Add exactly one legend based on legend_pos
        if legend_elements:
            legend_ax = axes[0, n_cols - 1]
            if legend_pos == 'inside':
                legend_ax.legend(
                    handles=legend_elements,
                    loc='upper right',
                    frameon=True,
                    title="Legend",
                    fontsize='small',
                    framealpha=0.95
                )
            elif legend_pos == 'outside':
                # Use bbox_to_anchor bounded to the uppermost right axes, NOT the figure.
                # Shift x by ~1.15 to clear the secondary y-axis label ("Failure Probability").
                legend_ax.legend(
                    handles=legend_elements,
                    loc='center left',
                    bbox_to_anchor=(1.18, 0.5), 
                    frameon=True,
                    title="Legend",
                )
        
        plt.tight_layout()
        if save_path:
            list_of_metadata = [[[r.config_meta for r in runs] for runs in row] for row in runs_grid.tolist()]
            _save_figure_with_metadata(fig, save_path, dpi, list_of_metadata)
        plt.show()
