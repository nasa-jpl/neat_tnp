import matplotlib.artist
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
from numpy.typing import NDArray
from typing import List, Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo

style = 'seaborn-v0_8-whitegrid'
plt.style.use(style)

# --- Metadata & Saving Utilities ---

def _read_png_metadata(path: str) -> dict:
    with Image.open(path) as img:
        if "custom_json" not in img.info:
            return {}
        return json.loads(img.info["custom_json"])

def _write_png_metadata(path: str, data: str):
    img = Image.open(path)
    metadata = PngInfo()
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
    fig.savefig(path, dpi=dpi)
    
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

def plot_graph_on_ax(
    ax: plt.Axes, 
    graph: PlotGraph, 
    title: str,
    setup_axis: bool = True,
    show_ids: bool = False,
    show_blobs: bool = True,
    show_legend: bool = True,
    show_grid: bool = False
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
                   vmin=0, vmax=max(1, np.max(cost_map)), zorder=1)
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
        plot_rows, plot_cols = np.where(highlight_map > 0)
        
        if len(plot_rows) > 0:
            # 1. Get alpha values (normalized)
            hl_values_nonzero = highlight_map[plot_rows, plot_cols].astype(float)
            
            hl_range = (0.3, 0.6) # Min alpha of 0.1
            hl_min, hl_max = hl_values_nonzero.min(), hl_values_nonzero.max()
            hl_denom = hl_max - hl_min
            
            if hl_denom > 0:
                hl_values_nonzero -= hl_min
                hl_values_nonzero *= (hl_range[1]-hl_range[0]) / hl_denom
                hl_values_nonzero += hl_range[0]
            else:
                hl_values_nonzero.fill(np.mean(hl_range))
            
            # 2. Create an (N, 4) RGBA color array
            highlight_col = [0.0, 0.3, 0.9]
            colors = np.zeros((len(plot_rows), 4))
            colors[:, 0:3] = highlight_col  # Set R, G, B
            colors[:, 3] = hl_values_nonzero # Set Alpha
            
            # 3. Create a list of Rectangle patches
            # A 1x1 cell is centered at (col, row). It spans (col-0.5) to (col+0.5).
            # We want a smaller patch, e.g., 80% (0.8x0.8).
            # So we start at (col - 0.4) and (row - 0.4).
            patch_size = 0.8
            offset = patch_size / 2.0
            patch_list = []
            for r, c in zip(plot_rows, plot_cols):
                # Bottom-left corner is (c - offset, r - offset)
                rect = patches.Rectangle((c - offset, r - offset), patch_size, patch_size)
                patch_list.append(rect)

            # 4. Create the collection
            collection = PatchCollection(
                patch_list,
                facecolors=colors,
                zorder=2
            )
            
            # 5. Add the collection to the axis
            ax.add_collection(collection)
            all_artists.append(collection)

            # 6. Add the dummy artist for the legend
            sc_dummy = ax.scatter([], [], c=highlight_col, marker='s', s=100, 
                                   alpha=0.6, label='Activated Cells', zorder=-1)
            all_artists.append(sc_dummy)

    # 6. Plot formatting (if requested)
    if setup_axis:
        ax.set_xlabel('Column Index (x)')
        ax.set_ylabel('Row Index (y)')
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
    show_grid: bool = False
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
    
    plot_graph_on_ax(ax, graph, title, show_ids=show_ids, show_blobs=show_blobs, show_grid=show_grid)
    
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
    show_grid: bool = False
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
            plot_graph_on_ax(ax, graph_list[i], current_titles[i], show_ids=show_ids, show_blobs=show_blobs, show_legend=False, show_grid=show_grid)
        else:
            ax.axis('off')
            
    # Create unified legend
    handles_dict = {}
    for i in range(n_graphs):
        h, l = axes_flat[i].get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in handles_dict:
                handles_dict[label] = handle
                
    if handles_dict:
        fig.legend(handles_dict.values(), handles_dict.keys(), loc='lower center', ncol=len(handles_dict), bbox_to_anchor=(0.5, 0.00))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        plt.tight_layout()
    
    if save_path:
         _save_figure_with_metadata(fig, save_path, dpi, metadata)
         
    plt.show()


def plot_graph_evolution(graph_list: List[PlotGraph], 
                           titles: Optional[List[str]] = None,
                           show_ids: bool = False,
                           show_blobs: bool = True,
                           show_grid: bool = False):
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
        current_artists = plot_graph_on_ax(ax, graph, title, setup_axis=False, show_ids=show_ids, show_blobs=show_blobs, show_grid=show_grid)
        
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
    current_artists = plot_graph_on_ax(ax, initial_graph, initial_title, show_ids=show_ids, show_blobs=show_blobs, show_grid=show_grid)
    
    plt.show()

@dataclass
class RunStatsPlot:
    """
    An algorithm-agnostic class for plotting statistics of evolutions for
    the evlutionary algorithms we consider.
    """
    title: str
    # the metrics that are to be visualized, e.g., "avg fitness"
    # each single metric should get its own line
    # TODO: add some styling control here
    metrics: List[str]
    # for each generation, a dict of metrics with values
    values: List[Dict[str, float]]

def plot_training(
    runs: List[RunStatsPlot], 
    title: str = "Evolution Performance",
    use_time_axis: bool = False,
    legend: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    metadata: Optional[Any] = None
):
    """
    Plots the training statistics for one or more algorithm runs.
    
    This function creates one subplot for each unique metric
    (e.g., "avg_fitness", "max_fitness") and plots the data
    from all provided runs on that subplot for comparison.
    
    The order of the subplots is determined by the order of metrics
    in the first run object (runs[0]).

    Args:
        runs: A list of RunStatsPlot data objects.
        title: The overall title for the figure.
        use_time_axis: If True, uses the "time" metric from the
                       data for the x-axis instead of "Generation".
    """
    
    # 1. Determine the order and list of metrics to plot
    if not runs:
        print("No runs to plot.")
        return

    # Use the first run's metrics (excluding "time") as the base order
    base_ordered_metrics = [m for m in runs[0].metrics if m != "time"]
    base_metrics_set = set(base_ordered_metrics)

    # Collect all unique metrics from all runs (excluding "time")
    all_metrics_set: Set[str] = set()
    for run in runs:
        all_metrics_set.update(m for m in run.metrics if m != "time")
        
    # Create the final list of metrics to plot
    # Start with the base order
    final_plot_metrics = list(base_ordered_metrics)
    
    # Add any metrics that were in other runs but not the first one
    # We sort the remaining set to ensure a consistent, non-random order
    # for these "extra" metrics.
    for metric in sorted(all_metrics_set):
        if metric not in base_metrics_set:
            final_plot_metrics.append(metric)

    n_metrics = len(final_plot_metrics)

    if n_metrics == 0:
        print("No metrics to plot (excluding 'time').")
        return

    # 2. Create a figure with a subplot for each metric
    fig, axes = plt.subplots(
        n_metrics, 1, 
        figsize=(8, 2 * n_metrics),
        sharex=True, 
        squeeze=False
    )

    # Create a simple mapping from metric name to its axis
    # This now uses the new final_plot_metrics list for ordering
    ax_map = {metric: axes[i, 0] for i, metric in enumerate(final_plot_metrics)}
    
    # Determine the x-axis label based on the toggle
    x_label = "Time (seconds)" if use_time_axis else "Generation"

    # 3. Plot data for each run
    for run in runs:
        
        # Determine the x-axis values for this run
        if use_time_axis:
            if "time" not in run.metrics:
                print(f"Warning: 'use_time_axis' is True but "
                      f"run '{run.title}' is missing 'time' metric. Skipping run.")
                continue
            x_values = [gen_data.get("time", np.nan) for gen_data in run.values]
        else:
            x_values = range(len(run.values))
        
        for metric_name in run.metrics:
            if metric_name in ax_map:
                ax = ax_map[metric_name]
                y_values = [
                    gen_data.get(metric_name, np.nan) 
                    for gen_data in run.values
                ]
                ax.plot(
                    x_values,
                    y_values, 
                    label=run.title, 
                    marker='.', 
                    markersize=4, 
                    alpha=0.8
                )

    # 4. Format all subplots
    # This loop also respects the new order from final_plot_metrics
    for i, metric_name in enumerate(final_plot_metrics):
        ax = axes[i, 0]
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.set_ylabel("Value")
        if legend:
            ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # 5. Set shared X-axis label on the bottom plot
    axes[-1, 0].set_xlabel(x_label)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        _save_figure_with_metadata(fig, save_path, dpi, metadata)
        
    plt.show()

def evolution_video(
    graph_list: List[PlotGraph],
    filename: str = "evolution.mp4",
    fps: int = 5,
    titles: Optional[List[str]] = None,
    dpi: int = 300,
    show_blobs: bool = True
):
    """
    Generates a video (MP4 or GIF) from a list of PlotGraph objects.

    Args:
        graph_list: The sequence of graphs to animate.
        filename: Output filename (e.g., 'video.mp4' or 'animation.gif').
        fps: Frames per second.
        titles: Optional list of titles for each frame.
        dpi: Dots per inch for the output video resolution.
        show_blobs: If True (default), display flexible nodes as green circles.
    """
    if not graph_list:
        print("Graph list is empty. Cannot generate video.")
        return

    n_graphs = len(graph_list)

    # --- Handle Titles ---
    if titles is None:
        plot_titles = [f"Generation {i}" for i in range(n_graphs)]
    elif len(titles) != n_graphs:
        print(f"Warning: Title count ({len(titles)}) mismatch with graphs ({n_graphs}).")
        plot_titles = [f"Graph {i}" for i in range(n_graphs)]
    else:
        plot_titles = titles

    # --- Setup Figure ---
    fig, ax = plt.subplots(figsize=(8, 4))

    print(f"Generating video '{filename}' with {n_graphs} frames...")

    # Progress helpers
    last_reported = {"frame": -1}
    report_every = max(1, n_graphs // 50)  # ~50 updates max

    def update(frame_idx):
        # Plot frame
        graph = graph_list[frame_idx]
        title = plot_titles[frame_idx]
        plot_graph_on_ax(ax, graph, title, setup_axis=True, show_blobs=show_blobs)

        # Progress output
        if (
            frame_idx == 0
            or frame_idx == n_graphs - 1
            or (frame_idx - last_reported["frame"]) >= report_every
        ):
            pct = 100.0 * (frame_idx + 1) / n_graphs
            print(f"Rendering frame {frame_idx + 1}/{n_graphs} ({pct:5.1f}%)")
            last_reported["frame"] = frame_idx

        return []

    # --- Create Animation ---
    anim = FuncAnimation(
        fig,
        update,
        frames=n_graphs,
        interval=1000 / fps,  # interval is in milliseconds
        blit=False
    )

    # --- Save to File ---
    try:
        if filename.lower().endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(filename, writer=writer, dpi=dpi)
        else:
            # Assumes ffmpeg is installed for MP4
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=fps*40)
            anim.save(filename, writer=writer, dpi=dpi)

        print(f"Successfully saved video to {filename}")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Ensure ffmpeg is installed for .mp4, or try saving as .gif")
    finally:
        plt.close(fig)

def calculate_pareto_mask_maximization(data: np.ndarray) -> np.ndarray:
    """
    Finds the non-dominated set (Pareto front) for a Maximization problem.
    Returns a boolean mask where True = Pareto efficient.
    """
    n_points = data.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        # Compare point i with all other points
        # A point is dominated if another point is:
        # >= in all objectives AND > in at least one objective
        all_greater_or_equal = np.all(data >= data[i], axis=1)
        any_strictly_greater = np.any(data > data[i], axis=1)
        
        is_dominated = np.any(all_greater_or_equal & any_strictly_greater)
        
        if is_dominated:
            is_pareto[i] = False
            
    return is_pareto

def plot_pareto_fronts(
    population_objectives: List[Tuple[float, ...]],
    objective_labels: Optional[List[str]] = None,
    title: str = "Pareto Front (Maximization)"
):
    if not population_objectives:
        print("No population data to plot.")
        return

    # --- Data Prep ---
    data = np.array(population_objectives)
    n_points, n_dims = data.shape
    
    # Calculate Pareto Front (Maximization logic)
    pareto_mask = calculate_pareto_mask_maximization(data)

    # Handle Labels
    if objective_labels is None:
        objective_labels = [f"Obj {i+1}" for i in range(n_dims)]
    elif len(objective_labels) != n_dims:
        objective_labels = [f"Obj {i+1}" for i in range(n_dims)]

    # --- Setup Figure ---
    # Reduced figure size (was 10,6)
    fig = plt.figure(figsize=(6, 4))
    
    # Styles
    # Reduced point sizes (s)
    style_dominated = {
        'c': 'gray', 'alpha': 0.3, 's': 10, 'marker': 'o', 'label': 'Dominated'
    }
    style_pareto = {
        'c': 'firebrick', 'alpha': 1.0, 's': 30, 'marker': 'o', 'label': 'Pareto Front'
    }

    # Determine Plot Type
    if n_dims == 2:
        ax = fig.add_subplot(111)
        
        # Plot Dominated
        ax.scatter(data[~pareto_mask, 0], data[~pareto_mask, 1], **style_dominated)
        # Plot Pareto
        ax.scatter(data[pareto_mask, 0], data[pareto_mask, 1], **style_pareto)
        
        ax.set_xlabel(objective_labels[0])
        ax.set_ylabel(objective_labels[1])
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Move legend to outside/top or keep small
        ax.legend(fontsize='small')

    elif n_dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Dominated
        ax.scatter(data[~pareto_mask, 0], data[~pareto_mask, 1], data[~pareto_mask, 2], 
                   **style_dominated)
        # Plot Pareto
        ax.scatter(data[pareto_mask, 0], data[pareto_mask, 1], data[pareto_mask, 2], 
                   **style_pareto)
        
        ax.set_xlabel(objective_labels[0])
        ax.set_ylabel(objective_labels[1])
        ax.set_zlabel(objective_labels[2])
        ax.legend(fontsize='small')

    else:
        # Parallel Coordinates
        ax = fig.add_subplot(111)
        x_range = range(n_dims)
        
        # Dominated lines
        for row in data[~pareto_mask]:
            ax.plot(x_range, row, c='gray', alpha=0.15, linewidth=0.8)
            
        # Pareto lines
        for row in data[pareto_mask]:
            ax.plot(x_range, row, c='firebrick', alpha=0.9, linewidth=1.5)

        ax.set_xticks(x_range)
        ax.set_xticklabels(objective_labels)
        ax.grid(True, axis='x', alpha=0.5)
        ax.set_ylabel("Objective Value")

    ax.set_title(title, fontsize=10)
    plt.tight_layout()
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
    runs: List[ConfigComparisonExperimentRun],
    show_means: bool = True,
    show_std: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> List[matplotlib.artist.Artist]:
    """
    Core plotting function for a single experiment result on given axes.
    Returns legend elements for later use.
    """
    # Extract data from runs objects
    success_results = [np.array(r.successes) for r in runs]
    failure_probabilities = [r.failure_prob for r in runs]
    labels = [r.label for r in runs]

    x = np.arange(1, len(success_results) + 1)
    
    # Colors
    COLOR_DOTS = '#1f77b4'
    COLOR_MEAN = '#d62728'
    COLOR_MEDIAN = '#2ca02c'
    COLOR_FAIL = '#555555'
    
    # --- Plot Data (Primary Axis) ---
    
    # A. Boxplot
    ax.boxplot(
        success_results, 
        positions=x, 
        showfliers=False, 
        patch_artist=True,
        boxprops=dict(facecolor='#EBF5FB', color=COLOR_DOTS, alpha=0.5, linewidth=1),
        medianprops=dict(color=COLOR_MEDIAN, linewidth=2.5),
        whiskerprops=dict(visible=False),
        capprops=dict(visible=False),
        widths=0.6,
        zorder=1
    )
    
    # B. Scatter points
    for i, r in enumerate(success_results):
        if len(r) == 0: continue
        jitter = np.random.normal(0, 0.05, size=len(r))
        ax.scatter(
            np.full_like(r, x[i]) + jitter, 
            r, 
            color=COLOR_DOTS, 
            alpha=0.6,
            s=25,
            edgecolors='white',
            linewidth=0.8,
            zorder=3
        )
    
    # C. Mean and Standard Deviation
    if show_means or show_std:
        means = [r.mean() if len(r) > 0 else np.nan for r in success_results]
        
        if show_std:
            stds = [r.std() if len(r) > 0 else 0.0 for r in success_results]
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
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    
    ax.set_xlabel('Algorithm Configurations', fontsize=11)
    
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
    runs: List[ConfigComparisonExperimentRun],
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
            metadata = [r.meta for r in runs]
            _save_figure_with_metadata(fig, save_path, dpi, metadata)
        plt.show()

def plot_config_performance_comparison_multi(
    list_of_runs: List[List[ConfigComparisonExperimentRun]],
    list_of_titles: Optional[List[str]] = None,
    horizontal: bool = True,
    show_means: bool = True,
    show_std: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    y_max: Optional[float] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """
    Plot multiple experiment results in a single figure with shared legend.
    
    Args:
        list_of_runs: List of lists of ConfigComparisonExperimentRun objects (one list per subplot)
        list_of_titles: List of titles (one per subplot)
        horizontal: If True, arrange subplots horizontally; if False, vertically
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
    
    n_plots = len(list_of_runs)
    
    # Calculate shared y-axis limits across all result sets to ensure comparable scales
    all_success_data = []
    for runs in list_of_runs:
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
        if horizontal:
            figsize = (8 * n_plots, 6)
        else:
            figsize = (10, 5 * n_plots)
            
    with plt.style.context(style):
        if horizontal:
            fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        else:
            fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        
        # Ensure axes is iterable
        if n_plots == 1:
            axes = [axes]
        
        legend_elements = None
    
        for i in range(n_plots):
            ax = axes[i]
            ax2 = ax.twinx()
            
            runs = list_of_runs[i]
            
            title = list_of_titles[i] if list_of_titles else None
            
            legend_elements = _plot_config_performance_comparison_single(
                ax, ax2, runs, show_means, show_std, shared_ylim, title
            )
        
        # Add legend inside the last data plot
        if legend_elements and n_plots > 0:
            axes[n_plots - 1].legend(
                handles=legend_elements,
                loc='upper right',
                frameon=True,
                title="Legend",
                fontsize='small',
                framealpha=0.95
            )
        
        plt.tight_layout()
        if save_path:
            list_of_metadata = [[r.meta for r in runs] for runs in list_of_runs]
            _save_figure_with_metadata(fig, save_path, dpi, list_of_metadata)
        plt.show()
