import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_solution(solution, mapping=None):
    scheme = solution.generate_schedule_scheme()
    plot_schedules(scheme, mapping=mapping)


def plot_schedules(schedules, titles=None, colormap_name="viridis", job_id_mapping=None, mapping=None):
    if not schedules:
        return

    # Support legacy 'mapping' kwarg
    if job_id_mapping is None and mapping is not None:
        job_id_mapping = mapping

    num_schedules = len(schedules)
    _, num_times = schedules[0].shape

    fig, axes = plt.subplots(
        num_schedules, 1, figsize=(15, 2 * num_schedules + 1), squeeze=False
    )

    all_job_ids = []
    for schedule in schedules:
        if schedule.ndim == 2:
            all_job_ids.extend(np.unique(schedule[schedule != -1]))

    if not all_job_ids:
        plt.show()
        return

    actual_unique_job_ids = np.unique(all_job_ids)
    num_distinct_jobs = len(actual_unique_job_ids)

    if num_distinct_jobs == 0:
        plt.show()
        return

    job_id_to_idx = {job_id: i for i, job_id in enumerate(actual_unique_job_ids)}

    color_palette = plt.get_cmap(colormap_name, num_distinct_jobs)

    permuted_indices = job_id_mapping
    if permuted_indices is None:
        permuted_indices = np.random.permutation(num_distinct_jobs)
    elif len(permuted_indices) != num_distinct_jobs:
        permuted_indices = np.random.permutation(num_distinct_jobs)

    for k, schedule_data in enumerate(schedules):
        ax = axes[k, 0]

        if schedule_data.ndim != 2:
            ax.text(0.5, 0.5, "Invalid schedule data", ha="center", va="center")
            continue

        num_resources = schedule_data.shape[0]

        for i in range(num_resources):
            for j in range(num_times):
                job_id = schedule_data[i, j]
                if job_id != -1 and job_id in job_id_to_idx:
                    base_idx = job_id_to_idx[job_id]
                    color_idx = permuted_indices[base_idx]
                    color = color_palette(color_idx)
                    ax.add_patch(
                        patches.Rectangle(
                            (j, i), 1, 1,
                            facecolor=color,
                            edgecolor="gray",
                            linewidth=0.25,
                        )
                    )

        ax.set_xlabel("Time")
        ax.set_ylabel(f"Resource {k + 1}")
        if titles is not None and k < len(titles):
            ax.set_title(titles[k])

        ax.set_xlim(0, num_times)
        ax.set_ylim(0, num_resources if num_resources > 0 else 1)
        ax.set_xticks(np.arange(0, num_times + 1, 1))
        ax.set_yticks(np.arange(0, num_resources + 1, 1))
        ax.set_yticklabels([])
        ax.grid(True, which="both", linestyle="-", linewidth=0.2, alpha=0.7)

        if num_resources > 0:
            ax.invert_yaxis()

    # Legend
    legend_handles = []
    try:
        sorted_ids = sorted(
            [jid.item() if hasattr(jid, "item") else jid for jid in actual_unique_job_ids]
        )
    except TypeError:
        sorted_ids = list(actual_unique_job_ids)

    for job_id in sorted_ids:
        if job_id not in job_id_to_idx:
            continue
        base_idx = job_id_to_idx[job_id]
        color_idx = permuted_indices[base_idx]
        color = color_palette(color_idx)
        legend_handles.append(
            patches.Patch(facecolor=color, edgecolor="gray", label=f"Job {job_id}")
        )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            title="Job Legend",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        fig.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        fig.tight_layout()

    plt.show()
