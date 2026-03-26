from cmath import rect

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import mplcursors

from matplotlib import rcParams
from config.settings import (
    TITLE, TITLE_SIZE, TITLE_FONT_WEIGHT,
    FONT_FAMILY, FONT_SANS_SERIF, FONT_COLOR,
    LABEL_SIZE, DAY_FONT_SIZE, MONTH_FONT_SIZE, MONTH_FONT_WEIGHT,
    X_LABEL, Y_LABEL,
    BAR_COLOR, DATE_FORMAT
)

# =========================
# OPTIONAL TEMPLATE COLORS
# =========================
TEMPLATE_COLORS = {
    "CES": "#56778f",
    "SD": "#91be6f",
    "FT": "#ff6e61"
}

# =========================
# STYLE CONFIG
# =========================
rcParams['font.family'] = FONT_FAMILY
rcParams['font.sans-serif'] = FONT_SANS_SERIF
rcParams['axes.titlesize'] = TITLE_SIZE
rcParams['axes.labelsize'] = LABEL_SIZE


# =========================
# LOADING FUNCTIONS
# =========================
def load_project_sheet(file_path, sheet_name="PROJECT"):
    """
    Load PROJECT sheet.

    Supports either:
    - deadline mode
    - startline mode
    - mixed mode

    Required base columns:
    - project_name
    - template_name

    Optional scheduling columns:
    - deadline
    - startline

    Optional:
    - buffer_days
    """
    projects = pd.read_excel(file_path, sheet_name=sheet_name)
    projects.columns = [str(c).strip() for c in projects.columns]

    # Base required columns
    required_base = ["project_name", "template_name"]
    missing_base = [c for c in required_base if c not in projects.columns]
    if missing_base:
        raise ValueError(f"Missing required columns in PROJECT sheet: {missing_base}")

    # At least one scheduling anchor must exist
    if "deadline" not in projects.columns and "startline" not in projects.columns:
        raise ValueError(
            "PROJECT sheet must contain at least one of: 'deadline' or 'startline'"
        )

    projects = projects.dropna(how="all").copy()
    projects = projects[projects["project_name"].notna()].copy()

    projects["project_name"] = projects["project_name"].astype(str).str.strip()
    projects["template_name"] = projects["template_name"].astype(str).str.strip()

    # Optional columns
    if "deadline" in projects.columns:
        projects["deadline"] = pd.to_datetime(projects["deadline"], errors="coerce")
    else:
        projects["deadline"] = pd.NaT

    if "startline" in projects.columns:
        projects["startline"] = pd.to_datetime(projects["startline"], errors="coerce")
    else:
        projects["startline"] = pd.NaT

    if "buffer_days" in projects.columns:
        projects["buffer_days"] = (
            pd.to_numeric(projects["buffer_days"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        projects["buffer_days"] = 0

    # Row-level validation: each row must have either deadline or startline
    invalid_rows = projects[
        projects["deadline"].isna() & projects["startline"].isna()
    ]

    if not invalid_rows.empty:
        raise ValueError(
            "Some PROJECT rows have neither 'deadline' nor 'startline':\n"
            + invalid_rows[["project_name", "template_name"]].to_string(index=False)
        )

    return projects


def load_template_sheet(file_path, template_name):
    tasks = pd.read_excel(file_path, sheet_name=template_name)
    tasks.columns = [str(c).strip() for c in tasks.columns]

    required = ["task_id", "dependencies", "task_group", "task_description", "duration_days", "role"]
    missing = [c for c in required if c not in tasks.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet '{template_name}': {missing}")

    tasks = tasks.dropna(how="all").copy()
    tasks = tasks[tasks["task_id"].notna()].copy()

    tasks["task_id"] = pd.to_numeric(tasks["task_id"], errors="coerce")
    if tasks["task_id"].isna().any():
        bad_rows = tasks[tasks["task_id"].isna()]
        raise ValueError(f"Some rows in sheet '{template_name}' have invalid task_id:\n{bad_rows}")
    tasks["task_id"] = tasks["task_id"].astype(int)

    tasks["dependencies"] = tasks["dependencies"].fillna("")
    tasks["task_group"] = tasks["task_group"].fillna("").astype(str).str.strip()
    tasks["task_description"] = tasks["task_description"].fillna("").astype(str).str.strip()
    tasks["role"] = tasks["role"].fillna("").astype(str).str.strip()
    tasks["duration_days"] = pd.to_numeric(tasks["duration_days"], errors="coerce")

    if tasks["duration_days"].isna().any():
        bad_rows = tasks[tasks["duration_days"].isna()]
        raise ValueError(
            f"Some rows in sheet '{template_name}' have invalid or blank duration_days:\n{bad_rows}"
        )

    tasks["duration_days"] = tasks["duration_days"].astype(int)

    return tasks


# =========================
# DEPENDENCY UTILITIES
# =========================
def parse_dependencies(dep_value):
    if pd.isna(dep_value) or str(dep_value).strip() == "":
        return []

    dep_str = str(dep_value).strip()

    try:
        deps = []
        for x in dep_str.split(","):
            x = x.strip()
            if x == "":
                continue
            deps.append(int(float(x)))  # handles 3.0 from Excel
        return deps
    except Exception:
        raise ValueError(f"Invalid dependency format: '{dep_value}'")


def validate_dependencies(tasks):
    task_ids = set(tasks["task_id"].tolist())

    for _, row in tasks.iterrows():
        deps = parse_dependencies(row["dependencies"])
        for dep in deps:
            if dep not in task_ids:
                raise ValueError(
                    f"Task {row['task_id']} depends on missing task_id {dep}"
                )


# =========================
# CORE SCHEDULING ENGINE
# =========================
def calculate_forward_schedule(tasks, start_date="2026-01-01"):
    """
    Forward dependency-only scheduler.

    Logic:
    - Tasks start as early as possible
    - Parallel branches are allowed
    - No resource/capacity constraints
    """
    tasks = tasks.copy().sort_values("task_id").reset_index(drop=True)
    validate_dependencies(tasks)

    start_date = pd.to_datetime(start_date)

    task_map = {row["task_id"]: row for _, row in tasks.iterrows()}
    dependencies = {
        row["task_id"]: parse_dependencies(row["dependencies"])
        for _, row in tasks.iterrows()
    }

    start_dates = {}
    end_dates = {}

    def compute_task_dates(task_id):
        if task_id in start_dates:
            return start_dates[task_id], end_dates[task_id]

        deps = dependencies[task_id]
        duration = int(task_map[task_id]["duration_days"])

        if not deps:
            task_start = start_date
        else:
            dep_end_dates = []
            for dep in deps:
                _, dep_end = compute_task_dates(dep)
                dep_end_dates.append(dep_end)

            task_start = max(dep_end_dates) + pd.Timedelta(days=1)

        task_end = task_start + pd.Timedelta(days=duration - 1)

        start_dates[task_id] = task_start
        end_dates[task_id] = task_end

        return task_start, task_end

    for tid in tasks["task_id"]:
        compute_task_dates(tid)

    tasks["start_date"] = tasks["task_id"].map(start_dates)
    tasks["end_date"] = tasks["task_id"].map(end_dates)

    return tasks.sort_values(by=["start_date", "task_id"]).reset_index(drop=True)


def calculate_backward_anchored_schedule(tasks, deadline, buffer_days=0):
    """
    Best scheduling logic for your use case:
    1. Build earliest-possible dependency schedule
    2. Shift the whole workflow backward so final finish = deadline - buffer
    """
    tasks = calculate_forward_schedule(tasks, start_date="2026-01-01")

    deadline = pd.to_datetime(deadline)
    target_finish = deadline - pd.Timedelta(days=buffer_days)

    current_finish = tasks["end_date"].max()
    shift_days = (target_finish - current_finish).days

    tasks["start_date"] = tasks["start_date"] + pd.Timedelta(days=shift_days)
    tasks["end_date"] = tasks["end_date"] + pd.Timedelta(days=shift_days)

    return tasks.sort_values(by=["start_date", "task_id"]).reset_index(drop=True)

def calculate_forward_from_startline(tasks, startline, buffer_days=0):
    """
    Forward scheduling anchored to a given startline.

    Logic:
    - Tasks start as early as possible
    - Dependencies respected
    - Parallel branches allowed
    - Buffer shifts actual work start forward
    """
    startline = pd.to_datetime(startline)
    effective_start = startline + pd.Timedelta(days=buffer_days)

    scheduled = calculate_forward_schedule(tasks, start_date=effective_start)

    return scheduled.sort_values(by=["start_date", "task_id"]).reset_index(drop=True)

# =========================
# PROJECT BUILDERS
# =========================
def build_project_schedule_backward(file_path, project_row):
    project_name = project_row["project_name"]
    template_name = project_row["template_name"]
    deadline = project_row["deadline"]
    buffer_days = project_row["buffer_days"]

    tasks = load_template_sheet(file_path, template_name)
    scheduled = calculate_backward_anchored_schedule(tasks, deadline, buffer_days)

    scheduled["project_name"] = project_name
    scheduled["template_name"] = template_name
    scheduled["deadline"] = deadline
    scheduled["buffer_days"] = buffer_days

    cols = [
        "project_name", "template_name", "task_id", "dependencies", "task_group",
        "task_description", "duration_days", "role", "start_date", "end_date",
        "deadline", "buffer_days"
    ]
    return scheduled[cols]

def build_project_schedule_forward(file_path, project_row):
    """
    Build schedule using startline instead of deadline.
    """
    project_name = project_row["project_name"]
    template_name = project_row["template_name"]
    startline = project_row["startline"]
    buffer_days = project_row.get("buffer_days", 0)

    tasks = load_template_sheet(file_path, template_name)
    scheduled = calculate_forward_from_startline(tasks, startline, buffer_days)

    scheduled["project_name"] = project_name
    scheduled["template_name"] = template_name
    scheduled["startline"] = startline
    scheduled["buffer_days"] = buffer_days

    cols = [
        "project_name", "template_name", "task_id", "dependencies", "task_group",
        "task_description", "duration_days", "role",
        "start_date", "end_date", "startline", "buffer_days"
    ]

    return scheduled[cols]


def build_master_schedule(file_path):
    projects = load_project_sheet(file_path)
    all_projects = []

    for _, row in projects.iterrows():

        if pd.notna(row.get("deadline")):
            schedule = build_project_schedule(file_path, row)

        elif pd.notna(row.get("startline")):
            schedule = build_project_schedule_forward(file_path, row)

        else:
            raise ValueError(
                f"Project '{row.get('project_name')}' / template '{row.get('template_name')}' "
                f"must have either deadline or startline."
            )

        all_projects.append(schedule)

    master = pd.concat(all_projects, ignore_index=True)

    return master.sort_values(by=["start_date", "project_name", "task_id"]).reset_index(drop=True)


# =========================
# GANTT HELPERS
# =========================
def build_week_ticks(start_date, end_date):
    mondays = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    return mondays, [d.strftime('%d') for d in mondays]


def prepare_gantt_labels(tasks):
    tasks = tasks.copy()
    tasks["short_label"] = (
        tasks["template_name"] + "-" +
        tasks["task_id"].astype(str).str.zfill(2) + " " +
        tasks["task_group"]
    )
    return tasks

def build_schedule_anchor_text(task):
    """
    Safely builds schedule anchor info for hover annotation.
    Supports both deadline mode and startline mode.
    """
    lines = []

    if "deadline" in task.index and pd.notna(task["deadline"]):
        lines.append(f"Deadline: {pd.to_datetime(task['deadline']).strftime('%d/%b/%Y')}")

    if "startline" in task.index and pd.notna(task["startline"]):
        lines.append(f"Startline: {pd.to_datetime(task['startline']).strftime('%d/%b/%Y')}")

    if "buffer_days" in task.index and pd.notna(task["buffer_days"]):
        lines.append(f"Buffer: {int(task['buffer_days'])} days")

    return "\n".join(lines)


# =========================
# GANTT PLOTTING
# =========================
# =========================
# REPAIRED GANTT PLOTTING (Labels on Right, Original X-Axis Logic Restored)
# =========================
def plot_gantt(tasks, output_path=None, show=True, return_fig=False, figsize=(14, 8)):
    if tasks.empty:
        print("No tasks to plot.")
        return

    tasks = prepare_gantt_labels(tasks)
    tasks = tasks.sort_values(
        by=["project_name", "template_name", "start_date", "task_id"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    # Adjust margins to fit labels on the right
    plt.subplots_adjust(right=0.85, left=0.2) 

    bars = []
    y_positions = []
    
    # Track unique project-template combinations for group separators
    group_markers = {}
    
    for idx, (_, task) in enumerate(tasks.iterrows()):
        duration = (task["end_date"] - task["start_date"]).days + 1
        color = TEMPLATE_COLORS.get(task["template_name"], BAR_COLOR)
        
        group_key = f"{task.project_name}|{task.template_name}"
        
        bar = ax.barh(
            idx,
            width=duration,
            left=task["start_date"],
            height=0.6,
            color=color
        )
        y_positions.append(idx)

        # Mark group boundaries
        if group_key not in group_markers:
            group_markers[group_key] = {'start_idx': idx, 'end_idx': idx}
        else:
            group_markers[group_key]['end_idx'] = idx

        for rect in bar:
            anchor_text = build_schedule_anchor_text(task)
            rect.annotation = (
                f"Project: {task.project_name}\n"
                f"Template: {task.template_name}\n"
                f"Task ID: {task.task_id}\n"
                f"Task Group: {task.task_group}\n"
                f"Description: {task.task_description}\n"
                f"Role: {task.role}\n"
                f"Start: {task.start_date.strftime('%d/%b/%Y')}\n"
                f"End: {task.end_date.strftime('%d/%b/%Y')}\n"
                f"Duration: {duration} days"
                + (f"\n{anchor_text}" if anchor_text else "")
            )
            bars.append(rect)

    # --- RESTORED X-AXIS & DATE RANGE LOGIC ---
    start_date = tasks["start_date"].min() - pd.Timedelta(days=3)
    end_date = tasks["end_date"].max() + pd.Timedelta(days=3)
    ax.set_xlim(start_date, end_date)
    
    total_days = (end_date - start_date).days
    if total_days <= 30:
        major_freq, minor_freq, major_format = 'D', 'D', '%d/%b'
    elif total_days <= 90:
        major_freq, minor_freq, major_format = '3D', 'D', '%d/%b'
    else:
        major_freq, minor_freq, major_format = 'W-MON', 'D', '%d/%b'
    
    major_ticks = pd.date_range(start=start_date, end=end_date, freq=major_freq)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([d.strftime(major_format) for d in major_ticks], 
                       fontsize=DAY_FONT_SIZE, color=FONT_COLOR, rotation=45, ha='right')
    
    if minor_freq:
        ax.set_xticks(pd.date_range(start=start_date, end=end_date, freq=minor_freq), minor=True)
    
    # Secondary x-axis with month/year view
    sec_ax = ax.secondary_xaxis('bottom')
    sec_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
    sec_ax.xaxis.set_major_locator(mdates.MonthLocator())
    sec_ax.tick_params(axis='x', labelsize=MONTH_FONT_SIZE, colors=FONT_COLOR)
    sec_ax.spines['bottom'].set_position(('outward', 35))
    sec_ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

    # --- NEW CLEAN Y-AXIS (Task on Left, Project/Template on Right) ---
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tasks["task_group"], fontsize=9, color=FONT_COLOR)

    for proj in tasks["project_name"].unique():
        proj_df = tasks[tasks["project_name"] == proj]
        mid_proj = proj_df.index.min() + (proj_df.index.max() - proj_df.index.min()) / 2
        # Project Label
        ax.text(1.12, mid_proj, proj, transform=ax.get_yaxis_transform(), 
                ha='center', va='center', fontweight='bold', fontsize=10, rotation=270)
        
        for temp in proj_df["template_name"].unique():
            temp_df = proj_df[proj_df["template_name"] == temp]
            mid_temp = temp_df.index.min() + (temp_df.index.max() - temp_df.index.min()) / 2
            # Template Label
            ax.text(1.04, mid_temp, temp, transform=ax.get_yaxis_transform(), 
                    ha='center', va='center', fontsize=9, rotation=270)
            
            # Horizontal Separator
            ax.axhline(y=temp_df.index.max() + 0.5, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # --- FINAL STYLING ---
    ax.set_title(TITLE, fontsize=TITLE_SIZE, color=FONT_COLOR).set_fontweight(TITLE_FONT_WEIGHT)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE, color=FONT_COLOR)
    ax.grid(axis='x', which='major', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.grid(axis='x', which='minor', linestyle=':', alpha=0.2, linewidth=0.5)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        sec_ax.spines[spine].set_visible(False)

    ax.invert_yaxis()
    
    cursor = mplcursors.cursor(bars, hover=mplcursors.HoverMode.Transient)
    @cursor.connect("add")
    def on_hover(sel):
        sel.annotation.set_text(sel.artist.annotation)
        sel.annotation.get_bbox_patch().set(facecolor="white", alpha=0.9)
        sel.annotation.set_fontsize(9)

    plt.tight_layout()
    if show: plt.show()
    if return_fig: return fig, ax


def plot_gantt_weekly(tasks, output_path=None, show=True, return_fig=False, figsize=(15, 8)):
    if tasks.empty:
        print("No tasks to plot.")
        return

    tasks = prepare_gantt_labels(tasks)
    tasks = tasks.sort_values(
        by=["project_name", "template_name", "start_date", "task_id"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    
    # Adjust right margin to make room for the new labels
    plt.subplots_adjust(right=0.85) 

    bars = []
    y_positions = []
    
    for idx, (_, task) in enumerate(tasks.iterrows()):
        duration = (task["end_date"] - task["start_date"]).days + 1
        color = TEMPLATE_COLORS.get(task["template_name"], BAR_COLOR)
        
        bar = ax.barh(idx, width=duration, left=task["start_date"], height=0.6, color=color)
        y_positions.append(idx)

        # Preserved hover logic
        for rect in bar:
            anchor_text = build_schedule_anchor_text(task)
            rect.annotation = (
                f"Project: {task.project_name}\n"
                f"Template: {task.template_name}\n"
                f"Task Group: {task.task_group}\n"
                f"Start: {task.start_date.strftime('%d/%b/%Y')}\n"
                f"End: {task.end_date.strftime('%d/%b/%Y')}"
            )
            bars.append(rect)

    # --- UPDATED Y-AXIS LABELING ---
    # Left Side: Only show the specific Task Group
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tasks["task_group"], fontsize=9, color=FONT_COLOR)

    # Right Side: Project and Template Labels
    for proj in tasks["project_name"].unique():
        proj_df = tasks[tasks["project_name"] == proj]
        mid_proj = proj_df.index.min() + (proj_df.index.max() - proj_df.index.min()) / 2
        
        # Project Label (Far Right)
        ax.text(1.12, mid_proj, proj, transform=ax.get_yaxis_transform(), 
                ha='center', va='center', fontweight='bold', fontsize=10, rotation=270)
        
        for temp in proj_df["template_name"].unique():
            temp_df = proj_df[proj_df["template_name"] == temp]
            mid_temp = temp_df.index.min() + (temp_df.index.max() - temp_df.index.min()) / 2
            
            # Template Label (Inside Right)
            ax.text(1.04, mid_temp, temp, transform=ax.get_yaxis_transform(), 
                    ha='center', va='center', fontsize=9, rotation=270)
            
            # Separator line to keep groups distinct
            ax.axhline(y=temp_df.index.max() + 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Visual separators for the right-side "Table"
    ax.annotate('', xy=(1.08, 0), xycoords='axes fraction', xytext=(1.08, 1), 
                arrowprops=dict(arrowstyle="-", color='black', alpha=0.2))
    ax.annotate('', xy=(1.16, 0), xycoords='axes fraction', xytext=(1.16, 1), 
                arrowprops=dict(arrowstyle="-", color='black', alpha=0.2))

    # --- PRESERVED X-AXIS & STYLE LOGIC ---
    start_date = tasks["start_date"].min() - pd.Timedelta(days=3)
    end_date = tasks["end_date"].max() + pd.Timedelta(days=3)
    ax.set_xlim(start_date, end_date)

    weekly_ticks = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    ax.set_xticks(weekly_ticks)
    ax.set_xticklabels([f"{d.strftime('%b %d')}-{(d+pd.Timedelta(days=6)).strftime('%d')}" for d in weekly_ticks], 
                       rotation=45, ha='right', fontsize=9, color=FONT_COLOR)
    
    ax.set_xticks(pd.date_range(start=start_date, end=end_date, freq="D"), minor=True)
    ax.grid(axis='x', which='major', linestyle='--', alpha=0.4)
    ax.grid(axis='x', which='minor', linestyle=':', alpha=0.2)

    ax.set_title(f"{TITLE} (Weekly View)", fontsize=TITLE_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.invert_yaxis()

    # Tooltip activation
    cursor = mplcursors.cursor(bars, hover=mplcursors.HoverMode.Transient)
    @cursor.connect("add")
    def on_hover(sel):
        sel.annotation.set_text(sel.artist.annotation)
        sel.annotation.get_bbox_patch().set(facecolor="white", alpha=0.9)

    plt.tight_layout()
    if output_path: plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if show: plt.show()
    if return_fig: return fig, ax


# =========================
# WEEKLY ROLE DEMAND
# =========================
def build_weekly_role_demand(tasks):
    if tasks.empty:
        return pd.DataFrame()

    tasks = tasks.copy()
    tasks["start_date"] = pd.to_datetime(tasks["start_date"])
    tasks["end_date"] = pd.to_datetime(tasks["end_date"])

    overall_start = tasks["start_date"].min()
    overall_end = tasks["end_date"].max()

    all_days = pd.date_range(start=overall_start, end=overall_end, freq="D")
    roles = sorted(tasks["role"].dropna().unique())

    daily_records = []

    for day in all_days:
        for role in roles:
            active = tasks[
                (tasks["role"] == role) &
                (tasks["start_date"] <= day) &
                (tasks["end_date"] >= day)
            ]

            daily_records.append({
                "date": day,
                "week_start": day - pd.Timedelta(days=day.weekday()),
                "role": role,
                "active_tasks": len(active)
            })

    daily_df = pd.DataFrame(daily_records)

    weekly_df = (
        daily_df.groupby(["week_start", "role"], as_index=False)["active_tasks"]
        .max()
        .rename(columns={"active_tasks": "people_needed"})
    )

    weekly_df["week_end"] = weekly_df["week_start"] + pd.Timedelta(days=6)
    weekly_df["week_label"] = weekly_df["week_start"].apply(
        lambda d: f"{d.strftime('%b')} W{((d.day - 1)//7)+1}"
    )

    return weekly_df.sort_values(by=["week_start", "role"]).reset_index(drop=True)


def plot_weekly_role_demand(tasks, output_path=None, show=True, return_fig=False, figsize=(14, 6)):
    demand_df = build_weekly_role_demand(tasks)

    if demand_df.empty:
        print("No demand data to plot.")
        return

    pivot = demand_df.pivot(index="week_label", columns="role", values="people_needed").fillna(0)

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Weekly Peak Role Demand", fontsize=TITLE_SIZE, color=FONT_COLOR, fontweight=TITLE_FONT_WEIGHT)
    ax.set_xlabel("Week")
    ax.set_ylabel("Peak People Needed")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    if return_fig:
        return fig, ax