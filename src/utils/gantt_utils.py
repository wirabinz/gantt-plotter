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
    projects = pd.read_excel(file_path, sheet_name=sheet_name)
    projects.columns = [str(c).strip() for c in projects.columns]

    required = ["project_name", "template_name", "deadline", "buffer_days"]
    missing = [c for c in required if c not in projects.columns]
    if missing:
        raise ValueError(f"Missing columns in PROJECT sheet: {missing}")

    projects = projects.dropna(how="all").copy()
    projects = projects[projects["project_name"].notna()].copy()

    projects["project_name"] = projects["project_name"].astype(str).str.strip()
    projects["template_name"] = projects["template_name"].astype(str).str.strip()
    projects["deadline"] = pd.to_datetime(projects["deadline"])
    projects["buffer_days"] = pd.to_numeric(projects["buffer_days"], errors="coerce").fillna(0).astype(int)

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


# =========================
# PROJECT BUILDERS
# =========================
def build_project_schedule(file_path, project_row):
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


def build_master_schedule(file_path, project_sheet="PROJECT"):
    projects = load_project_sheet(file_path, project_sheet)

    all_schedules = []
    for _, row in projects.iterrows():
        schedule = build_project_schedule(file_path, row)
        all_schedules.append(schedule)

    if not all_schedules:
        return pd.DataFrame()

    master = pd.concat(all_schedules, ignore_index=True)
    master = master.sort_values(by=["project_name", "template_name", "start_date", "task_id"]).reset_index(drop=True)
    return master


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


# =========================
# GANTT PLOTTING
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
    bars = []

    for _, task in tasks.iterrows():
        duration = (task["end_date"] - task["start_date"]).days + 1
        color = TEMPLATE_COLORS.get(task["template_name"], BAR_COLOR)
        y_label = f"{task.project_name} | {task.short_label}"

        bar = ax.barh(
            y_label,
            width=duration,
            left=task["start_date"],
            height=0.6,
            color=color
        )

        for rect in bar:
            rect.annotation = (
                f"Project: {task.project_name}\n"
                f"Template: {task.template_name}\n"
                f"Task ID: {task.task_id}\n"
                f"Task Group: {task.task_group}\n"
                f"Description: {task.task_description}\n"
                f"Role: {task.role}\n"
                f"Start: {task.start_date.strftime('%d/%b/%Y')}\n"
                f"End: {task.end_date.strftime('%d/%b/%Y')}\n"
                f"Duration: {duration} days\n"
                f"Deadline: {task.deadline.strftime('%d/%b/%Y')}"
            )
            bars.append(rect)

    cursor = mplcursors.cursor(bars, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")
    def on_hover(sel):
        sel.annotation.set_text(sel.artist.annotation)
        sel.annotation.get_bbox_patch().set(facecolor="white", alpha=0.9)
        sel.annotation.set_fontsize(9)

    start_date = tasks["start_date"].min() - pd.Timedelta(days=3)
    end_date = tasks["end_date"].max() + pd.Timedelta(days=3)

    week_positions, week_labels = build_week_ticks(start_date, end_date)

    ax.set_xlim(start_date, end_date)
    ax.set_xticks(week_positions)
    ax.set_xticklabels(week_labels, fontsize=DAY_FONT_SIZE, color=FONT_COLOR)

    ax.set_title(TITLE, fontsize=TITLE_SIZE, color=FONT_COLOR).set_fontweight(TITLE_FONT_WEIGHT)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE, color=FONT_COLOR)
    ax.set_ylabel("")
    ax.tick_params(axis='both', colors=FONT_COLOR)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    sec_ax = ax.secondary_xaxis('bottom')
    sec_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
    sec_ax.xaxis.set_major_locator(mdates.MonthLocator())
    sec_ax.tick_params(axis='x', labelsize=MONTH_FONT_SIZE, colors=FONT_COLOR)
    sec_ax.spines['bottom'].set_position(('outward', 20))

    for label in sec_ax.get_xticklabels():
        label.set_fontsize(MONTH_FONT_SIZE)
        label.set_weight(MONTH_FONT_WEIGHT)
        label.set_color(FONT_COLOR)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        sec_ax.spines[spine].set_visible(False)

    ax.invert_yaxis()

    handles = [mpatches.Patch(color=color, label=template) for template, color in TEMPLATE_COLORS.items()]
    ax.legend(handles=handles, loc='best', framealpha=0.8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    if return_fig:
        return fig, ax


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
    bars = []

    for _, task in tasks.iterrows():
        duration = (task["end_date"] - task["start_date"]).days + 1
        color = TEMPLATE_COLORS.get(task["template_name"], BAR_COLOR)
        y_label = f"{task.project_name} | {task.short_label}"

        bar = ax.barh(
            y_label,
            width=duration,
            left=task["start_date"],
            height=0.6,
            color=color
        )

        for rect in bar:
            rect.annotation = (
                f"Project: {task.project_name}\n"
                f"Template: {task.template_name}\n"
                f"Task ID: {task.task_id}\n"
                f"Task Group: {task.task_group}\n"
                f"Description: {task.task_description}\n"
                f"Role: {task.role}\n"
                f"Start: {task.start_date.strftime('%d/%b/%Y')}\n"
                f"End: {task.end_date.strftime('%d/%b/%Y')}\n"
                f"Duration: {duration} days\n"
                f"Deadline: {task.deadline.strftime('%d/%b/%Y')}"
            )
            bars.append(rect)

    cursor = mplcursors.cursor(bars, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")
    def on_hover(sel):
        sel.annotation.set_text(sel.artist.annotation)
        sel.annotation.get_bbox_patch().set(facecolor="white", alpha=0.9)
        sel.annotation.set_fontsize(9)

    start_date = tasks["start_date"].min() - pd.Timedelta(days=3)
    end_date = tasks["end_date"].max() + pd.Timedelta(days=3)

    ax.set_xlim(start_date, end_date)

    weekly_ticks = pd.date_range(start=start_date, end=end_date, freq="W-MON")

    def week_label(d):
        week_num = ((d.day - 1) // 7) + 1
        return f"{d.strftime('%b')} W{week_num}"

    weekly_labels = [week_label(d) for d in weekly_ticks]

    ax.set_xticks(weekly_ticks)
    ax.set_xticklabels(weekly_labels, rotation=45, ha='right', fontsize=9, color=FONT_COLOR)

    ax.set_title(f"{TITLE} (Weekly View)", fontsize=TITLE_SIZE, color=FONT_COLOR).set_fontweight(TITLE_FONT_WEIGHT)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='both', colors=FONT_COLOR)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.invert_yaxis()

    handles = [mpatches.Patch(color=color, label=template) for template, color in TEMPLATE_COLORS.items()]
    ax.legend(handles=handles, loc='best', framealpha=0.8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    if return_fig:
        return fig, ax


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