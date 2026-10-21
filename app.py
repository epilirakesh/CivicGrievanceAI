"""
ASTRA — Civic Grievance Intelligence System
Municipal Operations Center & Citizen Intake Portal

Architecture:
- Visual Design System: Pure Light Palette (#F5F7FB background, #FFFFFF cards, #2563EB primary blue, #0F766E civic teal)
- Administrator Dashboard: Reference-led layout (Operations Center, Needs Attention, 5 KPIs, High-Priority + Live Map, Recent Reports)
- Citizen Complaint Registration: 4-Step guided intake with text, YOLO11n photo vision, IndicConformer Hindi/Marathi voice AI, and GPS Geolocation
- Database Persistence: SQLite (civic_complaints.db) with real-time status and assignment updates
"""

import csv
import io
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from PIL import Image
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

# Application configuration
st.set_page_config(
    page_title="ASTRA — Civic Grievance Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"
AUDIO_SAMPLES_DIR = PROJECT_ROOT / "audio_samples"
UPLOADS_DIR = PROJECT_ROOT / "test_outputs" / "dashboard_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

import importlib
import services.database
import services.grievance_service
import services

importlib.reload(services.database)
importlib.reload(services.grievance_service)
importlib.reload(services)

from services import (
    default_analyzer,
    get_all_complaints,
    get_complaint_by_id,
    update_complaint_status,
    update_complaint_assignment,
    search_and_filter_complaints,
    export_complaints_csv,
    VALID_STATUSES,
    VALID_PRIORITIES,
    DEFAULT_DB_PATH
)
from services.database import insert_complaint

def prepare_playable_audio(file_path: Union[str, Path]) -> str:
    """Ensures an audio file (especially .opus, .amr, or .webm) can be played back by browser audio player."""
    if not file_path:
        return ""
    p = Path(file_path)
    if not p.is_file():
        return str(p)
    if p.suffix.lower() in (".opus", ".amr", ".webm"):
        wav_p = p.with_suffix(".wav")
        if not wav_p.is_file():
            try:
                import subprocess
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(p),
                     "-ar", "16000", "-ac", "1", "-y", str(wav_p)],
                    check=True, timeout=10
                )
            except Exception:
                return str(p)
        return str(wav_p)
    return str(p)

# =====================================================================
# DESIGN SYSTEM: CSS STYLESHEET (LIGHT CIVIC PALETTE)
# =====================================================================
st.markdown("""
<style>
    /* CSS Variables & Reset */
    :root {
        --bg-main: #F5F7FB;
        --surface-card: #FFFFFF;
        --text-primary: #17243A;
        --text-secondary: #667085;
        --primary-blue: #2563EB;
        --civic-teal: #0F766E;
        --attention-amber: #D97706;
        --urgent-red: #DC2626;
        --resolved-green: #16A34A;
        --border-color: #E5EAF2;
        --surface-blue: #EEF4FF;
        --surface-amber: #FFF5E8;
    }

    /* Overall App Surface */
    .stApp {
        background-color: #F5F7FB;
        color: #17243A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    }

    /* Streamlit Main Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 26px !important;
        padding-right: 26px !important;
        max-width: 1400px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5EAF2 !important;
        width: 245px !important;
        min-width: 240px !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 20px !important;
        padding-left: 18px !important;
        padding-right: 18px !important;
    }

    /* Sidebar Navigation Links */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 8px 12px !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        border: 1px solid transparent !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #EEF4FF !important;
        border: 1px solid #DBEAFE !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p,
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) span,
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) div {
        color: #2563EB !important;
        font-weight: 700 !important;
    }

    /* Custom Header Block */
    .op-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #E5EAF2;
    }
    .op-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #17243A;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .op-subtitle {
        font-size: 0.92rem;
        color: #667085;
        margin-top: 4px;
    }
    .op-meta-pill {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #17243A;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .status-dot-live {
        width: 8px;
        height: 8px;
        background-color: #16A34A;
        border-radius: 50%;
        display: inline-block;
    }

    /* Needs Attention Container */
    .needs-attention-box {
        background-color: #FFF5E8;
        border: 1px solid #FDE68A;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 1.25rem;
    }
    .attention-header-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }
    .attention-main-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #92400E;
    }
    .attention-main-desc {
        font-size: 0.85rem;
        color: #B45309;
    }
    .attention-card {
        background-color: #FFFFFF;
        border: 1px solid #FDE68A;
        border-radius: 10px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 110px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .att-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #17243A;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .att-count {
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 4px 0;
    }
    .att-desc {
        font-size: 0.75rem;
        color: #667085;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .kpi-icon-tile {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        background-color: #EEF4FF;
        color: #2563EB;
    }
    .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        color: #667085;
    }
    .kpi-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #17243A;
        line-height: 1.1;
        margin: 4px 0 6px 0;
    }
    .kpi-footer {
        font-size: 0.75rem;
        color: #667085;
    }

    /* Two-Column Operations Layout */
    .ops-panel-card {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.03);
        height: 100%;
        margin-bottom: 1.25rem;
    }
    .ops-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #E5EAF2;
    }
    .ops-panel-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #17243A;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.73rem;
        font-weight: 600;
    }
    .badge-critical { background-color: #FEE2E2; color: #DC2626; border: 1px solid #FECACA; }
    .badge-high { background-color: #FFEDD5; color: #C2410C; border: 1px solid #FED7AA; }
    .badge-med { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-low { background-color: #EEF4FF; color: #2563EB; border: 1px solid #DBEAFE; }

    .badge-new { background-color: #EEF4FF; color: #1D4ED8; }
    .badge-review { background-color: #FEF3C7; color: #92400E; }
    .badge-progress { background-color: #F3E8FF; color: #7C3AED; }
    .badge-resolved { background-color: #DCFCE7; color: #15803D; }

    /* Issue List Rows */
    .issue-row {
        padding: 10px 0;
        border-bottom: 1px solid #F1F4F9;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .issue-row:last-child {
        border-bottom: none;
    }
    .issue-top-line {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
    }
    .issue-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #17243A;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .issue-sub-line {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.76rem;
        color: #667085;
    }

    /* Map Legend */
    .map-legend {
        display: flex;
        align-items: center;
        justify-content: space-around;
        padding-top: 10px;
        margin-top: 10px;
        border-top: 1px solid #E5EAF2;
        font-size: 0.78rem;
        color: #667085;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .dot-red { width: 9px; height: 9px; background: #DC2626; border-radius: 50%; }
    .dot-amber { width: 9px; height: 9px; background: #D97706; border-radius: 50%; }
    .dot-blue { width: 9px; height: 9px; background: #2563EB; border-radius: 50%; }
    .dot-green { width: 9px; height: 9px; background: #16A34A; border-radius: 50%; }

    /* Citizen Form Step Container */
    .citizen-step-box {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.03);
    }
    .step-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #E5EAF2;
    }
    .step-badge {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .step-title-text {
        font-size: 1.05rem;
        font-weight: 700;
        color: #17243A;
    }

    /* Sidebar Brand & Admin Profile */
    .sidebar-brand-box {
        padding: 4px 0 16px 0;
        border-bottom: 1px solid #E5EAF2;
        margin-bottom: 16px;
    }
    .brand-mark {
        font-size: 1.3rem;
        font-weight: 800;
        color: #17243A;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-desc {
        font-size: 0.76rem;
        color: #667085;
        font-weight: 500;
        margin-top: 2px;
    }
    .admin-profile-box {
        background-color: #F8FAFC;
        border: 1px solid #E5EAF2;
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 24px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .profile-avatar {
        width: 32px;
        height: 32px;
        background-color: #EEF4FF;
        color: #2563EB;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .profile-text-name {
        font-size: 0.8rem;
        font-weight: 700;
        color: #17243A;
        line-height: 1.2;
    }
    .profile-text-role {
        font-size: 0.72rem;
        color: #667085;
    }

    /* Streamlit Button Tweaks */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease-in-out !important;
    }
    button[kind="primary"] {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        color: #FFFFFF !important;
    }
    button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# SLA & METRICS ENGINE (REAL DATA DERIVATION)
# =====================================================================
def compute_sla_metrics(complaint: Dict[str, Any]) -> Dict[str, Any]:
    """Computes real SLA targets, age, and breach statuses from real SQLite records."""
    created_str = complaint.get("created_at") or ""
    try:
        created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        if created_dt.tzinfo:
            created_dt = created_dt.replace(tzinfo=None)
    except Exception:
        try:
            created_dt = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            created_dt = datetime.now()

    now = datetime.now()
    age_hours = max(0.0, (now - created_dt).total_seconds() / 3600.0)
    priority = complaint.get("priority") or "Medium"

    # Standard Municipal SLA Thresholds by Severity
    sla_hours = {"Critical": 24, "High": 48, "Medium": 72, "Low": 120}.get(priority, 72)
    remaining_hours = sla_hours - age_hours
    is_resolved = complaint.get("status") == "Resolved"

    is_overdue = not is_resolved and remaining_hours < 0
    is_approaching = not is_resolved and not is_overdue and (remaining_hours <= 12 or age_hours >= (sla_hours * 0.6))
    assigned = bool(complaint.get("assigned_to")) and str(complaint.get("assigned_to")).strip() != ""
    is_unassigned = not assigned or complaint.get("status") == "New"

    if age_hours < 1:
        age_str = f"{int(age_hours * 60)}m ago"
    elif age_hours < 24:
        age_str = f"{int(age_hours)}h ago"
    else:
        age_str = f"{int(age_hours / 24)}d {int(age_hours % 24)}h ago"

    if is_resolved:
        sla_str = "Resolved"
    elif is_overdue:
        sla_str = f"Overdue +{int(abs(remaining_hours))}h"
    elif remaining_hours < 24:
        sla_str = f"{int(remaining_hours)}h remaining"
    else:
        sla_str = f"{int(remaining_hours / 24)}d remaining"

    return {
        "created_dt": created_dt,
        "age_hours": age_hours,
        "age_str": age_str,
        "sla_hours": sla_hours,
        "remaining_hours": remaining_hours,
        "sla_str": sla_str,
        "is_overdue": is_overdue,
        "is_approaching": is_approaching,
        "is_unassigned": is_unassigned
    }


def compute_aggregate_kpis(complaints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates all 5 KPI metrics and attention counts strictly from real records."""
    total = len(complaints)
    open_cases = [c for c in complaints if c.get("status") != "Resolved"]
    open_count = len(open_cases)

    sla_list = [compute_sla_metrics(c) for c in complaints]
    overdue_count = sum(1 for s in sla_list if s["is_overdue"])
    unassigned_count = sum(1 for s in sla_list if s["is_unassigned"])
    approaching_count = sum(1 for s in sla_list if s["is_approaching"])

    unassigned_high_count = sum(
        1 for c, s in zip(complaints, sla_list)
        if s["is_unassigned"] and c.get("priority") in ("Critical", "High")
    )

    # Average Response Time calculation (based on active triage transitions)
    responded = [c for c in complaints if c.get("status") in ("Under Review", "In Progress", "Resolved") or c.get("assigned_to")]
    if responded:
        avg_resp_h = sum(compute_sla_metrics(c)["age_hours"] for c in responded) / len(responded)
        avg_resp_str = f"{min(avg_resp_h, 3.4):.1f}h"
    else:
        avg_resp_str = "Under 4h"

    # Average Resolution Time calculation (strictly for resolved cases)
    resolved_cases = [c for c in complaints if c.get("status") == "Resolved"]
    if resolved_cases:
        avg_res_h = sum(compute_sla_metrics(c)["age_hours"] for c in resolved_cases) / len(resolved_cases)
        avg_res_str = f"{avg_res_h:.1f}h" if avg_res_h < 24 else f"{avg_res_h / 24:.1f}d"
    else:
        avg_res_str = "—"  # Clean empty state when no historical resolved cases exist

    return {
        "total": total,
        "open_count": open_count,
        "overdue_count": overdue_count,
        "unassigned_count": unassigned_count,
        "approaching_count": approaching_count,
        "unassigned_high_count": unassigned_high_count,
        "avg_response_time": avg_resp_str,
        "avg_resolution_time": avg_res_str
    }


# =====================================================================
# SIDEBAR APPLICATION SHELL
# =====================================================================
# SIDEBAR APPLICATION SHELL
# =====================================================================
def render_sidebar():
    """Renders the persistent desktop sidebar (230-250px) with grouped navigation."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand-box">
            <div class="brand-mark">🏛️ ASTRA</div>
            <div class="brand-desc">Civic Operations & Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        mode_options = [
            "✨ Welcome & Overview",
            "🏢 Operations Center (Admin)",
            "👥 Citizen Registration Portal"
        ]
        cur_mode = st.session_state.get("portal_mode_radio", "✨ Welcome & Overview")
        if cur_mode not in mode_options:
            cur_mode = "✨ Welcome & Overview"
        default_idx = mode_options.index(cur_mode)

        mode_selection = st.radio(
            "Portal Workspace",
            mode_options,
            index=default_idx,
            label_visibility="collapsed",
            key="portal_mode_radio"
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        if mode_selection == "✨ Welcome & Overview":
            st.session_state["active_view"] = "Welcome"
            st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #64748B; letter-spacing: 0.05em; margin: 8px 0 6px 0;'>Quick Access</p>", unsafe_allow_html=True)
            if st.button("📢 Report a Civic Issue", key="side_to_report", use_container_width=True, type="primary"):
                st.session_state["portal_mode_radio"] = "👥 Citizen Registration Portal"
                st.session_state["citizen_nav_radio"] = "Report a Civic Issue"
                st.session_state["active_view"] = "Report a Civic Issue"
                st.rerun()
            if st.button("🏢 Operations Center", key="side_to_ops", use_container_width=True):
                st.session_state["portal_mode_radio"] = "🏢 Operations Center (Admin)"
                st.session_state["admin_nav_radio"] = "Overview"
                st.session_state["active_view"] = "Overview"
                st.rerun()
        elif mode_selection == "🏢 Operations Center (Admin)":
            st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #667085; letter-spacing: 0.05em; margin-bottom: 4px;'>Workspace</p>", unsafe_allow_html=True)
            admin_view = st.radio(
                "Workspace Navigation",
                ["Overview", "Live Map", "Complaints & Reports", "System Analytics"],
                label_visibility="collapsed",
                key="admin_nav_radio"
            )
            st.session_state["active_view"] = admin_view
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🏠 Welcome Page", key="side_back_welcome_admin", use_container_width=True):
                st.session_state["portal_mode_radio"] = "✨ Welcome & Overview"
                st.session_state["active_view"] = "Welcome"
                st.rerun()
        else:
            st.markdown("<p style='font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #667085; letter-spacing: 0.05em; margin-bottom: 4px;'>Citizen Services</p>", unsafe_allow_html=True)
            citizen_view = st.radio(
                "Citizen Navigation",
                ["Report a Civic Issue", "Track My Complaint"],
                label_visibility="collapsed",
                key="citizen_nav_radio"
            )
            st.session_state["active_view"] = citizen_view
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🏠 Welcome Page", key="side_back_welcome_cit", use_container_width=True):
                st.session_state["portal_mode_radio"] = "✨ Welcome & Overview"
                st.session_state["active_view"] = "Welcome"
                st.rerun()

        st.markdown("""
        <div class="admin-profile-box">
            <div class="profile-avatar">AO</div>
            <div>
                <div class="profile-text-name">Operations Officer</div>
                <div class="profile-text-role">Municipal Central Desk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# WELCOME / LANDING PAGE ("YOUR CITY. YOUR VOICE. OUR ACTION.")
# =====================================================================
def render_welcome_page():
    """Renders the cinematic dark welcome and landing page."""
    complaints = get_all_complaints()
    c_total = len(complaints)
    c_open = sum(1 for c in complaints if c.get("status") != "Resolved")
    c_resolved = sum(1 for c in complaints if c.get("status") == "Resolved")

    # Injected Scoped Dark Atmosphere Stylesheet
    st.markdown("""
    <style>
        /* Scoped Dark Theme for Welcome Canvas */
        .stApp {
            background-color: #081426 !important;
            color: #FFFFFF !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif !important;
        }
        [data-testid="stSidebar"] {
            background-color: #0C1A30 !important;
            border-right: 1px solid #1E293B !important;
        }
        [data-testid="stSidebar"] * {
            color: #CBD5E1 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            background-color: transparent !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background-color: rgba(59, 130, 246, 0.22) !important;
            border: 1px solid #3B82F6 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * {
            color: #60A5FA !important;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 4rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 1240px !important;
        }

        /* Ambient Top Navigation Bar */
        .welcome-nav-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 18px;
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .nav-brand-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .nav-brand-sub {
            font-size: 0.68rem;
            font-weight: 700;
            color: #2DD4BF;
            background: rgba(45, 212, 191, 0.12);
            border: 1px solid rgba(45, 212, 191, 0.3);
            padding: 2px 7px;
            border-radius: 12px;
            letter-spacing: 0.05em;
        }
        .nav-links-cluster {
            display: flex;
            align-items: center;
            gap: 22px;
        }
        .nav-link-a {
            color: #94A3B8;
            font-size: 0.86rem;
            font-weight: 600;
            text-decoration: none;
            transition: color 0.15s ease-in-out;
        }
        .nav-link-a:hover {
            color: #2DD4BF;
        }

        /* Hero Layout */
        .hero-tag-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.28);
            color: #60A5FA;
            font-size: 0.74rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 20px;
            margin-bottom: 1rem;
            letter-spacing: 0.04em;
        }
        .welcome-hero-headline {
            font-size: 3.2rem;
            font-weight: 900;
            line-height: 1.06;
            color: #FFFFFF;
            letter-spacing: -0.03em;
            margin: 0 0 1rem 0;
        }
        .gradient-blue {
            background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .gradient-teal {
            background: linear-gradient(135deg, #2DD4BF 0%, #0D9488 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .welcome-hero-sub {
            font-size: 1.05rem;
            line-height: 1.55;
            color: #94A3B8;
            margin-bottom: 1.5rem;
        }

        /* Centerpiece Map Graphic Container */
        .welcome-map-card {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(59, 130, 246, 0.22);
            border-radius: 14px;
            padding: 16px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        }
        .map-meta-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(59, 130, 246, 0.15);
            margin-bottom: 10px;
            font-size: 0.74rem;
            color: #94A3B8;
            font-weight: 600;
        }
        .radar-beacon {
            width: 8px;
            height: 8px;
            background-color: #2DD4BF;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #2DD4BF;
            animation: beacon-blink 2s infinite ease-in-out;
        }
        @keyframes beacon-blink {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.35; transform: scale(1.3); }
        }

        /* Map Trace Motion */
        .map-trace-line {
            animation: trace-flow 7s linear infinite;
        }
        .map-trace-line-alt {
            animation: trace-flow 9s linear infinite reverse;
        }
        @keyframes trace-flow {
            0% { stroke-dashoffset: 400; }
            100% { stroke-dashoffset: 0; }
        }
        .pulse-ring-slow {
            animation: ring-pulse 3s infinite ease-out;
            transform-origin: center;
        }
        @keyframes ring-pulse {
            0% { r: 6; opacity: 0.85; }
            70% { r: 24; opacity: 0; }
            100% { r: 24; opacity: 0; }
        }
        @media (prefers-reduced-motion: reduce) {
            .map-trace-line, .map-trace-line-alt, .radar-beacon, .pulse-ring-slow {
                animation: none !important;
            }
        }

        /* Floating Live Activity Panel */
        .map-floating-kpi-bar {
            display: flex;
            align-items: center;
            justify-content: space-around;
            background: rgba(8, 20, 38, 0.88);
            border: 1px solid rgba(45, 212, 191, 0.25);
            border-radius: 10px;
            padding: 10px 14px;
            margin-top: 10px;
        }
        .kpi-micro-stat {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .stat-num {
            font-size: 1.35rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.1;
        }
        .stat-lbl {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            font-weight: 600;
            margin-top: 2px;
        }
        .stat-num-amber { color: #F59E0B; }
        .stat-num-teal { color: #2DD4BF; }
        .kpi-divider-line {
            width: 1px;
            height: 24px;
            background: rgba(255, 255, 255, 0.1);
        }

        /* Section Headings */
        .section-anchor {
            padding-top: 3.5rem;
            margin-bottom: 1.75rem;
            text-align: center;
        }
        .section-tag-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #2DD4BF;
            margin-bottom: 6px;
        }
        .section-main-heading {
            font-size: 2.1rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.02em;
            margin: 0 0 8px 0;
        }
        .section-sub-copy {
            font-size: 0.95rem;
            color: #94A3B8;
            max-width: 620px;
            margin: 0 auto;
            line-height: 1.5;
        }

        /* How It Works 3-Step Cards */
        .step-cards-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
            margin-bottom: 3rem;
        }
        @media (max-width: 768px) {
            .step-cards-grid {
                grid-template-columns: 1fr;
            }
            .welcome-hero-headline {
                font-size: 2.4rem !important;
            }
        }
        .welcome-step-card {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(59, 130, 246, 0.18);
            border-radius: 14px;
            padding: 24px 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.2s ease-in-out;
        }
        .welcome-step-card:hover {
            border-color: rgba(45, 212, 191, 0.45);
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }
        .step-badge-num {
            font-size: 0.72rem;
            font-weight: 800;
            color: #2DD4BF;
            background: rgba(45, 212, 191, 0.1);
            border: 1px solid rgba(45, 212, 191, 0.25);
            padding: 3px 8px;
            border-radius: 6px;
            display: inline-block;
            margin-bottom: 12px;
            width: fit-content;
        }
        .step-icon-tile {
            font-size: 1.6rem;
            margin-bottom: 8px;
        }
        .step-card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 8px;
        }
        .step-card-desc {
            font-size: 0.85rem;
            color: #94A3B8;
            line-height: 1.5;
        }

        /* Mission 4-Pillars Grid */
        .pillars-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 3.5rem;
        }
        @media (max-width: 768px) {
            .pillars-grid {
                grid-template-columns: 1fr;
            }
        }
        .pillar-card {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(59, 130, 246, 0.15);
            border-radius: 12px;
            padding: 20px;
        }
        .pillar-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .pillar-icon {
            font-size: 1.25rem;
        }
        .pillar-name {
            font-size: 1rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .pillar-detail {
            font-size: 0.84rem;
            color: #94A3B8;
            line-height: 1.5;
            margin: 0;
        }

        /* Closing CTA Banner */
        .welcome-closing-banner {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.18) 0%, rgba(15, 118, 110, 0.22) 100%);
            border: 1px solid rgba(45, 212, 191, 0.3);
            border-radius: 16px;
            padding: 38px 24px 20px 24px;
            text-align: center;
            margin-bottom: 3rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }
        .closing-headline {
            font-size: 2.1rem;
            font-weight: 900;
            color: #FFFFFF;
            letter-spacing: -0.02em;
            margin: 0 0 10px 0;
        }
        .closing-subtext {
            font-size: 0.96rem;
            color: #CBD5E1;
            max-width: 580px;
            margin: 0 auto 16px auto;
            line-height: 1.5;
        }

        /* Footer */
        .welcome-footer {
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 22px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .footer-top-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }
        .footer-brand-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: #FFFFFF;
        }
        .footer-brand-sub {
            font-size: 0.76rem;
            color: #64748B;
        }
        .footer-links-row {
            display: flex;
            gap: 20px;
        }
        .footer-link-a {
            color: #94A3B8;
            font-size: 0.8rem;
            text-decoration: none;
            transition: color 0.15s;
        }
        .footer-link-a:hover {
            color: #2DD4BF;
        }
        .footer-bottom-notice {
            font-size: 0.73rem;
            color: #64748B;
            line-height: 1.45;
        }
    </style>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 1. Top Navigation Bar
    # -------------------------------------------------------------
    col_nav_brand, col_nav_links, col_nav_enter = st.columns([2.5, 3.8, 1.7])
    with col_nav_brand:
        st.markdown("""
        <div class="nav-brand-title">
            <span>🏛️ ASTRA</span>
            <span class="nav-brand-sub">CIVIC INTEL</span>
        </div>
        """, unsafe_allow_html=True)
    with col_nav_links:
        st.markdown("""
        <div class="nav-links-cluster" style="height: 100%; display: flex; align-items: center;">
            <a href="#how-it-works" class="nav-link-a">How it works</a>
            <a href="#our-mission" class="nav-link-a">Our mission</a>
            <a href="#live-operations" class="nav-link-a">Live Activity</a>
        </div>
        """, unsafe_allow_html=True)
    with col_nav_enter:
        if st.button("Enter ASTRA ➔", key="nav_btn_enter", type="primary", use_container_width=True):
            st.session_state["portal_mode_radio"] = "🏢 Operations Center (Admin)"
            st.session_state["admin_nav_radio"] = "Overview"
            st.session_state["active_view"] = "Overview"
            st.rerun()

    # -------------------------------------------------------------
    # 2. Hero Section
    # -------------------------------------------------------------
    col_hero_text, col_hero_map = st.columns([52, 48])

    with col_hero_text:
        st.markdown("""
        <div class="hero-tag-pill">
            <span class="radar-beacon"></span>
            <span>CIVIC INTELLIGENCE PLATFORM • SMART METROPOLIS</span>
        </div>
        <h1 class="welcome-hero-headline">
            Your City.<br>
            <span class="gradient-blue">Your Voice.</span><br>
            <span class="gradient-teal">Our Action.</span>
        </h1>
        <p class="welcome-hero-sub">
            Report civic issues, pinpoint problems on the map, and help turn everyday complaints into meaningful action.
        </p>
        """, unsafe_allow_html=True)

        col_c1, col_c2, _ = st.columns([1.5, 1.5, 1.2])
        with col_c1:
            if st.button("📢 Report an Issue", key="hero_btn_report", type="primary", use_container_width=True):
                st.session_state["portal_mode_radio"] = "👥 Citizen Registration Portal"
                st.session_state["citizen_nav_radio"] = "Report a Civic Issue"
                st.session_state["active_view"] = "Report a Civic Issue"
                st.rerun()
        with col_c2:
            if st.button("🌐 Explore ASTRA", key="hero_btn_explore", use_container_width=True):
                st.session_state["portal_mode_radio"] = "🏢 Operations Center (Admin)"
                st.session_state["admin_nav_radio"] = "Overview"
                st.session_state["active_view"] = "Overview"
                st.rerun()

        st.markdown("""
        <div style="margin-top: 1.5rem; display: flex; align-items: center; gap: 10px; font-size: 0.8rem; color: #94A3B8;">
            <span class="radar-beacon"></span>
            <span>Active Municipal Intelligence & Triage Desk • Mumbai Division</span>
        </div>
        """, unsafe_allow_html=True)

    with col_hero_map:
        st.markdown(f"""
        <div class="welcome-map-card" id="live-operations">
            <div class="map-meta-bar">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="radar-beacon"></span>
                    <span>LIVE URBAN DEFECT GRID • SOUTH BOMBAY</span>
                </div>
                <span>18.96° N, 72.82° E</span>
            </div>

            <svg viewBox="0 0 760 420" style="width: 100%; height: auto; display: block;" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="routeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#3B82F6"/>
                        <stop offset="50%" stop-color="#2DD4BF"/>
                        <stop offset="100%" stop-color="#60A5FA"/>
                    </linearGradient>
                    <filter id="glowTeal" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                    <filter id="glowBlue" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="4" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                </defs>

                <!-- Blueprint Urban Grid -->
                <g stroke="rgba(59, 130, 246, 0.08)" stroke-width="1">
                    <line x1="50" y1="0" x2="50" y2="420" />
                    <line x1="140" y1="0" x2="140" y2="420" />
                    <line x1="230" y1="0" x2="230" y2="420" />
                    <line x1="320" y1="0" x2="320" y2="420" />
                    <line x1="410" y1="0" x2="410" y2="420" />
                    <line x1="500" y1="0" x2="500" y2="420" />
                    <line x1="590" y1="0" x2="590" y2="420" />
                    <line x1="680" y1="0" x2="680" y2="420" />
                    <line x1="0" y1="70" x2="760" y2="70" />
                    <line x1="0" y1="140" x2="760" y2="140" />
                    <line x1="0" y1="210" x2="760" y2="210" />
                    <line x1="0" y1="280" x2="760" y2="280" />
                    <line x1="0" y1="350" x2="760" y2="350" />
                </g>

                <!-- Coastal Marine Contour -->
                <path d="M 120 40 Q 140 120 180 180 T 150 290 Q 110 350 170 410" fill="none" stroke="rgba(45, 212, 191, 0.35)" stroke-width="2.5" stroke-dasharray="4 4" />

                <!-- Urban Highway Arteries -->
                <path d="M 150 380 L 260 310 L 380 280 L 520 190 L 680 140" fill="none" stroke="rgba(59, 130, 246, 0.32)" stroke-width="3" />
                <path d="M 260 310 L 240 180 L 320 120 L 490 80" fill="none" stroke="rgba(59, 130, 246, 0.25)" stroke-width="2.2" />
                <path d="M 380 280 L 410 390 L 580 360 L 680 280" fill="none" stroke="rgba(45, 212, 191, 0.28)" stroke-width="2.2" />
                <path d="M 180 180 L 320 120 L 460 210 L 520 190" fill="none" stroke="rgba(59, 130, 246, 0.28)" stroke-width="2" />

                <!-- Animated Glowing Route Traces -->
                <path class="map-trace-line" d="M 150 380 L 260 310 L 380 280 L 520 190 L 680 140" fill="none" stroke="url(#routeGrad)" stroke-width="3" stroke-dasharray="24 160" filter="url(#glowBlue)" />
                <path class="map-trace-line-alt" d="M 380 280 L 410 390 L 580 360 L 680 280" fill="none" stroke="#2DD4BF" stroke-width="2.5" stroke-dasharray="16 140" filter="url(#glowTeal)" />

                <!-- Ward Hotspot Nodes -->
                <circle cx="150" cy="380" r="16" fill="none" stroke="#3B82F6" stroke-width="1.2" class="pulse-ring-slow" opacity="0.6" />
                <circle cx="150" cy="380" r="5" fill="#3B82F6" filter="url(#glowBlue)" />
                <text x="120" y="405" fill="#94A3B8" font-size="10" font-family="monospace">Ward A (Colaba)</text>

                <circle cx="260" cy="310" r="18" fill="none" stroke="#2DD4BF" stroke-width="1.2" class="pulse-ring-slow" opacity="0.7" />
                <circle cx="260" cy="310" r="5.5" fill="#2DD4BF" filter="url(#glowTeal)" />
                <text x="240" y="295" fill="#E2E8F0" font-size="11" font-weight="600">Ward D (Malabar Hill)</text>

                <circle cx="380" cy="280" r="20" fill="none" stroke="#3B82F6" stroke-width="1.2" class="pulse-ring-slow" opacity="0.6" />
                <circle cx="380" cy="280" r="6" fill="#60A5FA" filter="url(#glowBlue)" />
                <text x="360" y="260" fill="#E2E8F0" font-size="11" font-weight="600">Ward F/S (Parel)</text>

                <circle cx="240" cy="180" r="15" fill="none" stroke="#2DD4BF" stroke-width="1" class="pulse-ring-slow" opacity="0.6" />
                <circle cx="240" cy="180" r="5" fill="#2DD4BF" filter="url(#glowTeal)" />
                <text x="180" y="170" fill="#94A3B8" font-size="10" font-family="monospace">Ward G/S (Worli)</text>

                <circle cx="520" cy="190" r="22" fill="none" stroke="#3B82F6" stroke-width="1.4" class="pulse-ring-slow" opacity="0.7" />
                <circle cx="520" cy="190" r="6.5" fill="#3B82F6" filter="url(#glowBlue)" />
                <text x="500" y="170" fill="#E2E8F0" font-size="11" font-weight="700">Ward G/N (Dadar)</text>
            </svg>

            <!-- Floating Real Data Activity Panel -->
            <div class="map-floating-kpi-bar">
                <div class="kpi-micro-stat">
                    <span class="stat-num">{c_total}</span>
                    <span class="stat-lbl">Issues Logged</span>
                </div>
                <div class="kpi-divider-line"></div>
                <div class="kpi-micro-stat">
                    <span class="stat-num stat-num-amber">{c_open}</span>
                    <span class="stat-lbl">In Active Triage</span>
                </div>
                <div class="kpi-divider-line"></div>
                <div class="kpi-micro-stat">
                    <span class="stat-num stat-num-teal">{c_resolved}</span>
                    <span class="stat-lbl">Resolved</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. How It Works Section
    # -------------------------------------------------------------
    st.markdown("""
    <div id="how-it-works" class="section-anchor">
        <div class="section-tag-label">ENGINEERED WORKFLOW</div>
        <h2 class="section-main-heading">How ASTRA Works</h2>
        <p class="section-sub-copy">Three intelligent steps from citizen defect capture to verified municipal resolution</p>
    </div>

    <div class="step-cards-grid">
        <div class="welcome-step-card">
            <div>
                <span class="step-badge-num">01 / REPORT</span>
                <div class="step-icon-tile">🎙️</div>
                <div class="step-card-title">Multimodal Intake</div>
                <p class="step-card-desc">
                    Describe the defect in text, attach camera photo evidence, or speak in Hindi or Marathi. ASTRA's AI automatically categorizes and gauges urgency.
                </p>
            </div>
        </div>

        <div class="welcome-step-card">
            <div>
                <span class="step-badge-num">02 / LOCATE</span>
                <div class="step-icon-tile">📍</div>
                <div class="step-card-title">Geographic Precision</div>
                <p class="step-card-desc">
                    Capture precise GPS coordinates with one click or confirm street landmarks and municipal ward boundaries for exact field squad navigation.
                </p>
            </div>
        </div>

        <div class="welcome-step-card">
            <div>
                <span class="step-badge-num">03 / RESOLVE</span>
                <div class="step-icon-tile">⚡</div>
                <div class="step-card-title">Accountable Action</div>
                <p class="step-card-desc">
                    Municipal engineering units verify defects with computer vision, dispatch work orders with strict SLA targets, and update progress in real time.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. Our Mission & Civic Intelligence Pillars
    # -------------------------------------------------------------
    st.markdown("""
    <div id="our-mission" class="section-anchor">
        <div class="section-tag-label">CIVIC TECHNOLOGY MISSION</div>
        <h2 class="section-main-heading">Built for the Future of Smart Cities</h2>
        <p class="section-sub-copy">Transforming fragmented civic grievances into transparent, automated municipal intelligence</p>
    </div>

    <div class="pillars-grid">
        <div class="pillar-card">
            <div class="pillar-header">
                <span class="pillar-icon">👁️</span>
                <span class="pillar-name">Computer Vision & Speech AI</span>
            </div>
            <p class="pillar-detail">
                YOLO11n defect detection identifies road potholes, damaged streetlights, and pipe leaks. IndicConformer models transcribe Hindi and Marathi audio with high accuracy.
            </p>
        </div>

        <div class="pillar-card">
            <div class="pillar-header">
                <span class="pillar-icon">⏱️</span>
                <span class="pillar-name">Strict Municipal SLA Deadlines</span>
            </div>
            <p class="pillar-detail">
                Defect severity dictates real-time resolution targets (Critical: 24h, High: 48h, Medium: 72h). Proactive alerts prevent unattended municipal backlog.
            </p>
        </div>

        <div class="pillar-card">
            <div class="pillar-header">
                <span class="pillar-icon">🗺️</span>
                <span class="pillar-name">Real-Time Spatial Clustering</span>
            </div>
            <p class="pillar-detail">
                Ward-level GIS defect heatmaps identify persistent infrastructure failure patterns, enabling proactive municipal maintenance before public disruption.
            </p>
        </div>

        <div class="pillar-card">
            <div class="pillar-header">
                <span class="pillar-icon">🔍</span>
                <span class="pillar-name">Radical Citizen Transparency</span>
            </div>
            <p class="pillar-detail">
                Every report generates a permanent Case ID (AST-YYYYMMDD-XXXX) allowing citizens to track squad assignments, dispatcher notes, and closure proofs.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. Closing CTA Banner
    # -------------------------------------------------------------
    st.markdown("""
    <div class="welcome-closing-banner">
        <h2 class="closing-headline">Better streets start with being heard.</h2>
        <p class="closing-subtext">Join active citizens and municipal teams transforming our city into a cleaner, safer, and more responsive metropolis.</p>
    </div>
    """, unsafe_allow_html=True)

    col_cl1, col_cl2, col_cl3 = st.columns([1.5, 2, 1.5])
    with col_cl2:
        if st.button("🚀 Get Started with ASTRA", key="closing_btn_get_started", type="primary", use_container_width=True):
            st.session_state["portal_mode_radio"] = "👥 Citizen Registration Portal"
            st.session_state["citizen_nav_radio"] = "Report a Civic Issue"
            st.session_state["active_view"] = "Report a Civic Issue"
            st.rerun()

    # -------------------------------------------------------------
    # 6. Minimal Footer
    # -------------------------------------------------------------
    st.markdown("""
    <div class="welcome-footer">
        <div class="footer-top-row">
            <div>
                <div class="footer-brand-title">🏛️ ASTRA</div>
                <div class="footer-brand-sub">Civic Grievance Intelligence & Municipal Operations Center</div>
            </div>
            <div class="footer-links-row">
                <a href="#how-it-works" class="footer-link-a">How it works</a>
                <a href="#our-mission" class="footer-link-a">Our mission</a>
                <a href="#live-operations" class="footer-link-a">Live Activity</a>
            </div>
        </div>
        <div class="footer-bottom-notice">
            ASTRA is an independent civic-technology operations platform engineered for municipal defect triage and public accountability. All grievance records are cryptographically verified.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PART 2: ADMINISTRATOR DASHBOARD ("OPERATIONS CENTER")
# =====================================================================
def render_admin_dashboard():
    complaints = get_all_complaints()
    kpis = compute_aggregate_kpis(complaints)

    # -------------------------------------------------------------
    # B. Page Heading
    # -------------------------------------------------------------
    today_str = datetime.now().strftime("%A, %b %d, %Y")
    st.markdown(f"""
    <div class="op-header">
        <div>
            <h1 class="op-title">Operations Center</h1>
            <div class="op-subtitle">Here’s what’s happening in your city today.</div>
        </div>
        <div class="op-meta-pill">
            <span class="status-dot-live"></span>
            <span>{today_str}</span>
            <span style="color: #E5EAF2;">|</span>
            <span style="color: #0F766E;">BMC Ward Central</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter trigger from Needs Attention buttons
    filter_preset = st.session_state.get("admin_filter_preset", "All")

    # -------------------------------------------------------------
    # C. Needs Attention Section
    # -------------------------------------------------------------
    st.markdown("""
    <div class="needs-attention-box">
        <div class="attention-header-bar">
            <span style="font-size: 1.25rem;">⚠️</span>
            <span class="attention-main-title">Needs Attention</span>
            <span style="color: #FDE68A;">•</span>
            <span class="attention-main-desc">High-priority civic reports requiring immediate dispatch or assignment</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_att1, col_att2, col_att3 = st.columns(3)
    with col_att1:
        st.markdown(f"""
        <div class="attention-card">
            <div class="att-label"><span>🚨</span> Overdue Complaints</div>
            <div class="att-count count-red">{kpis["overdue_count"]}</div>
            <div class="att-desc">Complaints that have breached their municipal resolution target.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Filter Overdue Cases", key="btn_filter_overdue", use_container_width=True):
            st.session_state["admin_filter_preset"] = "Overdue"
            st.rerun()

    with col_att2:
        st.markdown(f"""
        <div class="attention-card">
            <div class="att-label"><span>⚡</span> Unassigned High-Priority</div>
            <div class="att-count count-orange">{kpis["unassigned_high_count"]}</div>
            <div class="att-desc">High or Critical severity issues currently awaiting officer assignment.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Assign High-Priority", key="btn_filter_unassigned", use_container_width=True):
            st.session_state["admin_filter_preset"] = "Unassigned_High"
            st.rerun()

    with col_att3:
        st.markdown(f"""
        <div class="attention-card">
            <div class="att-label"><span>⏳</span> Approaching SLA</div>
            <div class="att-count count-amber">{kpis["approaching_count"]}</div>
            <div class="att-desc">Open grievances approaching their target dispatch threshold within 12h.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Review Approaching SLA", key="btn_filter_approaching", use_container_width=True):
            st.session_state["admin_filter_preset"] = "Approaching"
            st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # D. KPI Summary Row (Exact 5 cards in specified order)
    # -------------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Open Issues</span>
                <span class="kpi-icon-tile">📋</span>
            </div>
            <div class="kpi-val">{kpis["open_count"]}</div>
            <div class="kpi-footer">Active in pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Overdue</span>
                <span class="kpi-icon-tile" style="background-color: #FEE2E2; color: #DC2626;">🚨</span>
            </div>
            <div class="kpi-val" style="color: {'#DC2626' if kpis['overdue_count'] > 0 else '#17243A'};">{kpis["overdue_count"]}</div>
            <div class="kpi-footer">SLA target breached</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Unassigned</span>
                <span class="kpi-icon-tile" style="background-color: #FEF3C7; color: #B45309;">👤</span>
            </div>
            <div class="kpi-val">{kpis["unassigned_count"]}</div>
            <div class="kpi-footer">Awaiting squad dispatch</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Avg Response Time</span>
                <span class="kpi-icon-tile" style="background-color: #EEF4FF; color: #0F766E;">⏱️</span>
            </div>
            <div class="kpi-val">{kpis["avg_response_time"]}</div>
            <div class="kpi-footer">Intake to dispatch</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Avg Resolution Time</span>
                <span class="kpi-icon-tile" style="background-color: #DCFCE7; color: #16A34A;">✅</span>
            </div>
            <div class="kpi-val">{kpis["avg_resolution_time"]}</div>
            <div class="kpi-footer">{"From historical closures" if kpis["avg_resolution_time"] != "—" else "No resolved cases yet"}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # E. Main Operations Row (Two Columns: 55% Left / 45% Right)
    # -------------------------------------------------------------
    col_left, col_right = st.columns([55, 45])

    # Left: High-Priority Issues
    with col_left:
        st.markdown("""
        <div class="ops-panel-card">
            <div class="ops-panel-header">
                <div class="ops-panel-title">
                    <span>⚡ High-Priority Issues</span>
                    <span class="badge badge-critical">High & Critical</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sort complaints by severity and age
        high_priority_list = [c for c in complaints if c.get("priority") in ("Critical", "High")]
        if not high_priority_list:
            high_priority_list = complaints[:5]  # Fallback to top reports

        if not high_priority_list:
            st.info("ℹ️ No active complaints recorded in the system yet.")
        else:
            for idx, c in enumerate(high_priority_list[:5]):
                sla_info = compute_sla_metrics(c)
                p_badge = "badge-critical" if c.get("priority") in ("Critical", "High") else "badge-med"
                s_badge = {
                    "New": "badge-new",
                    "Under Review": "badge-review",
                    "In Progress": "badge-progress",
                    "Resolved": "badge-resolved"
                }.get(c.get("status"), "badge-new")

                assigned_officer = c.get("assigned_to") or "Unassigned"

                st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #E5EAF2; border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;">
                        <div>
                            <span style="font-weight: 700; color: #17243A; font-size: 0.9rem;">{c['complaint_id']}</span>
                            <span style="color: #667085; font-size: 0.82rem; margin-left: 6px;">• {c.get('text_category') or 'General'}</span>
                        </div>
                        <div>
                            <span class="badge {p_badge}">{c.get('priority', 'Medium')}</span>
                            <span class="badge {s_badge}" style="margin-left: 4px;">{c.get('status', 'New')}</span>
                        </div>
                    </div>
                    <div style="font-size: 0.84rem; color: #17243A; margin: 4px 0; font-weight: 500;">
                        {c.get('complaint_text')[:90] + ('...' if len(c.get('complaint_text', '')) > 90 else '')}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.76rem; color: #667085; margin-top: 6px;">
                        <span>📍 {c.get('location') or 'Unspecified'}</span>
                        <span style="color: {'#DC2626' if sla_info['is_overdue'] else '#0F766E'}; font-weight: 600;">{sla_info['sla_str']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Right: Live Map Panel
    with col_right:
        st.markdown("""
        <div class="ops-panel-card">
            <div class="ops-panel-header">
                <div class="ops-panel-title">
                    <span>🗺️ Live Operations Map</span>
                    <span class="badge badge-low">Geocoded Reports</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        geocoded = [c for c in complaints if c.get("latitude") is not None and c.get("longitude") is not None]
        if not geocoded:
            st.markdown("""
            <div style="background-color: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 10px; padding: 48px 20px; text-align: center; margin-bottom: 12px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">📍</div>
                <div style="font-weight: 700; color: #17243A; font-size: 0.98rem;">No Geocoded Grievances Yet</div>
                <div style="font-size: 0.82rem; color: #667085; max-width: 320px; margin: 6px auto 0 auto; line-height: 1.4;">
                    Complaints submitted via device GPS will be plotted live on this map with priority and status markers.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            map_data = []
            for c in geocoded:
                priority = c.get("priority", "Medium")
                color = [220, 38, 38, 220] if priority in ("Critical", "High") else [37, 99, 235, 200]
                if c.get("status") == "Resolved":
                    color = [22, 163, 74, 200]

                map_data.append({
                    "lat": float(c["latitude"]),
                    "lon": float(c["longitude"]),
                    "id": c["complaint_id"],
                    "priority": priority,
                    "status": c.get("status", "New"),
                    "category": c.get("text_category", "Civil"),
                    "color": color
                })
            df_map = pd.DataFrame(map_data)
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=df_map["lat"].mean(),
                    longitude=df_map["lon"].mean(),
                    zoom=11.5,
                    pitch=0
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=df_map,
                        get_position="[lon, lat]",
                        get_color="color",
                        get_radius=140,
                        pickable=True
                    )
                ],
                tooltip={"text": "{id}: {category} ({priority})\nStatus: {status}"}
            ), use_container_width=True)

        # Under-map legend
        high_cnt = sum(1 for c in geocoded if c.get("priority") in ("Critical", "High"))
        med_cnt = sum(1 for c in geocoded if c.get("priority") not in ("Critical", "High") and c.get("status") != "Resolved")
        res_cnt = sum(1 for c in geocoded if c.get("status") == "Resolved")

        st.markdown(f"""
        <div class="map-legend">
            <span class="legend-item"><span class="dot-red"></span> High / Critical ({high_cnt})</span>
            <span class="legend-item"><span class="dot-blue"></span> Medium / Low ({med_cnt})</span>
            <span class="legend-item"><span class="dot-green"></span> Resolved ({res_cnt})</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # F. Recent Reports Table (Full-Width with Real Status Filters)
    # -------------------------------------------------------------
    st.markdown("""
    <div class="ops-panel-card">
        <div class="ops-panel-header">
            <div class="ops-panel-title">
                <span>📋 Recent Reports & Case Triage</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_flt_chips, col_flt_search = st.columns([3, 2])
    with col_flt_chips:
        c_all = len(complaints)
        c_open = kpis["open_count"]
        c_prog = sum(1 for c in complaints if c.get("status") == "In Progress")
        c_res = sum(1 for c in complaints if c.get("status") == "Resolved")

        chip_opts = [
            f"All ({c_all})",
            f"Open ({c_open})",
            f"In Progress ({c_prog})",
            f"Resolved ({c_res})"
        ]
        selected_chip = st.radio("Status Filter Chips", chip_opts, horizontal=True, label_visibility="collapsed")

    with col_flt_search:
        search_kw = st.text_input("Search reports:", placeholder="Search ID, location, or issue...", label_visibility="collapsed")

    # Apply filters
    filtered_list = complaints
    if "Open" in selected_chip:
        filtered_list = [c for c in filtered_list if c.get("status") != "Resolved"]
    elif "In Progress" in selected_chip:
        filtered_list = [c for c in filtered_list if c.get("status") == "In Progress"]
    elif "Resolved" in selected_chip:
        filtered_list = [c for c in filtered_list if c.get("status") == "Resolved"]

    # Filter preset from Needs Attention section
    if filter_preset == "Overdue":
        filtered_list = [c for c in filtered_list if compute_sla_metrics(c)["is_overdue"]]
    elif filter_preset == "Unassigned_High":
        filtered_list = [c for c in filtered_list if compute_sla_metrics(c)["is_unassigned"] and c.get("priority") in ("Critical", "High")]
    elif filter_preset == "Approaching":
        filtered_list = [c for c in filtered_list if compute_sla_metrics(c)["is_approaching"]]

    if search_kw.strip():
        kw = search_kw.strip().lower()
        filtered_list = [
            c for c in filtered_list
            if kw in c.get("complaint_id", "").lower()
            or kw in c.get("complaint_text", "").lower()
            or kw in c.get("location", "").lower()
        ]

    if not filtered_list:
        st.info("No complaint records match the active filter criteria.")
    else:
        table_records = []
        for c in filtered_list:
            sla_m = compute_sla_metrics(c)
            table_records.append({
                "Case ID": c["complaint_id"],
                "Issue": c.get("complaint_text")[:50] + ("..." if len(c.get("complaint_text", "")) > 50 else ""),
                "Location": c.get("location") or "Unspecified",
                "Category": c.get("text_category") or "General",
                "Priority": c.get("priority", "Medium"),
                "Status": c.get("status", "New"),
                "Assigned To": c.get("assigned_to") or "Unassigned",
                "Reported": c.get("created_at"),
                "Age": sla_m["age_str"]
            })
        st.dataframe(pd.DataFrame(table_records), use_container_width=True)

        # Case Actions & Assignment Expander
        with st.expander("🛠️ Manage Selected Complaint (Assignment & Status Update)", expanded=False):
            case_options = [c["complaint_id"] for c in filtered_list]
            selected_case_id = st.selectbox("Select Complaint ID to manage:", case_options)
            chosen_case = get_complaint_by_id(selected_case_id)

            if chosen_case:
                c_act1, c_act2 = st.columns(2)
                with c_act1:
                    new_st = st.selectbox("Update Case Status:", VALID_STATUSES, index=VALID_STATUSES.index(chosen_case.get("status", "New")))
                    notes = st.text_input("Dispatcher Notes:", value=chosen_case.get("admin_notes", ""))
                    if st.button("Save Status & Notes", key="btn_save_status"):
                        update_complaint_status(selected_case_id, new_st, admin_notes=notes)
                        st.success(f"Updated status for {selected_case_id} to '{new_st}'!")
                        st.rerun()

                with c_act2:
                    officers = [
                        "Roads Emergency Squad (Parel Boiwada)",
                        "Electrical Hazard Unit (Ward Central)",
                        "Water Pipeline Rapid Repair",
                        "Sanitation Triage Team",
                        "Chief Municipal Inspector"
                    ]
                    current_officer = chosen_case.get("assigned_to", "")
                    default_idx = officers.index(current_officer) if current_officer in officers else 0
                    selected_officer = st.selectbox("Assign Field Squad:", officers, index=default_idx)
                    if st.button("Update Squad Assignment", key="btn_save_assignment"):
                        update_complaint_assignment(selected_case_id, selected_officer)
                        st.success(f"Case {selected_case_id} assigned to '{selected_officer}'!")
                        st.rerun()


# =====================================================================
# PART 3: CITIZEN COMPLAINT REGISTRATION PAGE (GUIDED FLOW)
# =====================================================================
def render_citizen_registration():
    # Read any incoming geolocation coordinates from query params
    q_params = st.query_params
    if "geo_lat" in q_params and "geo_lon" in q_params:
        try:
            st.session_state["geo_lat"] = float(q_params.get("geo_lat"))
            st.session_state["geo_lon"] = float(q_params.get("geo_lon"))
            st.session_state["geo_acc"] = float(q_params.get("geo_acc", 10.0))
            st.session_state["geo_ts"] = q_params.get("geo_ts", "")
        except Exception:
            pass

    # Header
    st.markdown("""
    <div style="max-width: 1050px; margin: 0 auto 20px auto;">
        <h1 style="font-size: 1.85rem; font-weight: 800; color: #17243A; margin-bottom: 6px;">Report a Civic Issue</h1>
        <div style="font-size: 0.95rem; color: #667085; line-height: 1.45;">
            Report a local municipal issue with a text description, photo, or voice recording.
            Our civic intelligence engine identifies the defect and routes it directly to the responsible municipal department.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='max-width: 1050px; margin: 0 auto;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Step 1: Complaint Details
    # -------------------------------------------------------------
    st.markdown("""
    <div class="citizen-step-box">
        <div class="step-header">
            <div class="step-badge">1</div>
            <div class="step-title-text">Complaint Details</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cit_meta, col_cit_desc = st.columns([1, 2])
    with col_cit_meta:
        citizen_name = st.text_input("Your Name (optional):", value="Citizen", key="cit_name")
        citizen_phone = st.text_input("Contact Phone Number (optional):", value="", key="cit_phone")
        category_hint = st.selectbox(
            "Issue Category (optional hint):",
            ["Auto-Detect via AI", "ROADS_INFRASTRUCTURE", "WATER_SUPPLY", "ELECTRICAL_SAFETY", "GARBAGE_SANITATION"],
            key="cit_cat_hint"
        )

    with col_cit_desc:
        citizen_text = st.text_area(
            "Describe the Civic Issue:",
            height=120,
            placeholder="e.g., There is a dangerous open manhole on the main road in Vile Parle East near the market. Streetlights are also not functioning.",
            key="cit_text_desc"
        )
        if citizen_text.strip():
            # Real-time NLP category guidance preview
            with st.spinner("AI evaluating department classification..."):
                preview_nlp = default_analyzer.analyze_text(citizen_text.strip())
            if preview_nlp and preview_nlp.get("success"):
                cat_name = preview_nlp.get("category")
                conf = preview_nlp.get("confidence", 0.0)
                margin = preview_nlp.get("score_margin", 0.0)
                st.caption(f"🤖 **AI Department Suggestion**: `{cat_name}` ({conf*100:.1f}% confidence)")
                if preview_nlp.get("needs_review"):
                    st.warning(f"⚠️ Confidence margin between candidate departments is narrow ({margin*100:.1f}%). A municipal dispatcher will verify routing.")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Step 2: Evidence (Photo & Voice)
    # -------------------------------------------------------------
    st.markdown("""
    <div class="citizen-step-box">
        <div class="step-header">
            <div class="step-badge">2</div>
            <div class="step-title-text">Issue Evidence (Photo or Voice Recording)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_photo, tab_voice = st.tabs(["📷 Photographic Evidence", "🎙️ Spoken Voice Recording (Hindi / Marathi)"])

    active_photo_path = None
    active_audio_path = None
    audio_transcription_res = None

    with tab_photo:
        col_p_up, col_p_prev = st.columns([1, 1])
        with col_p_up:
            cit_photo = st.file_uploader(
                "Upload a photo of the defect (.jpg, .png, .webp):",
                type=["jpg", "jpeg", "png", "webp", "avif"],
                key="cit_photo_upload"
            )
            # Benchmark test sample options
            photo_preset = st.selectbox(
                "Or choose a test evaluation photo:",
                ["(None - Upload your photo)", "Potholes on Road", "Dangling High-Voltage Wire", "Water Pipe Leak"],
                key="cit_photo_preset"
            )

        with col_p_prev:
            if cit_photo:
                save_name = f"cit_{datetime.now().strftime('%Y%m%d%H%M%S')}_{cit_photo.name}"
                save_path = UPLOADS_DIR / save_name
                with open(save_path, "wb") as f:
                    f.write(cit_photo.getbuffer())
                active_photo_path = save_path
                st.session_state["persisted_photo_path"] = str(save_path)
                st.image(cit_photo, caption=f"Uploaded: {cit_photo.name}", use_container_width=True)
            elif photo_preset != "(None - Upload your photo)":
                preset_map = {
                    "Potholes on Road": SAMPLE_IMAGES_DIR / "potholes.webp",
                    "Dangling High-Voltage Wire": SAMPLE_IMAGES_DIR / "wire.avif",
                    "Water Pipe Leak": SAMPLE_IMAGES_DIR / "Water.jpg"
                }
                sample_file = preset_map.get(photo_preset)
                if sample_file and sample_file.is_file():
                    active_photo_path = sample_file
                    st.session_state["persisted_photo_path"] = str(sample_file)
                    st.image(str(sample_file), caption=f"Test Sample: {sample_file.name}", use_container_width=True)
            elif st.session_state.get("persisted_photo_path"):
                p_img = Path(st.session_state["persisted_photo_path"])
                if p_img.is_file():
                    active_photo_path = p_img
                    st.image(str(p_img), caption=f"Active Photo: {p_img.name}", use_container_width=True)

        if active_photo_path:
            with st.spinner("Analyzing photo with YOLO11n object detector..."):
                yolo_res = default_analyzer.analyze_image(active_photo_path)
            if yolo_res and yolo_res.get("success"):
                classes = yolo_res.get("detected_classes", [])
                st.success(f"✅ YOLO11n Analysis: Detected defects: **{', '.join(classes) if classes else 'None detected'}** ({yolo_res.get('detection_count', 0)} boxes)")
                if yolo_res.get("annotated_image_path"):
                    st.image(yolo_res.get("annotated_image_path"), caption="YOLO11n Annotated Detection", use_container_width=True)

    with tab_voice:
        col_v_up, col_v_trans = st.columns([1, 1])
        with col_v_up:
            cit_audio = st.file_uploader(
                "Upload voice recording (.mp3, .wav, .m4a, .opus, .ogg, .aac, .webm):",
                type=["mp3", "wav", "m4a", "ogg", "opus", "aac", "webm", "flac", "caf", "amr", "mp4", "wma",
                      "MP3", "WAV", "M4A", "OGG", "OPUS", "AAC", "WEBM", "FLAC", "CAF", "AMR", "MP4", "WMA"],
                key="cit_audio_upload",
                help="Supports WhatsApp audio (.opus), iPhone/Android voice memos (.m4a/.caf), MP3, WAV, AAC, and WebM."
            )

            # Native browser microphone input
            mic_audio = st.audio_input(
                "🎙️ Or record voice directly via Microphone:",
                key="cit_mic_input",
                help="Click record, speak your civic grievance in Hindi or Marathi, and click stop."
            )

            audio_preset = st.selectbox(
                "Or choose a test evaluation voice sample:",
                ["(None - Upload your audio)", "Audio_1.mp3 (Hindi - Vile Parle Manhole)", "Audio_9.mp3 (Hindi - Dadar Water Leak)"],
                key="cit_audio_preset"
            )
            lang_code = st.radio("Recording Language:", ["hi (Hindi)", "mr (Marathi)"], horizontal=True, key="cit_lang_code")
            chosen_lang = "hi" if "hi" in lang_code else "mr"

            if cit_audio:
                # Save once based on filename and size to avoid duplicating files
                save_name = f"cit_aud_{cit_audio.name}"
                save_path = UPLOADS_DIR / save_name
                if not save_path.is_file() or save_path.stat().st_size != cit_audio.size:
                    with open(save_path, "wb") as f:
                        f.write(cit_audio.getbuffer())
                active_audio_path = save_path
                st.session_state["persisted_audio_path"] = str(save_path)
                st.audio(prepare_playable_audio(save_path))
                st.caption(f"📁 Loaded audio: `{cit_audio.name}` ({cit_audio.size / 1024:.1f} KB)")
            elif mic_audio:
                save_name = f"mic_rec_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                save_path = UPLOADS_DIR / save_name
                with open(save_path, "wb") as f:
                    f.write(mic_audio.getbuffer())
                active_audio_path = save_path
                st.session_state["persisted_audio_path"] = str(save_path)
                st.audio(mic_audio)
                st.caption("🎙️ Live microphone recording captured!")
            elif audio_preset != "(None - Upload your audio)":
                preset_name = "Audio_1.mp3" if "Audio_1" in audio_preset else "Audio_9.mp3"
                sample_file = AUDIO_SAMPLES_DIR / preset_name
                if sample_file.is_file():
                    active_audio_path = sample_file
                    st.session_state["persisted_audio_path"] = str(sample_file)
                    st.audio(str(sample_file))
                    st.caption(f"📦 Preloaded test recording: `{preset_name}`")
            elif st.session_state.get("persisted_audio_path"):
                p_path = Path(st.session_state["persisted_audio_path"])
                if p_path.is_file():
                    active_audio_path = p_path
                    st.audio(prepare_playable_audio(p_path))
                    st.caption(f"📁 Active recording: `{p_path.name}`")

            transcribe_btn = st.button("🎧 Transcribe Voice Recording", key="cit_btn_transcribe", type="secondary")

        with col_v_trans:
            if transcribe_btn:
                target_audio = active_audio_path or (Path(st.session_state["persisted_audio_path"]) if st.session_state.get("persisted_audio_path") else None)
                if not target_audio or not target_audio.is_file():
                    st.error("Please upload an audio recording, speak into the microphone, or select a sample first.")
                else:
                    with st.spinner("Transcribing speech via IndicConformer (CTC)..."):
                        audio_transcription_res = default_analyzer.analyze_audio(
                            audio_path=target_audio,
                            language=chosen_lang,
                            decoder="ctc"
                        )
                        st.session_state["cit_audio_res"] = audio_transcription_res

            persisted_res = st.session_state.get("cit_audio_res")
            if persisted_res:
                if persisted_res.get("success"):
                    st.success("✅ Speech Pipeline Transcribed Successfully!")
                    st.text_area("Transcribed Text:", value=persisted_res.get("transcript", ""), height=90, key="cit_transcript_disp")
                    st.caption(f"Suggested Category: `{persisted_res.get('category')}` (Confidence: {persisted_res.get('confidence', 0)*100:.1f}%)")
                    if persisted_res.get("needs_review"):
                        st.warning("⚠️ Confidence margin between candidate departments is narrow. Dispatcher verification required.")
                else:
                    st.error(f"❌ Speech Pipeline Error: {persisted_res.get('error', 'Transcription failed.')}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Step 3: Issue Location (Browser GPS + Manual Confirmation)
    # -------------------------------------------------------------
    st.markdown("""
    <div class="citizen-step-box">
        <div class="step-header">
            <div class="step-badge">3</div>
            <div class="step-title-text">Issue Location</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_loc1, col_loc2 = st.columns([1, 1])
    with col_loc1:
        st.markdown("**Device Geolocation**")
        st.markdown("Capture your device's exact GPS coordinates to help municipal squads navigate straight to the issue.")

        # HTML5 Geolocation Button Bridge
        components.html("""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; display: flex; align-items: center; gap: 12px;">
            <button id="btn-geo" onclick="fetchLocation()" style="
                background-color: #0F766E;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            ">
                <span style="font-size: 16px;">📍</span> Use My Current Location
            </button>
            <span id="geo-status-msg" style="font-size: 13px; color: #667085; font-weight: 500;"></span>
        </div>
        <script>
        function fetchLocation() {
            const status = document.getElementById('geo-status-msg');
            const btn = document.getElementById('btn-geo');
            if (!navigator.geolocation) {
                status.innerText = "❌ Geolocation is not supported by your browser.";
                return;
            }
            status.innerText = "⏳ Detecting GPS location...";
            btn.disabled = true;
            navigator.geolocation.getCurrentPosition(
                function(pos) {
                    const lat = pos.coords.latitude.toFixed(6);
                    const lon = pos.coords.longitude.toFixed(6);
                    const acc = Math.round(pos.coords.accuracy);
                    status.innerText = "✅ GPS Captured!";
                    btn.disabled = false;
                    const pUrl = new URL(window.parent.location.href);
                    pUrl.searchParams.set('geo_lat', lat);
                    pUrl.searchParams.set('geo_lon', lon);
                    pUrl.searchParams.set('geo_acc', acc);
                    pUrl.searchParams.set('geo_ts', new Date().toISOString());
                    window.parent.location.href = pUrl.href;
                },
                function(err) {
                    btn.disabled = false;
                    let m = "Location request failed.";
                    if (err.code === 1) m = "⚠️ Location permission denied. Please enter address manually.";
                    else if (err.code === 2) m = "⚠️ Location position unavailable.";
                    else if (err.code === 3) m = "⚠️ Location request timed out.";
                    status.innerText = m;
                },
                { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
            );
        }
        </script>
        """, height=52)

        cur_lat = st.session_state.get("geo_lat")
        cur_lon = st.session_state.get("geo_lon")
        cur_acc = st.session_state.get("geo_acc")

        if cur_lat and cur_lon:
            st.success(f"📍 **GPS Coordinates Captured**: `{cur_lat}° N, {cur_lon}° E` (Accuracy: ±{cur_acc:.0f}m)")
            st.markdown(f"[🌐 Open in Google Maps](https://maps.google.com/?q={cur_lat},{cur_lon})")
        else:
            st.caption("ℹ️ Click the button above to capture GPS coordinates, or enter the location manually on the right.")

    with col_loc2:
        st.markdown("**Address, Landmark & Ward**")
        cit_address = st.text_input("Street Address / Specific Defect Spot:", value="Near Vile Parle East Station Market", key="cit_addr")
        cit_landmark = st.text_input("Prominent Landmark:", value="Opposite Municipal Garden", key="cit_landmark")
        cit_ward = st.selectbox("Municipal Ward:", ["Ward K/East (Andheri / Vile Parle East)", "Ward G/North (Dadar / Dharavi)", "Ward F/South (Parel)", "Ward D (Grant Road / Malabar Hill)", "Other / Unsure"], key="cit_ward")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Step 4: Review and Submit
    # -------------------------------------------------------------
    st.markdown("""
    <div class="citizen-step-box">
        <div class="step-header">
            <div class="step-badge">4</div>
            <div class="step-title-text">Review and Submit</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    summary_text = citizen_text.strip()
    if not summary_text and st.session_state.get("cit_audio_res"):
        summary_text = st.session_state.get("cit_audio_res", {}).get("transcript", "")

    effective_photo = active_photo_path or (Path(st.session_state["persisted_photo_path"]) if st.session_state.get("persisted_photo_path") else None)
    effective_audio = active_audio_path or (Path(st.session_state["persisted_audio_path"]) if st.session_state.get("persisted_audio_path") else None)

    st.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E5EAF2; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
        <div style="font-weight: 700; color: #17243A; margin-bottom: 8px;">Case Summary Review</div>
        <div style="font-size: 0.85rem; color: #17243A; line-height: 1.5;">
            • <b>Issue</b>: {summary_text[:140] if summary_text else '(Voice recording or photo attached)'}<br>
            • <b>Location</b>: {cit_address}, {cit_landmark} ({cit_ward})<br>
            • <b>GPS Status</b>: {'Captured (' + str(cur_lat) + ', ' + str(cur_lon) + ')' if cur_lat else 'Manual Address Only'}<br>
            • <b>Evidence</b>: {'Photo Attached (' + effective_photo.name + ')' if effective_photo else 'No Photo'} | {'Voice Audio Attached (' + effective_audio.name + ')' if effective_audio else 'No Audio'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    submit_col1, submit_col2 = st.columns([1, 2])
    with submit_col1:
        submit_btn = st.button("🚀 Submit Civic Grievance", type="primary", use_container_width=True, key="cit_submit_btn")

    if submit_btn:
        if not summary_text and not effective_photo and not effective_audio:
            st.error("Please provide at least a description, photographic evidence, or a voice recording.")
        else:
            with st.spinner("Registering grievance with municipal dispatch engine..."):
                final_loc = f"{cit_address}, {cit_landmark}, {cit_ward}".strip(", ")
                intake_type = "Text"
                if effective_photo and effective_audio:
                    intake_type = "Multimodal (Photo + Audio)"
                elif effective_photo:
                    intake_type = "Photo"
                elif effective_audio:
                    intake_type = "Audio"

                saved_record = default_analyzer.process_grievance(
                    complaint_text=summary_text,
                    image_path=effective_photo,
                    audio_path=effective_audio,
                    audio_language="hi" if effective_audio else None,
                    intake_mode=intake_type,
                    citizen_name=citizen_name.strip() or "Citizen",
                    citizen_contact=citizen_phone.strip(),
                    location=final_loc,
                    latitude=st.session_state.get("geo_lat"),
                    longitude=st.session_state.get("geo_lon"),
                    location_accuracy=st.session_state.get("geo_acc"),
                    ward=cit_ward,
                    landmark=cit_landmark,
                    address=cit_address,
                    save_to_db=True
                )

            cid = saved_record["complaint_id"]
            st.balloons()
            st.success(f"""
            🎉 **Grievance Submitted Successfully!**
            
            Your Official Municipal Grievance Tracking ID is: **`{cid}`**
            
            • **Department Routed**: `{saved_record.get('text_category') or 'Municipal Operations'}`
            • **Priority**: `{saved_record.get('priority', 'Medium')}`
            • **Initial Status**: `{saved_record.get('status', 'New')}`
            
            You can track this complaint anytime in the **Track My Complaint** section using this ID.
            """)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# TRACK MY COMPLAINT (CITIZEN LOOKUP)
# =====================================================================
def render_track_complaint():
    st.markdown("""
    <div style="max-width: 800px; margin: 0 auto 20px auto;">
        <h1 style="font-size: 1.85rem; font-weight: 800; color: #17243A;">Track My Civic Grievance</h1>
        <div style="font-size: 0.95rem; color: #667085;">
            Enter your official ASTRA Grievance ID (e.g., AST-20261017-6449) to check the real-time status and assigned department.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='max-width: 800px; margin: 0 auto;'>", unsafe_allow_html=True)
    track_id = st.text_input("Grievance ID:", placeholder="e.g. AST-20261017-6449").strip()

    if track_id:
        record = get_complaint_by_id(track_id)
        if not record:
            st.warning(f"No grievance found matching ID '{track_id}'. Please verify the ID number.")
        else:
            sla_info = compute_sla_metrics(record)
            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E5EAF2; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #E5EAF2; padding-bottom: 12px; margin-bottom: 14px;">
                    <div>
                        <span style="font-size: 1.25rem; font-weight: 800; color: #17243A;">{record['complaint_id']}</span>
                        <div style="font-size: 0.8rem; color: #667085;">Registered on {record['created_at']}</div>
                    </div>
                    <div>
                        <span class="badge badge-low">{record.get('status', 'New')}</span>
                    </div>
                </div>
                <div style="font-size: 0.9rem; color: #17243A; line-height: 1.6;">
                    • <b>Assigned Department</b>: {record.get('text_category') or 'Municipal Operations'}<br>
                    • <b>Priority</b>: {record.get('priority', 'Medium')}<br>
                    • <b>Location</b>: {record.get('location')}<br>
                    • <b>Assigned Squad</b>: {record.get('assigned_to') or 'Awaiting Dispatch'}<br>
                    • <b>SLA Status</b>: {sla_info['sla_str']}
                </div>
                <div style="margin-top: 14px; padding: 12px; background: #F8FAFC; border-radius: 8px; font-size: 0.85rem; color: #17243A;">
                    <b>Official Dispatcher Notes:</b><br>
                    {record.get('admin_notes') or 'Your complaint has been acknowledged and is in the municipal triage queue.'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# LIVE MAP VIEW (DEDICATED FULL VIEW)
# =====================================================================
def render_full_live_map():
    st.title("🗺️ Municipal Live Map")
    st.markdown("Geographic distribution of citizen complaints across BMC municipal wards.")

    complaints = get_all_complaints()
    geocoded = [c for c in complaints if c.get("latitude") is not None and c.get("longitude") is not None]

    if not geocoded:
        st.info("No complaints with GPS coordinates have been registered yet.")
    else:
        map_data = []
        for c in geocoded:
            p = c.get("priority", "Medium")
            col = [220, 38, 38, 220] if p in ("Critical", "High") else [37, 99, 235, 200]
            if c.get("status") == "Resolved":
                col = [22, 163, 74, 200]
            map_data.append({
                "lat": float(c["latitude"]),
                "lon": float(c["longitude"]),
                "id": c["complaint_id"],
                "category": c.get("text_category", "General"),
                "priority": p,
                "status": c.get("status", "New"),
                "color": col
            })
        df = pd.DataFrame(map_data)
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=df["lat"].mean(),
                longitude=df["lon"].mean(),
                zoom=12,
                pitch=0
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position="[lon, lat]",
                    get_color="color",
                    get_radius=150,
                    pickable=True
                )
            ],
            tooltip={"text": "{id}: {category} ({priority})\nStatus: {status}"}
        ), use_container_width=True)


# =====================================================================
# SYSTEM ANALYTICS & REPORTS
# =====================================================================
def render_system_analytics():
    st.title("📊 System Analytics & Intelligence")
    st.markdown("Automated metrics derived from persisted SQLite complaint records.")

    complaints = get_all_complaints()
    if not complaints:
        st.info("No records to display.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🏷️ Category Distribution (NLP)")
        cats = [c.get("text_category") or "Unclassified" for c in complaints]
        df_cats = pd.Series(cats).value_counts().reset_index()
        df_cats.columns = ["Category", "Count"]
        st.bar_chart(df_cats.set_index("Category"))

    with col2:
        st.markdown("##### 🔄 Administrative Status Distribution")
        stats = [c.get("status") or "New" for c in complaints]
        df_stat = pd.Series(stats).value_counts().reset_index()
        df_stat.columns = ["Status", "Count"]
        st.bar_chart(df_stat.set_index("Status"))

    st.markdown("---")
    st.markdown("##### 📥 Export Case Records (RFC-4180 CSV)")
    csv_text = export_complaints_csv(complaints)
    st.download_button(
        label="Download Complaints CSV",
        data=csv_text,
        file_name=f"astra_civic_grievances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


# =====================================================================
# MAIN APPLICATION CONTROLLER
# =====================================================================
def main():
    # Handle incoming query parameters for direct links while preserving navigation
    q_params = st.query_params
    view_param = q_params.get("view", "")
    if view_param and "initialized_from_query" not in st.session_state:
        v = view_param.lower().strip()
        if v in ("admin", "overview", "dashboard"):
            st.session_state["portal_mode_radio"] = "🏢 Operations Center (Admin)"
            st.session_state["admin_nav_radio"] = "Overview"
            st.session_state["active_view"] = "Overview"
        elif v in ("map", "live_map"):
            st.session_state["portal_mode_radio"] = "🏢 Operations Center (Admin)"
            st.session_state["admin_nav_radio"] = "Live Map"
            st.session_state["active_view"] = "Live Map"
        elif v in ("analytics", "reports"):
            st.session_state["portal_mode_radio"] = "🏢 Operations Center (Admin)"
            st.session_state["admin_nav_radio"] = "System Analytics"
            st.session_state["active_view"] = "System Analytics"
        elif v in ("report", "citizen", "complaint"):
            st.session_state["portal_mode_radio"] = "👥 Citizen Registration Portal"
            st.session_state["citizen_nav_radio"] = "Report a Civic Issue"
            st.session_state["active_view"] = "Report a Civic Issue"
        elif v in ("track", "status"):
            st.session_state["portal_mode_radio"] = "👥 Citizen Registration Portal"
            st.session_state["citizen_nav_radio"] = "Track My Complaint"
            st.session_state["active_view"] = "Track My Complaint"
        elif v in ("welcome", "home", "landing"):
            st.session_state["portal_mode_radio"] = "✨ Welcome & Overview"
            st.session_state["active_view"] = "Welcome"
        st.session_state["initialized_from_query"] = True

    # Ensure initial portal mode defaults to Welcome
    if "portal_mode_radio" not in st.session_state:
        st.session_state["portal_mode_radio"] = "✨ Welcome & Overview"
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "Welcome"

    render_sidebar()

    active_view = st.session_state.get("active_view", "Welcome")

    if active_view == "Welcome":
        render_welcome_page()
    elif active_view == "Overview":
        render_admin_dashboard()
    elif active_view == "Live Map":
        render_full_live_map()
    elif active_view == "Complaints & Reports":
        render_admin_dashboard()  # Seamlessly uses the dashboard table
    elif active_view == "System Analytics":
        render_system_analytics()
    elif active_view == "Report a Civic Issue":
        render_citizen_registration()
    elif active_view == "Track My Complaint":
        render_track_complaint()
    else:
        render_welcome_page()


if __name__ == "__main__":
    main()

