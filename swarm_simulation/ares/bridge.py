"""Bridge layer: converts canonical ares SwarmState into the frontend schema
expected by the Next.js Mission Command Center (types/swarm.ts).
"""
from __future__ import annotations

import math
from typing import Optional

from ares.sim import Simulator
from ares.state import SwarmState, Role, Health, PoIStatus, dist


def map_role(role: Role) -> str:
    if role == Role.IDLE:
        return "IDLE"
    if role == Role.SURVEYOR:
        return "SURVEYOR"
    if role == Role.RELAY:
        return "RELAY"
    if role == Role.BACKUP_RELAY:
        return "BACKUP_RELAY"
    if role == Role.EMERGENCY_RESPONDER:
        return "EMERGENCY"
    if role in (Role.RETURN_TO_BASE, Role.CHARGING):
        return "RTB"
    if role == Role.FAILED:
        return "FAILED"
    # fallback for hybrid or recovering
    return "RELAY"


def map_network_state(net_state_value: str) -> str:
    if net_state_value == "DISCONNECTED":
        return "CRITICAL"
    if net_state_value in ("HEALTHY", "DEGRADED", "CRITICAL", "RECOVERING"):
        return net_state_value
    return "HEALTHY"


def compute_uav_pdr(uav_id: str, state: SwarmState) -> float:
    if uav_id not in state.connected_to_gcs:
        return 0.0
    uav = state.uavs.get(uav_id)
    if not uav or not uav.connectivity_path:
        return 0.0
    
    path = uav.connectivity_path
    pdr_product = 1.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        k = tuple(sorted((a, b)))
        if k in state.links:
            pdr_product *= state.links[k].pdr
        else:
            pdr_product *= 0.8
    return max(0.0, min(1.0, pdr_product))


def to_frontend_state(
    sim: Simulator,
    scenario_name: str = "baseline",
    session_id: str = "live_session",
    is_paused: bool = False,
) -> dict:
    state = sim.state
    sc = sim.scenario

    # 1. GCS info
    gcs_pos = list(state.gcs_position)
    if len(gcs_pos) == 2:
        gcs_pos.append(15.0)

    # 2. UAVs dictionary
    uavs_dict = {}
    for uid, uav in state.uavs.items():
        pos = [round(x, 1) for x in uav.position]
        vel = [round(x, 1) for x in uav.velocity]
        target = [round(x, 1) for x in uav.target] if uav.target else list(pos)
        
        # Risk assessment
        if uav.health == Health.FAILED:
            risk = "HIGH"
            health_score = 0.0
        elif uav.battery < 20.0 or uav.health == Health.DEGRADED:
            risk = "HIGH"
            health_score = 0.5
        elif uav.battery < 45.0:
            risk = "MEDIUM"
            health_score = 0.8
        else:
            risk = "LOW"
            health_score = 1.0

        pdr = compute_uav_pdr(uid, state)
        path = ["GCS"] + (uav.connectivity_path if uav.connectivity_path else [uid])

        uavs_dict[uid] = {
            "id": uid,
            "position": pos,
            "velocity": vel,
            "target_position": target,
            "battery": round(max(0.0, min(100.0, uav.battery)), 1),
            "role": map_role(uav.role),
            "health": health_score,
            "risk": risk,
            "current_task": uav.current_task,
            "path_to_gcs": path,
            "end_to_end_pdr": round(pdr, 3),
        }
        # adaptive-controller latent, only present with the dynamic flight model
        flight = getattr(sim, "flight", None)
        if flight is not None and uid in flight.latent:
            uavs_dict[uid]["latent_z"] = [round(x, 3) for x in flight.latent[uid]]
        if flight is not None and uid in flight.param_estimate:
            uavs_dict[uid]["param_estimate"] = flight.param_estimate[uid]

    # 3. PoIs dictionary
    pois_dict = {}
    for pid, poi in state.pois.items():
        if poi.status == PoIStatus.COMPLETE:
            status = "COMPLETED"
        elif poi.status in (PoIStatus.ASSIGNED, PoIStatus.IN_PROGRESS) or poi.assigned_uav:
            status = "IN_PROGRESS"
        else:
            status = "PENDING"

        pos = [round(x, 1) for x in poi.position]
        pois_dict[pid] = {
            "id": pid,
            "name": f"PoI {pid} (P={poi.priority:.1f})",
            "position": pos,
            "priority": round(poi.priority, 2),
            "survey_duration_s": 15,
            "status": status,
            "assigned_uav": poi.assigned_uav,
        }

    # 4. Links list
    links_list = []
    for (a, b), lk in state.links.items():
        dist_val = max(1.0, lk.distance)
        # Standard free-space / log-distance path loss approximation for RSSI display
        rssi = round(-40.0 - 20.0 * math.log10(dist_val), 1)
        rssi = max(-95.0, min(-30.0, rssi))

        links_list.append({
            "uav_a": a,
            "uav_b": b,
            "distance_m": round(lk.distance, 1),
            "rssi_dbm": rssi,
            "pdr": round(lk.pdr, 3),
            "latency_ms": round(lk.latency_ms, 1),
            "los": True,
            "active": True,
        })

    # 5. Challenge Metrics
    all_pdrs = [lk.pdr for lk in state.links.values()]
    mean_pdr = (sum(all_pdrs) / len(all_pdrs) * 100.0) if all_pdrs else 100.0

    all_lats = [lk.latency_ms for lk in state.links.values()]
    mean_lat = (sum(all_lats) / len(all_lats)) if all_lats else 20.0

    total_pois = len(state.pois)
    completed_pois = sum(1 for p in state.pois.values() if p.status == PoIStatus.COMPLETE)
    mission_completion_rate = (completed_pois / total_pois) if total_pois > 0 else 1.0

    # Weight scores
    sum_pri_all = sum(p.priority for p in state.pois.values())
    sum_pri_done = sum(p.priority for p in state.pois.values() if p.status == PoIStatus.COMPLETE)
    pri_score = (sum_pri_done / sum_pri_all) if sum_pri_all > 0 else 1.0

    # MTTR & recovery metrics
    tracker = sim.recovery_planner.tracker
    recovery_times = tracker.recovery_times
    mttr = (sum(recovery_times) / len(recovery_times)) if recovery_times else 0.0
    recoveries_count = tracker.recoveries_completed
    total_downtime = tracker.total_downtime

    # Overall score estimator
    mission_subscore = pri_score * 100.0
    comm_subscore = mean_pdr
    role_subscore = max(50.0, 100.0 - sim.state.role_changes * 2.0)
    recovery_subscore = 100.0 if mttr == 0 else max(40.0, 100.0 - mttr * 2.0)
    safety_subscore = 100.0 if (state.collisions == 0 and state.geofence_violations == 0) else 50.0
    innovation_subscore = 92.0

    overall = (
        0.25 * mission_subscore
        + 0.25 * comm_subscore
        + 0.20 * role_subscore
        + 0.15 * recovery_subscore
        + 0.10 * safety_subscore
        + 0.05 * innovation_subscore
    )

    metrics_obj = {
        "session_id": session_id,
        "mission_time_s": round(state.time, 1),
        "mission_completion_rate": round(mission_completion_rate * 100.0, 1),
        "priority_weighted_score": round(pri_score * 100.0, 1),
        "mean_pdr": round(mean_pdr, 1),
        "mean_latency_ms": round(mean_lat, 1),
        "connectivity_availability": round((len(state.connected_to_gcs) / max(1, len(state.alive_uavs()))) * 100.0, 1),
        "total_downtime_s": round(total_downtime, 1),
        "relay_reallocations": state.relay_reallocations,
        "recoveries_count": recoveries_count,
        "mttr_s": round(mttr, 2),
        "collision_count": state.collisions,
        "min_separation_m": round(state.min_separation_observed if state.min_separation_observed != float("inf") else 20.0, 1),
        "overall_score": round(overall, 1),

        "subscores": {
            "mission": round(mission_subscore, 1),
            "communication": round(comm_subscore, 1),
            "role_management": round(role_subscore, 1),
            "fault_recovery": round(recovery_subscore, 1),
            "safety": round(safety_subscore, 1),
            "innovation": round(innovation_subscore, 1),
        },
    }

    # 6. Events list (most recent 40)
    events_list = []
    category_map = {
        "UAV_FAILURE_DETECTED": ("FAULT", "danger"),
        "LINK_DEGRADED": ("COMM_DEGRADE", "warning"),
        "PRIORITY_ESCALATED": ("PRIORITY", "warning"),
        "RECOVERY_STARTED": ("RECOVERY", "warning"),
        "RECOVERY_COMPLETED": ("RESTORED", "success"),
        "RELAY_ASSIGNMENT": ("MISSION", "info"),
        "TASK_ASSIGNMENT": ("MISSION", "info"),
        "RETURN_TO_BASE": ("SAFETY", "warning"),
        "SEPARATION_VIOLATION": ("SAFETY", "danger"),
        "GEOFENCE_VIOLATION": ("SAFETY", "danger"),
    }

    for ev in sim.log.entries[-40:]:
        cat, sev = category_map.get(ev.event_type, ("MISSION", "info"))
        t_val = getattr(ev, "timestamp", getattr(ev, "time", state.time))
        events_list.append({
            "timestamp": round(t_val, 1),
            "category": cat,
            "title": f"{ev.event_type}: {ev.source}",
            "description": f"{ev.reason} (old: {ev.old_state} -> new: {ev.new_state})",
            "severity": sev,
        })


    if not events_list:
        events_list.append({
            "timestamp": round(state.time, 1),
            "category": "MISSION",
            "title": "ARES-X Swarm Online",
            "description": f"Swarm operational with {len(state.uavs)} UAVs under scenario '{scenario_name}'.",
            "severity": "info",
        })

    return {
        "timestamp": round(state.time, 1),
        "scenario": scenario_name,
        "global_network_state": map_network_state(state.global_network_state.value),
        "gcs": {
            "id": "GCS_ALPHA",
            "position": gcs_pos,
        },
        "uavs": uavs_dict,
        "pois": pois_dict,
        "links": links_list,
        "metrics": metrics_obj,
        "events": events_list,
        "is_paused": is_paused,
    }
