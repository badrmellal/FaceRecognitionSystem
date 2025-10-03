#!/usr/bin/env python3
"""
FIXED API Backend for Face Recognition System
WebSocket streaming compatible with system
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import time
import cv2
import numpy as np

# Import system modules
from database import DatabaseManager
from config import Config

try:
    from face_recognition_system import FaceRecognitionSystem

    SYSTEM_AVAILABLE = True
except ImportError:
    FaceRecognitionSystem = None
    SYSTEM_AVAILABLE = False

app = FastAPI(
    title=" Face Recognition API",
    description="API for Face Recognition with Adaptive Thresholds",
    version="6.0.0-BADR"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
recognition_system: Optional[FaceRecognitionSystem] = None
monitoring_active = False


# Pydantic models
class PersonInfo(BaseModel):
    person_id: str
    num_encodings: int
    metadata: Dict[str, Any]
    last_seen: Optional[str] = None


class AccessEvent(BaseModel):
    timestamp: str
    event_type: str = "UNKNOWN"
    person_id: str = "UNKNOWN"
    person_name_french: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    visit_duration_seconds: Optional[float] = None
    track_id: Optional[str] = None
    location: str = "entrance_gate"
    entrance_zone: Optional[str] = None
    entrance_direction: Optional[str] = None
    distance_type: Optional[str] = None
    entry_time: Optional[str] = None
    message_french: Optional[str] = None


class SystemStats(BaseModel):
    frames_captured: int
    frames_processed: int
    faces_detected: int
    faces_recognized: int
    authorized_detections: int
    unauthorized_detections: int
    alerts_triggered: int
    detection_fps: float
    recognition_fps: float
    active_tracks: int
    system_uptime: float
    evidence_saved: int = 0
    temporal_decisions: int = 0
    adaptive_gap_decisions: int = 0
    high_confidence_overrides: int = 0


class MonitoringConfig(BaseModel):
    rtsp_url: str
    duration: Optional[int] = None
    config_file: str = "config.json"


class DateRangeQuery(BaseModel):
    start_date: str
    end_date: str
    person_id: Optional[str] = None


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """FIXED WebSocket streaming compatible """
    await websocket.accept()

    # Wait for system to be ready
    timeout_count = 0
    while timeout_count < 100:  # 10 second timeout
        if monitoring_active and recognition_system is not None:
            break
        await asyncio.sleep(0.1)
        timeout_count += 1
    else:
        await websocket.close()
        return

    logger.info("🔴 WebSocket client connected - BADR VERSION")

    try:
        while monitoring_active and recognition_system:
            try:
                # Get current frame safely
                current_frame = None
                with recognition_system.frame_lock:
                    if recognition_system.current_frame is not None:
                        current_frame = recognition_system.current_frame.copy()

                if current_frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Get active detections from enterprise system
                detections = []
                try:
                    detections = list(recognition_system.active_tracks.values())
                except Exception as e:
                    logger.debug(f"Error getting detections: {e}")

                # Create display frame
                display_frame = current_frame.copy()

                # FIXED: Use correct method names from Enterprise system
                try:
                    if hasattr(recognition_system, '_draw_enterprise_detections'):
                        recognition_system._draw_enterprise_detections(display_frame, detections)
                    elif hasattr(recognition_system, '_draw_enhanced_detections'):
                        recognition_system._draw_enhanced_detections(display_frame, detections)
                    else:
                        # Fallback: draw detections manually
                        _draw_detections_fallback(display_frame, detections)

                    if hasattr(recognition_system, '_draw_enterprise_interface'):
                        recognition_system._draw_enterprise_interface(display_frame)
                    elif hasattr(recognition_system, '_draw_enhanced_interface'):
                        recognition_system._draw_enhanced_interface(display_frame)
                    else:
                        # Fallback: draw basic interface
                        _draw_interface_fallback(display_frame, recognition_system)

                except Exception as e:
                    logger.warning(f"Error drawing on frame: {e}")
                    # Use fallback drawing
                    _draw_detections_fallback(display_frame, detections)
                    _draw_interface_fallback(display_frame, recognition_system)

                # Encode and send frame
                ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    await websocket.send_bytes(buffer.tobytes())

                # Control frame rate (~30 FPS)
                await asyncio.sleep(0.033)

            except Exception as e:
                logger.error(f"WebSocket frame processing error: {e}")
                break

    except WebSocketDisconnect:
        logger.info("🔴 WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


def _draw_detections_fallback(frame: np.ndarray, detections: List):
    """Fallback detection drawing if enterprise methods not available"""
    try:
        for detection in detections:
            # Extract bbox safely
            bbox = getattr(detection, 'bbox', None)
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            # Determine color based on status
            status = getattr(detection, 'status', 'unknown')
            if status == "authorized":
                color = (0, 255, 0)  # Green
                status_text = f"✅ AUTHORIZED"
                if hasattr(detection, 'person_id') and detection.person_id:
                    status_text = f"✅ {detection.person_id.replace('_', ' ').upper()}"
            elif status == "SECURITY_ALERT":
                color = (0, 0, 255)  # Red
                status_text = f"🚨 SECURITY ALERT"
            elif status == "analyzing":
                color = (0, 255, 255)  # Yellow
                status_text = f"🔍 ANALYZING"
            else:
                color = (255, 100, 0)  # Blue
                status_text = f"👤 DETECTING"

            # Draw detection box
            thickness = 6 if status in ["authorized", "SECURITY_ALERT"] else 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw status label
            label_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(frame, status_text, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw additional info
            track_id = getattr(detection, 'track_id', 'unknown')
            confidence = getattr(detection, 'recognition_confidence', 0.0)

            info_text = f"Track: {track_id}"
            if confidence > 0:
                info_text += f" | Conf: {confidence:.2f}"

            cv2.putText(frame, info_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    except Exception as e:
        logger.error(f"Fallback detection drawing error: {e}")


def _draw_interface_fallback(frame: np.ndarray, recognition_system):
    """Fallback interface drawing"""
    try:
        h, w = frame.shape[:2]

        # Header
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.putText(frame, "🏢 ENTERPRISE SURVEILLANCE ACTIVE", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Status
        active_tracks = len(getattr(recognition_system, 'active_tracks', {}))
        status_text = f"Active Tracks: {active_tracks} | Status: OPERATIONAL"
        cv2.putText(frame, status_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 120, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    except Exception as e:
        logger.error(f"Fallback interface drawing error: {e}")


@app.get("/api/health")
async def health_check():
    """Health check with enterprise info"""
    system_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "monitoring_active": monitoring_active,
        "version": "6.0.0-ENTERPRISE",
        "system_available": SYSTEM_AVAILABLE,
        "features": {
            "enterprise_recognition": True,
            "adaptive_thresholds": True,
            "temporal_voting": True,
            "quality_weighting": True,
            "high_confidence_override": True
        }
    }

    if recognition_system:
        try:
            system_info["active_tracks"] = len(getattr(recognition_system, 'active_tracks', {}))
            system_info["known_faces"] = len(getattr(recognition_system, 'known_faces', {}))

            # Enterprise recognition engine info
            if hasattr(recognition_system, 'recognition_engine'):
                engine = recognition_system.recognition_engine
                system_info["recognition_engine"] = {
                    "base_threshold": getattr(engine, 'base_threshold', 0.72),
                    "high_confidence_override": getattr(engine, 'high_confidence_override', 0.90),
                    "temporal_window": getattr(engine, 'temporal_window', 8.0),
                    "adaptive_gaps": True
                }
        except Exception as e:
            logger.debug(f"Error getting system info: {e}")

    return system_info


@app.get("/api/system/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system stats with enterprise metrics"""
    try:
        if recognition_system:
            stats = recognition_system.stats.copy()
            stats['active_tracks'] = len(getattr(recognition_system, 'active_tracks', {}))
            stats['system_uptime'] = time.time() - getattr(recognition_system, 'start_time', time.time())

            # Ensure all required fields exist
            stats.setdefault('evidence_saved', 0)
            stats.setdefault('temporal_decisions', 0)
            stats.setdefault('adaptive_gap_decisions', 0)
            stats.setdefault('high_confidence_overrides', 0)
        else:
            stats = {
                'frames_captured': 0, 'frames_processed': 0,
                'faces_detected': 0, 'faces_recognized': 0,
                'authorized_detections': 0, 'unauthorized_detections': 0,
                'alerts_triggered': 0, 'detection_fps': 0.0,
                'recognition_fps': 0.0, 'active_tracks': 0,
                'system_uptime': 0.0, 'evidence_saved': 0,
                'temporal_decisions': 0, 'adaptive_gap_decisions': 0,
                'high_confidence_overrides': 0
            }

        return SystemStats(**stats)

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return SystemStats(
            frames_captured=0, frames_processed=0, faces_detected=0,
            faces_recognized=0, authorized_detections=0, unauthorized_detections=0,
            alerts_triggered=0, detection_fps=0.0, recognition_fps=0.0,
            active_tracks=0, system_uptime=0.0, evidence_saved=0,
            temporal_decisions=0, adaptive_gap_decisions=0, high_confidence_overrides=0
        )


@app.get("/api/database/persons", response_model=List[PersonInfo])
async def get_known_persons():
    """Get known persons"""
    try:
        db = DatabaseManager()
        persons = []
        for pid in db.get_known_persons():
            info = db.get_person_info(pid)
            if info:
                persons.append(PersonInfo(
                    person_id=pid,
                    num_encodings=info['num_encodings'],
                    metadata=info['metadata']
                ))
        return persons
    except Exception as e:
        logger.error(f"Error getting persons: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/database/persons/{person_id}")
async def add_person(person_id: str, request: Dict[str, Any]):
    """Add person to database"""
    try:
        db = DatabaseManager()
        image_paths = request.get('image_paths', [])
        metadata = request.get('metadata', {})

        if db.add_person(person_id, image_paths, metadata):
            # Reload face database in recognition system
            if recognition_system and hasattr(recognition_system, '_load_face_database'):
                try:
                    recognition_system._load_face_database()
                except Exception as e:
                    logger.warning(f"Could not reload face database: {e}")
            return {"message": f"Person {person_id} added successfully"}
        raise HTTPException(status_code=400, detail="Failed to add person")
    except Exception as e:
        logger.error(f"Error adding person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/database/persons/{person_id}")
async def remove_person(person_id: str):
    """Remove person from database"""
    try:
        db = DatabaseManager()
        if db.remove_person(person_id):
            # Reload face database in recognition system
            if recognition_system and hasattr(recognition_system, '_load_face_database'):
                try:
                    recognition_system._load_face_database()
                except Exception as e:
                    logger.warning(f"Could not reload face database: {e}")
            return {"message": f"Person {person_id} removed successfully"}
        raise HTTPException(status_code=404, detail="Person not found")
    except Exception as e:
        logger.error(f"Error removing person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/access")
async def get_access_logs(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        person_id: Optional[str] = None,
        limit: int = 100
):
    """Get access logs with support for entry/exit events"""
    try:
        log_file = Path("logs/logging_access.log")
        if not log_file.exists():
            return []

        logs = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())

                    # Apply filters
                    if start_date and event.get('timestamp', '') < start_date:
                        continue
                    if end_date and event.get('timestamp', '') > end_date:
                        continue
                    if person_id and event.get('person_id') != person_id:
                        continue

                    # Ensure required fields and handle new event types
                    event.setdefault('event_type', 'UNKNOWN')
                    event.setdefault('person_id', 'UNKNOWN')

                    # Add French name if not present
                    if not event.get('person_name_french') and event.get('person_id') != 'UNKNOWN':
                        event['person_name_french'] = event['person_id'].replace('_', ' ').title()

                    # Add French message if not present
                    if not event.get('message_french'):
                        if event['event_type'] == 'AUTHORIZED_ENTRY':
                            event['message_french'] = f"{event.get('person_name_french', 'Personnel')} est entré(e)"
                        elif event['event_type'] == 'AUTHORIZED_EXIT':
                            event['message_french'] = f"{event.get('person_name_french', 'Personnel')} est sorti(e)"
                        elif event['event_type'] == 'AUTHORIZED_ACCESS':
                            event[
                                'message_french'] = f"Accès autorisé pour {event.get('person_name_french', 'Personnel')}"
                        elif event['event_type'] == 'SECURITY_ALERT':
                            event['message_french'] = "Alerte sécurité - Personne non autorisée"

                    logs.append(AccessEvent(**event))

                    if len(logs) >= limit:
                        break

                except (json.JSONDecodeError, ValidationError) as e:
                    logger.debug(f"Skipping invalid log entry: {e}")
                    continue

        return list(reversed(logs))

    except Exception as e:
        logger.error(f"Error getting access logs: {e}")
        return []


@app.get("/api/logs/today")
async def get_today_logs():
    """Get today's logs"""
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    return await get_access_logs(start_date=today, end_date=tomorrow)


@app.post("/api/export/excel")
async def export_to_excel(query: DateRangeQuery):
    """Export logs to Excel"""
    try:
        logs = await get_access_logs(
            start_date=query.start_date,
            end_date=query.end_date,
            person_id=query.person_id,
            limit=10000
        )

        df = pd.DataFrame([log.dict() for log in logs]) if logs else pd.DataFrame()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enterprise_access_report_{timestamp}.xlsx"
        filepath = Path("exports") / filename
        filepath.parent.mkdir(exist_ok=True)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Access Logs', index=False)

            # Auto-adjust column widths
            if not df.empty:
                worksheet = writer.sheets['Access Logs']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        return FileResponse(
            str(filepath),
            filename=filename,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary with entry/exit statistics"""
    try:
        today_logs = await get_today_logs()
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        today_str = datetime.now().strftime("%Y-%m-%d")
        week_logs = await get_access_logs(start_date=week_ago, end_date=today_str, limit=10000)

        # Count different event types
        person_activity = {}
        entries_today = 0
        exits_today = 0

        for log in today_logs:
            if log.event_type == "AUTHORIZED_ENTRY" and log.person_id != "UNKNOWN":
                person_activity[log.person_id] = person_activity.get(log.person_id, 0) + 1
                entries_today += 1
            elif log.event_type == "AUTHORIZED_EXIT":
                exits_today += 1

        recent_alerts = [log for log in today_logs if log.event_type == "SECURITY_ALERT"][-5:]
        recent_entries = [log for log in today_logs if log.event_type == "AUTHORIZED_ENTRY"][-10:]

        return {
            "today": {
                "total_events": len(today_logs),
                "authorized_access": len(
                    [l for l in today_logs if l.event_type in ["AUTHORIZED_ACCESS", "AUTHORIZED_ENTRY"]]),
                "authorized_entries": entries_today,
                "authorized_exits": exits_today,
                "security_alerts": len([l for l in today_logs if l.event_type == "SECURITY_ALERT"]),
                "unique_persons": len({l.person_id for l in today_logs
                                       if l.event_type in ["AUTHORIZED_ACCESS",
                                                           "AUTHORIZED_ENTRY"] and l.person_id != "UNKNOWN"})
            },
            "week": {
                "total_events": len(week_logs),
                "daily_average": len(week_logs) / 7 if week_logs else 0
            },
            "person_activity": person_activity,
            "recent_alerts": recent_alerts,
            "recent_entries": recent_entries,
            "monitoring_status": monitoring_active
        }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return {
            "today": {"total_events": 0, "authorized_access": 0, "authorized_entries": 0,
                      "authorized_exits": 0, "security_alerts": 0, "unique_persons": 0},
            "week": {"total_events": 0, "daily_average": 0},
            "person_activity": {},
            "recent_alerts": [],
            "recent_entries": [],
            "monitoring_status": monitoring_active,
            "error": str(e)
        }


@app.get("/api/logs/entries/today")
async def get_today_entries():
    """Get today's entry logs specifically"""
    try:
        today_logs = await get_today_logs()
        entries = [log for log in today_logs if log.event_type == "AUTHORIZED_ENTRY"]

        # Sort by timestamp (most recent first)
        entries.sort(key=lambda x: x.timestamp, reverse=True)

        return entries
    except Exception as e:
        logger.error(f"Error getting today's entries: {e}")
        return []



@app.get("/api/logs/visits/today")
async def get_today_visits():
    """Get entry/exit pairs for today (complete visits)"""
    try:
        today_logs = await get_today_logs()

        # Group by person and track_id to match entries with exits
        visits = []
        entries = {}  # person_id -> entry_log

        for log in sorted(today_logs, key=lambda x: x.timestamp):
            if log.event_type == "AUTHORIZED_ENTRY":
                entries[log.person_id] = log
            elif log.event_type == "AUTHORIZED_EXIT" and log.person_id in entries:
                entry_log = entries[log.person_id]
                visit = {
                    "person_id": log.person_id,
                    "person_name_french": log.person_name_french,
                    "entry_time": entry_log.timestamp,
                    "exit_time": log.timestamp,
                    "duration_seconds": log.visit_duration_seconds,
                    "entrance_zone": entry_log.entrance_zone,
                    "track_id": log.track_id
                }
                visits.append(visit)
                del entries[log.person_id]  # Remove completed visit

        # Add ongoing visits (entries without exits)
        for person_id, entry_log in entries.items():
            visit = {
                "person_id": person_id,
                "person_name_french": entry_log.person_name_french,
                "entry_time": entry_log.timestamp,
                "exit_time": None,
                "duration_seconds": None,
                "entrance_zone": entry_log.entrance_zone,
                "track_id": entry_log.track_id,
                "status": "en_cours"  # ongoing
            }
            visits.append(visit)

        return visits

    except Exception as e:
        logger.error(f"Error getting today's visits: {e}")
        return []


@app.post("/api/monitoring/start")
async def start_monitoring(config: MonitoringConfig, background_tasks: BackgroundTasks):
    """Start enterprise monitoring"""
    global recognition_system, monitoring_active

    if not SYSTEM_AVAILABLE:
        raise HTTPException(status_code=500, detail="Enterprise face recognition system not available")

    if monitoring_active:
        raise HTTPException(status_code=400, detail="Monitoring already active")

    background_tasks.add_task(run_monitoring, config)
    return {
        "message": "Enterprise monitoring started",
        "rtsp_url": config.rtsp_url,
        "version": "6.0.0-ENTERPRISE",
        "features": {
            "adaptive_thresholds": True,
            "temporal_voting": True,
            "quality_weighting": True,
            "high_confidence_override": True
        }
    }


@app.post("/api/monitoring/stop")
async def stop_monitoring():
    """Stop monitoring"""
    global recognition_system, monitoring_active
    if not monitoring_active:
        raise HTTPException(status_code=400, detail="Monitoring not active")

    if recognition_system:
        try:
            recognition_system._stop_monitoring()
        except Exception as e:
            logger.warning(f"Error stopping monitoring: {e}")
        recognition_system = None
    monitoring_active = False
    return {"message": "Enterprise monitoring stopped"}


@app.get("/api/monitoring/status")
async def get_monitoring_status():
    """Get monitoring status"""
    status_info = {
        "active": monitoring_active,
        "system_initialized": recognition_system is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "6.0.0-ENTERPRISE"
    }

    if recognition_system:
        try:
            status_info.update({
                "active_tracks": len(getattr(recognition_system, 'active_tracks', {})),
                "known_faces": len(getattr(recognition_system, 'known_faces', {})),
                "features": {
                    "enterprise_recognition": True,
                    "adaptive_thresholds": True,
                    "temporal_voting": True,
                    "quality_weighting": True
                }
            })
        except Exception as e:
            logger.debug(f"Error getting detailed status: {e}")

    return status_info


def run_monitoring(config: MonitoringConfig):
    """Run enterprise monitoring in background"""
    global recognition_system, monitoring_active
    try:
        monitoring_active = True
        recognition_system = FaceRecognitionSystem(config.config_file)
        recognition_system.start_time = time.time()

        logger.info(f"🏢 Starting Enterprise surveillance system...")
        recognition_system.start_monitoring(config.rtsp_url, config.duration)

    except Exception as e:
        logger.error(f"Enterprise monitoring error: {e}")
    finally:
        monitoring_active = False
        recognition_system = None


@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    try:
        cfg = Config()
        return {
            "camera": cfg.get_optimized_camera_config(),
            "display": cfg.get_display_config(),
            "alerts": cfg.get_alert_config(),
            "face_recognition": {
                "model": cfg.get("face_recognition.model"),
                "similarity_threshold": cfg.get("face_recognition.similarity_threshold"),
                "target_size": cfg.get("face_recognition.target_size")
            },
            "enterprise": {
                "adaptive_thresholds": True,
                "temporal_voting": True,
                "quality_weighting": True,
                "high_confidence_override": True
            }
        }
    except Exception as e:
        logger.error(f"Config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/config")
async def update_config(config_updates: Dict[str, Any]):
    """Update system configuration"""
    try:
        cfg = Config()
        for section, values in config_updates.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    cfg.set(f"{section}.{key}", value)
            else:
                cfg.set(section, values)

        return {"message": "Enterprise configuration updated successfully"}
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug/enterprise")
async def get_enterprise_debug():
    """Get enterprise recognition debug info"""
    if not recognition_system:
        raise HTTPException(status_code=400, detail="System not running")

    try:
        debug_info = {
            "system_type": "Enterprise Face Recognition",
            "version": "6.0.0-ENTERPRISE",
            "active_tracks": len(getattr(recognition_system, 'active_tracks', {})),
            "known_faces": len(getattr(recognition_system, 'known_faces', {})),
            "features": {
                "adaptive_thresholds": True,
                "temporal_voting": True,
                "quality_weighting": True,
                "high_confidence_override": True
            }
        }

        # Enterprise engine info
        if hasattr(recognition_system, 'recognition_engine'):
            engine = recognition_system.recognition_engine
            debug_info["recognition_engine"] = {
                "base_threshold": getattr(engine, 'base_threshold', 0.72),
                "high_confidence_override": getattr(engine, 'high_confidence_override', 0.90),
                "temporal_window": getattr(engine, 'temporal_window', 8.0),
                "gap_requirements": {
                    "very_high_conf": getattr(engine, 'gap_very_high_conf', 0.015),
                    "high_conf": getattr(engine, 'gap_high_conf', 0.025),
                    "medium_conf": getattr(engine, 'gap_medium_conf', 0.035),
                    "low_conf": getattr(engine, 'gap_low_conf', 0.05)
                }
            }

        # Current detections
        current_detections = []
        for track_id, detection in getattr(recognition_system, 'active_tracks', {}).items():
            det_info = {
                "track_id": track_id,
                "status": getattr(detection, 'status', 'unknown'),
                "person_id": getattr(detection, 'person_id', None),
                "confidence": getattr(detection, 'recognition_confidence', 0.0),
                "positive_frames": getattr(detection, 'positive_frames', 0),
                "negative_frames": getattr(detection, 'negative_frames', 0),
                "duration": getattr(detection, 'duration', 0.0)
            }
            current_detections.append(det_info)

        debug_info["current_detections"] = current_detections

        return debug_info

    except Exception as e:
        logger.error(f"Enterprise debug error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create necessary directories
    Path("exports").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/security_evidence").mkdir(exist_ok=True)
    Path("logs/screenshots").mkdir(exist_ok=True)

    print("🏢 Starting Enterprise Face Recognition API Server")
    print("✅ Adaptive thresholds: ACTIVE")
    print("✅ Temporal voting: ACTIVE")
    print("✅ Quality weighting: ACTIVE")
    print("✅ High confidence override: ACTIVE")
    print("🔧 WebSocket streaming: COMPATIBLE")
    print("📦 Detection boxes: VISIBLE")
    print("🌐 API docs: http://localhost:8000/docs")
    print("🔍 Enterprise debug: http://localhost:8000/api/debug/enterprise")
    print("📊 Health check: http://localhost:8000/api/health")

    if not SYSTEM_AVAILABLE:
        print("⚠️  WARNING: Enterprise face recognition system not available")
        print("   Make sure face_recognition_system.py is in the same directory")
    else:
        print("✅ Enterprise face recognition system: AVAILABLE")

    print("\n🎯 Enterprise recognition with boxes in stream!")
    print("🔗 Frontend connection: ws://localhost:8000/ws/stream")

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")