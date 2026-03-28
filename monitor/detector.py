import cv2
import numpy as np
from ultralytics import YOLO
import os
from django.conf import settings
import time

class VideoDetector:
    def __init__(self, video_source='webcam', video_path=None):
        self.video_source = video_source
        self.video_path = video_path
        self.cap = None
        self.model = YOLO('yolov8n.pt')
        self.polygon_points = []
        self.sub_zones = []
        self.final_polygon = None
        self.is_playing = True
        self.is_drawing = True
        self.total_frames = 0
        self.fps = 30
        self.current_frame_pos = 0
        self.VIDEO_WIDTH = 640
        self.VIDEO_HEIGHT = 480
        self.frame_cache = None
        self.total_count = 0
        self.light_states = []
        self.light_off_timers = []
        self.DEBOUNCE_DELAY = 3.0
        self.max_polygon_points = 4
        self.frame_read_errors = 0
        self.MAX_FRAME_ERRORS = 5
        
    def initialize_camera(self):
        """Initialize video capture based on source"""
        try:
            if self.cap:
                self.cap.release()
            
            if self.video_source == 'webcam':
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            elif self.video_source == 'upload' and self.video_path:
                self.cap = cv2.VideoCapture(self.video_path)
            elif self.video_source == 'select' and self.video_path:
                upload_dir = os.path.join(settings.BASE_DIR, 'uploads')
                full_path = os.path.join(upload_dir, self.video_path)
                self.cap = cv2.VideoCapture(full_path)
            else:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
                if self.total_frames > 0:
                    orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.VIDEO_HEIGHT = int(orig_height * (self.VIDEO_WIDTH / orig_width))
                
                self.frame_read_errors = 0
                return True
            else:
                print(f"ERROR: Could not open {self.video_source}")
                return False
        except Exception as e:
            print(f"ERROR initializing camera: {e}")
            return False
    
    def get_factor_pairs(self, n):
        """Get all row×col combinations for a number"""
        factors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                factors.append((i, n // i))
                if i != n // i:
                    factors.append((n // i, i))
        return sorted(factors)
    
    def interpolate_point(self, p1, p2, t):
        """Interpolate a point between p1 and p2 at ratio t (0.0 to 1.0)"""
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        return (x, y)
    
    def split_polygon_into_grid(self, polygon, rows, cols):
        """Split a polygon into smaller polygons with same shape"""
        sub_zones = []
        n_points = len(polygon)
        
        if n_points < 3:
            return sub_zones
        
        if n_points == 4:
            top_edge_points = []
            bottom_edge_points = []
            
            for col in range(cols + 1):
                t = col / cols
                top_edge_points.append(self.interpolate_point(polygon[0], polygon[1], t))
            
            for col in range(cols + 1):
                t = col / cols
                bottom_edge_points.append(self.interpolate_point(polygon[3], polygon[2], t))
            
            for row in range(rows):
                for col in range(cols):
                    t_row1 = row / rows
                    t_row2 = (row + 1) / rows
                    
                    p1 = self.interpolate_point(top_edge_points[col], bottom_edge_points[col], t_row1)
                    p2 = self.interpolate_point(top_edge_points[col + 1], bottom_edge_points[col + 1], t_row1)
                    p3 = self.interpolate_point(top_edge_points[col + 1], bottom_edge_points[col + 1], t_row2)
                    p4 = self.interpolate_point(top_edge_points[col], bottom_edge_points[col], t_row2)
                    
                    zone_pts = [p1, p2, p3, p4]
                    
                    sub_zones.append({
                        'polygon': zone_pts,
                        'row': row,
                        'col': col,
                        'count': 0,
                        'light_on': False,
                        'last_detection_time': 0
                    })
        else:
            pts = np.array(polygon, dtype=np.float32)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            
            cell_width = (x_max - x_min) / cols
            cell_height = (y_max - y_min) / rows
            
            for row in range(rows):
                for col in range(cols):
                    cell_x1 = x_min + (col * cell_width)
                    cell_y1 = y_min + (row * cell_height)
                    cell_x2 = cell_x1 + cell_width
                    cell_y2 = cell_y1 + cell_height
                    
                    zone_pts = [
                        (cell_x1, cell_y1),
                        (cell_x2, cell_y1),
                        (cell_x2, cell_y2),
                        (cell_x1, cell_y2)
                    ]
                    
                    center_x = (cell_x1 + cell_x2) / 2
                    center_y = (cell_y1 + cell_y2) / 2
                    dist = cv2.pointPolygonTest(pts, (center_x, center_y), False)
                    
                    if dist >= 0:
                        sub_zones.append({
                            'polygon': zone_pts,
                            'row': row,
                            'col': col,
                            'count': 0,
                            'light_on': False,
                            'last_detection_time': 0
                        })
        
        self.light_states = [False] * len(sub_zones)
        self.light_off_timers = [0] * len(sub_zones)
        
        return sub_zones
    
    def configure_zone_split(self, total_zones, choice):
        """Configure zone splitting based on user input"""
        if len(self.polygon_points) < 4:
            return False, "Need exactly 4 polygon points"
        
        if len(self.polygon_points) != 4:
            return False, "Polygon must have exactly 4 points"
        
        options = self.get_factor_pairs(total_zones)
        
        if choice < 1 or choice > len(options):
            choice = 1
        
        num_rows, num_cols = options[choice - 1]
        self.sub_zones = self.split_polygon_into_grid(self.polygon_points, num_rows, num_cols)
        self.final_polygon = self.polygon_points
        self.is_drawing = False
        self.is_playing = True
        
        self.light_states = [False] * len(self.sub_zones)
        self.light_off_timers = [0] * len(self.sub_zones)
        
        return True, f"Created {len(self.sub_zones)} sub-zones ({num_rows}×{num_cols})"
    
    def reset_polygon(self):
        """Reset all polygon and zone data"""
        self.polygon_points = []
        self.sub_zones = []
        self.final_polygon = None
        self.is_drawing = True
        self.total_count = 0
        self.light_states = []
        self.light_off_timers = []
    
    def add_polygon_point(self, x, y):
        """Add a point to the polygon (MAX 4 POINTS)"""
        if self.is_drawing and len(self.polygon_points) < self.max_polygon_points:
            self.polygon_points.append((x, y))
            return True, len(self.polygon_points)
        elif self.is_drawing and len(self.polygon_points) >= self.max_polygon_points:
            return False, self.max_polygon_points
        return False, len(self.polygon_points)
    
    def is_polygon_complete(self):
        """Check if polygon has 4 points"""
        return len(self.polygon_points) >= self.max_polygon_points
    
    def control_video(self, action):
        """Control video playback"""
        try:
            if action == 'play_pause' and not self.is_drawing:
                self.is_playing = not self.is_playing
                return self.is_playing
            elif action == 'rewind' and not self.is_drawing:
                if self.video_source != 'webcam' and self.cap and self.cap.isOpened():
                    new_pos = max(0, self.current_frame_pos - (5 * self.fps))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    return f"Rewound to frame {int(new_pos)}"
            elif action == 'forward' and not self.is_drawing:
                if self.video_source != 'webcam' and self.cap and self.cap.isOpened():
                    new_pos = min(self.total_frames, self.current_frame_pos + (5 * self.fps))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    return f"Forward to frame {int(new_pos)}"
        except Exception as e:
            print(f"ERROR in video control: {e}")
            self.initialize_camera()
        return None
    
    def process_frame(self):
        """Process a single frame with detection"""
        try:
            if self.cap is None or not self.cap.isOpened():
                print("WARNING: Camera not open, attempting to reinitialize...")
                if not self.initialize_camera():
                    return self._create_error_frame("Camera Error - Check Connection")
            
            current_time = time.time()
            
            if self.is_playing or self.frame_cache is None:
                try:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        self.frame_read_errors += 1
                        print(f"WARNING: Frame read failed ({self.frame_read_errors}/{self.MAX_FRAME_ERRORS})")
                        
                        if self.frame_read_errors >= self.MAX_FRAME_ERRORS:
                            print("ERROR: Too many frame errors, reinitializing camera...")
                            if self.initialize_camera():
                                self.frame_read_errors = 0
                            else:
                                return self._create_error_frame("Camera Disconnected")
                        
                        if self.frame_cache is not None:
                            frame = self.frame_cache
                        else:
                            return self._create_error_frame("Waiting for Camera...")
                    else:
                        self.frame_read_errors = 0
                        self.frame_cache = frame
                except Exception as e:
                    print(f"ERROR reading frame: {e}")
                    self.frame_read_errors += 1
                    if self.frame_read_errors >= self.MAX_FRAME_ERRORS:
                        self.initialize_camera()
                        self.frame_read_errors = 0
                    if self.frame_cache is not None:
                        frame = self.frame_cache
                    else:
                        return self._create_error_frame("Camera Error")
            else:
                frame = self.frame_cache
            
            if frame is None:
                return self._create_error_frame("No Frame Available")
            
            try:
                display_frame = cv2.resize(frame, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
            except Exception as e:
                print(f"ERROR resizing frame: {e}")
                return self._create_error_frame("Frame Resize Error")
            
            output = display_frame.copy()
            self.total_count = 0
            
            if self.is_drawing:
                for i, pt in enumerate(self.polygon_points):
                    cv2.circle(output, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
                    cv2.putText(output, str(i+1), (int(pt[0])+10, int(pt[1])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if len(self.polygon_points) > 1:
                    pts = np.array(self.polygon_points, dtype=np.int32)
                    cv2.polylines(output, [pts], True, (0, 0, 255), 2)
                
                remaining = self.max_polygon_points - len(self.polygon_points)
                if remaining > 0:
                    cv2.putText(output, f"DRAWING: {len(self.polygon_points)}/4 points ({remaining} left)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(output, "POLYGON COMPLETE! Press S to lock & split", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                try:
                    results = self.model(display_frame, classes=[0], verbose=False)
                except Exception as e:
                    print(f"ERROR in YOLO detection: {e}")
                    results = []
                
                for zone in self.sub_zones:
                    zone['count'] = 0
                
                if results:
                    for r in results:
                        try:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box)
                                
                                box_height = y2 - y1
                                head_height = int(box_height * 0.15)
                                head_y1 = y1
                                head_y2 = y1 + head_height
                                cv2.rectangle(output, (x1, head_y1), (x2, head_y2), (255, 0, 0), 1)
                                
                                check_x = int((x1 + x2) / 2)
                                check_y = int(y2)
                                
                                cv2.circle(output, (check_x, check_y), 6, (255, 255, 255), -1)
                                cv2.line(output, (check_x, check_y), (check_x, check_y - 10), (255, 255, 255), 2)
                                
                                person_zone_index = -1
                                
                                for i, zone in enumerate(self.sub_zones):
                                    dist = cv2.pointPolygonTest(
                                        np.array(zone['polygon'], dtype=np.float32), 
                                        (check_x, check_y), 
                                        False
                                    )
                                    
                                    if dist >= 0:
                                        person_zone_index = i
                                        zone['count'] += 1
                                        zone['last_detection_time'] = current_time
                                        self.total_count += 1
                                        break
                                
                                if person_zone_index >= 0:
                                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(output, f"Z{person_zone_index+1}", (x1, y1 - 10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        except Exception as e:
                            print(f"ERROR processing detection box: {e}")
                            continue
                
                for i, zone in enumerate(self.sub_zones):
                    if zone['count'] > 0:
                        self.light_states[i] = True
                        zone['light_on'] = True
                    else:
                        time_since_detection = current_time - zone['last_detection_time']
                        if time_since_detection >= self.DEBOUNCE_DELAY:
                            self.light_states[i] = False
                            zone['light_on'] = False
                
                for i, zone in enumerate(self.sub_zones):
                    pts = np.array(zone['polygon'], dtype=np.int32)
                    
                    if zone['light_on']:
                        base_color = (0, 255, 0)
                    else:
                        base_color = (0, 0, 255)
                    
                    color = (
                        int(base_color[0] * ((i * 73) % 100) / 100),
                        int(base_color[1] * ((i * 137) % 100) / 100),
                        int(base_color[2] * ((i * 201) % 100) / 100)
                    )
                    
                    overlay = output.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)
                    cv2.polylines(output, [pts], True, color, 2)
                    
                    zone_center_x = int(np.mean([p[0] for p in zone['polygon']]))
                    zone_center_y = int(np.mean([p[1] for p in zone['polygon']]))
                    light_status = "💡 ON" if zone['light_on'] else "💡 OFF"
                    cv2.putText(output, f"Z{i+1}:{zone['count']} {light_status}", 
                               (zone_center_x - 40, zone_center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if self.final_polygon:
                    pts = np.array(self.final_polygon, dtype=np.int32)
                    cv2.polylines(output, [pts], True, (255, 255, 0), 3)
            
            if self.video_source != 'webcam' and self.cap and self.cap.isOpened():
                try:
                    self.current_frame_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                except:
                    pass
            
            output = self.draw_controller_bar(output)
            
            try:
                _, buffer = cv2.imencode('.jpg', output)
                frame_bytes = buffer.tobytes()
                return frame_bytes
            except Exception as e:
                print(f"ERROR encoding frame: {e}")
                return self._create_error_frame("Encoding Error")
                
        except Exception as e:
            print(f"CRITICAL ERROR in process_frame: {e}")
            return self._create_error_frame(f"System Error: {str(e)[:30]}")
    
    def _create_error_frame(self, message):
        """Create a black error frame with text"""
        frame = np.zeros((self.VIDEO_HEIGHT, self.VIDEO_WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame, message, (50, self.VIDEO_HEIGHT//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        except:
            return None
    
    def draw_controller_bar(self, frame):
        """Draw video controller bar at bottom"""
        h, w, _ = frame.shape
        bar_height = 50
        bar = np.zeros((bar_height, w, 3), dtype=np.uint8) + 30
        
        if self.total_frames > 0:
            fill = int((self.current_frame_pos / self.total_frames) * (w - 20))
            cv2.rectangle(bar, (10, 15), (w-10, 35), (100, 100, 100), -1)
            cv2.rectangle(bar, (10, 15), (10 + fill, 35), (0, 255, 0), -1)
        
        cv2.putText(bar, f"Frame: {int(self.current_frame_pos)}/{self.total_frames}", 
                   (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        status = "PLAYING" if self.is_playing else "PAUSED"
        color = (0, 255, 0) if self.is_playing else (0, 0, 255)
        cv2.putText(bar, status, (w - 120, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(bar, f"Total: {self.total_count}", (w//2 - 40, 42), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return np.vstack((frame, bar))
    
    def get_zone_options(self, total_zones):
        """Get available matrix options for zone count"""
        options = self.get_factor_pairs(total_zones)
        return [{'choice': i+1, 'rows': r, 'cols': c} for i, (r, c) in enumerate(options)]
    
    def get_status(self):
        """Get current detector status"""
        return {
            'is_drawing': self.is_drawing,
            'is_playing': self.is_playing,
            'polygon_points': len(self.polygon_points),
            'max_polygon_points': self.max_polygon_points,
            'polygon_complete': self.is_polygon_complete(),
            'sub_zones': len(self.sub_zones),
            'total_count': self.total_count,
            'current_frame': int(self.current_frame_pos),
            'total_frames': self.total_frames,
            'video_source': self.video_source,
            'light_states': self.light_states,
            'debounce_delay': self.DEBOUNCE_DELAY,
            'camera_open': self.cap.isOpened() if self.cap else False
        }
    
    def get_light_status(self):
        """Get light status for all zones"""
        return [
            {'zone': i+1, 'light_on': state, 'count': self.sub_zones[i]['count'] if i < len(self.sub_zones) else 0}
            for i, state in enumerate(self.light_states)
        ]
    
    def set_light_state(self, light_number, state):
        """Manually set light state (for dashboard control)"""
        if 1 <= light_number <= len(self.light_states):
            self.light_states[light_number - 1] = state
            if light_number <= len(self.sub_zones):
                self.sub_zones[light_number - 1]['light_on'] = state
                self.sub_zones[light_number - 1]['last_detection_time'] = time.time() if state else 0
            return True
        return False
    
    def get_all_light_states_dict(self):
        """Get light states as dictionary for ESP API"""
        light_dict = {}
        for i, state in enumerate(self.light_states):
            light_dict[f"light{i+1}"] = "on" if state else "off"
        return light_dict
    
    def release(self):
        """Release video capture"""
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None


# ============================================
# GLOBAL LIGHT STATE STORAGE (For ESP API)
# These functions MUST be at module level
# ============================================

GLOBAL_LIGHT_STATES = {}
GLOBAL_LIGHT_INITIALIZED = False

def get_global_light_states():
    """Get global light states for ESP API"""
    return GLOBAL_LIGHT_STATES.copy()

def set_global_light_state(light_number, state):
    """Set global light state for ESP API"""
    GLOBAL_LIGHT_STATES[f"light{light_number}"] = "on" if state else "off"

def initialize_global_lights(num_lights):
    """Initialize global light states"""
    global GLOBAL_LIGHT_INITIALIZED
    for i in range(1, num_lights + 1):
        if f"light{i}" not in GLOBAL_LIGHT_STATES:
            GLOBAL_LIGHT_STATES[f"light{i}"] = "off"
    GLOBAL_LIGHT_INITIALIZED = True

def sync_global_lights_from_detector(detector):
    """Sync global light states from detector (with debouncing)"""
    if detector and len(detector.sub_zones) > 0:
        for i, zone in enumerate(detector.sub_zones):
            set_global_light_state(i + 1, zone['light_on'])