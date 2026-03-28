from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import os
import time
from .detector import VideoDetector, get_global_light_states, set_global_light_state, initialize_global_lights, sync_global_lights_from_detector

# Global detector instance (one per session in production)
detector = None

# API Key for ESP Security (Change this in production!)
ESP_API_KEY = "mysecretkey"

def get_detector(request):
    """Get or create detector instance for session"""
    global detector
    
    # Check session for video source preference
    video_source = request.session.get('video_source', 'webcam')
    video_path = request.session.get('video_path', None)
    
    if detector is None or detector.video_source != video_source:
        if detector:
            detector.release()
        detector = VideoDetector(video_source=video_source, video_path=video_path)
        detector.initialize_camera()
    
    return detector

def index(request):
    """Main index page"""
    # Get available videos from uploads folder
    upload_dir = os.path.join(settings.BASE_DIR, 'uploads')
    available_videos = []
    if os.path.exists(upload_dir):
        available_videos = [f for f in os.listdir(upload_dir) 
                          if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    return render(request, 'monitor/index.html', {
        'available_videos': available_videos
    })

def video_stream(request):
    """MJPEG video streaming endpoint"""
    det = get_detector(request)
    
    def generate_frames():
        while True:
            frame = det.process_frame()
            if frame:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

@csrf_exempt
def add_polygon_point(request):
    """Add a polygon point from frontend"""
    if request.method == 'POST':
        data = json.loads(request.body)
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        det = get_detector(request)
        success, count = det.add_polygon_point(x, y)
        is_complete = det.is_polygon_complete()
        
        return JsonResponse({
            'success': success,
            'points_count': count,
            'max_points': det.max_polygon_points,
            'is_complete': is_complete,
            'message': 'Polygon complete! Press S to lock & split' if is_complete else f'Point {count}/4 added'
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

@csrf_exempt
def configure_zones(request):
    """Configure zone splitting"""
    if request.method == 'POST':
        data = json.loads(request.body)
        total_zones = data.get('total_zones', 4)
        choice = data.get('choice', 1)
        
        det = get_detector(request)
        success, message = det.configure_zone_split(total_zones, choice)
        
        if success:
            # Initialize global light states for ESP
            initialize_global_lights(len(det.sub_zones))
            # Sync with detector
            sync_global_lights_from_detector(det)
        
        return JsonResponse({
            'success': success,
            'message': message,
            'zones_count': len(det.sub_zones),
            'options': det.get_zone_options(total_zones),
            'light_count': len(det.sub_zones)
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

@csrf_exempt
def reset_polygon(request):
    """Reset polygon and zones"""
    if request.method == 'POST':
        det = get_detector(request)
        det.reset_polygon()
        
        return JsonResponse({
            'success': True,
            'message': 'Polygon reset - Draw 4 points'
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

@csrf_exempt
def video_control(request):
    """Control video playback"""
    if request.method == 'POST':
        data = json.loads(request.body)
        action = data.get('action', '')
        
        det = get_detector(request)
        result = det.control_video(action)
        
        return JsonResponse({
            'success': True,
            'action': action,
            'result': result,
            'status': det.get_status()
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

@csrf_exempt
def set_video_source(request):
    """Set video source (webcam, upload, or select)"""
    if request.method == 'POST':
        data = json.loads(request.body)
        source = data.get('source', 'webcam')
        video_path = data.get('video_path', None)
        
        request.session['video_source'] = source
        request.session['video_path'] = video_path
        
        # Reinitialize detector
        global detector
        if detector:
            detector.release()
            detector = None
        
        return JsonResponse({
            'success': True,
            'message': f'Video source set to {source}'
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

@csrf_exempt
def get_status(request):
    """Get current detector status"""
    if request.method == 'GET':
        det = get_detector(request)
        return JsonResponse(det.get_status())
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

@csrf_exempt
def get_zone_options(request):
    """Get available zone matrix options"""
    if request.method == 'POST':
        data = json.loads(request.body)
        total_zones = data.get('total_zones', 4)
        
        det = get_detector(request)
        options = det.get_zone_options(total_zones)
        
        return JsonResponse({
            'success': True,
            'options': options
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

@csrf_exempt
def get_light_status(request):
    """Get light status for all zones"""
    if request.method == 'GET':
        det = get_detector(request)
        return JsonResponse({
            'success': True,
            'lights': det.get_light_status()
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

@csrf_exempt
def set_light_state(request):
    """Manually set light state from dashboard"""
    if request.method == 'POST':
        data = json.loads(request.body)
        light_number = data.get('light_number', 0)
        state = data.get('state', 'off')  # 'on' or 'off'
        
        det = get_detector(request)
        state_bool = state.lower() == 'on'
        
        # Set detector light state
        success = det.set_light_state(light_number, state_bool)
        
        # Set global light state for ESP
        set_global_light_state(light_number, state_bool)
        
        return JsonResponse({
            'success': success,
            'light_number': light_number,
            'state': state,
            'message': f'Light {light_number} turned {state}'
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})


# ============================================
# ESP LIGHT CONTROL API ENDPOINTS
# ============================================

def verify_api_key(request):
    """Verify ESP API Key from header"""
    api_key = request.headers.get('X-API-Key', '')
    return api_key == ESP_API_KEY

@csrf_exempt
def esp_get_lights(request):
    """
    ESP API: Get all light states
    ESP device calls this endpoint to check current light states
    
    Requires: X-API-Key header
    """
    if request.method == 'GET':
        # Verify API Key
        if not verify_api_key(request):
            return JsonResponse({
                'success': False,
                'error': 'Invalid or missing API key'
            }, status=403)
        
        # Sync with detector first
        det = get_detector(None)
        sync_global_lights_from_detector(det)
        
        light_states = get_global_light_states()
        
        # If no lights configured, return empty (all OFF by default)
        if not light_states:
            return JsonResponse({
                'success': True,
                'message': 'No lights configured - all OFF by default',
                'lights': {}
            })
        
        return JsonResponse({
            'success': True,
            'lights': light_states
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'}, status=405)

@csrf_exempt
def esp_set_light(request):
    """
    ESP API: Set individual light state
    ESP device can call this to confirm state changes
    
    Requires: X-API-Key header
    """
    if request.method == 'POST':
        # Verify API Key
        if not verify_api_key(request):
            return JsonResponse({
                'success': False,
                'error': 'Invalid or missing API key'
            }, status=403)
        
        data = json.loads(request.body)
        light_number = data.get('light_number', 0)
        state = data.get('state', 'off')
        
        if light_number < 1:
            return JsonResponse({
                'success': False,
                'error': 'Invalid light number'
            })
        
        state_bool = state.lower() == 'on'
        set_global_light_state(light_number, state_bool)
        
        # Also update detector if exists
        global detector
        if detector:
            detector.set_light_state(light_number, state_bool)
        
        return JsonResponse({
            'success': True,
            'light_number': light_number,
            'state': state,
            'message': f'Light {light_number} set to {state}'
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'}, status=405)

@csrf_exempt
def esp_get_light(request, light_number):
    """
    ESP API: Get specific light state
    ESP device calls this to check individual light
    
    Requires: X-API-Key header
    """
    if request.method == 'GET':
        # Verify API Key
        if not verify_api_key(request):
            return JsonResponse({
                'success': False,
                'error': 'Invalid or missing API key'
            }, status=403)
        
        light_states = get_global_light_states()
        light_key = f"light{light_number}"
        
        if light_key in light_states:
            return JsonResponse({
                'success': True,
                'light_number': light_number,
                'state': light_states[light_key]
            })
        else:
            return JsonResponse({
                'success': False,
                'error': f'Light {light_number} not found'
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'}, status=405)

@csrf_exempt
def esp_sync_lights(request):
    """
    ESP API: Sync all lights with current detection states
    Called by ESP to get latest states based on person detection
    
    Requires: X-API-Key header
    """
    if request.method == 'GET':
        # Verify API Key
        if not verify_api_key(request):
            return JsonResponse({
                'success': False,
                'error': 'Invalid or missing API key'
            }, status=403)
        
        det = get_detector(None)  # Get detector without request
        
        if det and len(det.sub_zones) > 0:
            # Sync global states from detector (includes debouncing)
            sync_global_lights_from_detector(det)
            
            light_states = get_global_light_states()
            
            return JsonResponse({
                'success': True,
                'lights': light_states,
                'total_zones': len(det.sub_zones),
                'total_people': det.total_count,
                'debounce_delay': det.DEBOUNCE_DELAY
            })
        else:
            return JsonResponse({
                'success': True,
                'message': 'No zones configured - all lights OFF by default',
                'lights': {}
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid method'}, status=405)