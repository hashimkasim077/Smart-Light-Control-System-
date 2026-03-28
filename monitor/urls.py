from django.urls import path
from . import views

app_name = 'monitor'

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    path('video-stream/', views.video_stream, name='video_stream'),
    
    # Polygon & Zone Management
    path('add-point/', views.add_polygon_point, name='add_point'),
    path('configure-zones/', views.configure_zones, name='configure_zones'),
    path('reset-polygon/', views.reset_polygon, name='reset_polygon'),
    
    # Video Controls
    path('video-control/', views.video_control, name='video_control'),
    path('set-source/', views.set_video_source, name='set_source'),
    
    # Status & Light Status
    path('get-status/', views.get_status, name='get_status'),
    path('get-zone-options/', views.get_zone_options, name='get_zone_options'),
    path('get-light-status/', views.get_light_status, name='get_light_status'),
    path('set-light-state/', views.set_light_state, name='set_light_state'),
    
    # ESP Light Control API
    path('api/esp/lights/', views.esp_get_lights, name='esp_get_lights'),
    path('api/esp/light/set/', views.esp_set_light, name='esp_set_light'),
    path('api/esp/light/<int:light_number>/', views.esp_get_light, name='esp_get_light'),
    path('api/esp/sync/', views.esp_sync_lights, name='esp_sync_lights'),
]