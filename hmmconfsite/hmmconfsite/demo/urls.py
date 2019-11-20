from django.urls import path, re_path, register_converter
from . import views


app_name = 'demo'
urlpatterns = [
    path('', views.index, name='index'),
    path('about', views.about, name='about'),
    path('upload_data', views.UploadDataView.as_view(), name='upload_data'),
    path('ajax/get_barplot_case', views.get_barplot_case, name='get_barplot_case'),
    path('event_stream/<int:event_id>', views.json_event_stream_data, name='json_event_stream'),
    path('ajax/replay_previous_event', views.replay_previous_event, name='replay_previous_event'),
    path('ajax/replay_next_event', views.replay_next_event, name='replay_next_event'),
    path('ajax/retrieve_record', views.retrieve_record, name='retrieve_record'),
    path('ajax/state_transition', views.state_transition, name='state_transition'),
    path('ajax/observation_update', views.observation_update, name='observation_update'),
    path('json_conformance_data/<int:event_id>', views.json_conformance_data, name='json_conformance'),
    path('ajax/compute_conformance', views.compute_conformance, name='compute_conformance'),
]
