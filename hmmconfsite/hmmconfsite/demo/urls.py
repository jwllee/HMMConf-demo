from django.urls import path, re_path, register_converter
from . import views


app_name = 'demo'
urlpatterns = [
    path('', views.index, name='index'),
    path('upload_data', views.UploadDataView.as_view(), name='upload_data'),
    path('ajax/get_barplot_case', views.get_barplot_case, name='get_barplot_case'),
]