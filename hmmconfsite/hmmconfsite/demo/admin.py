from django.contrib import admin
from .models import Event, Log, State

# Register your models here.
admin.site.register(Event)
admin.site.register(Log)
admin.site.register(State)
