from django.contrib import admin
from .models import Document, QueryLog, UserSession

admin.site.register(Document)
admin.site.register(QueryLog)
admin.site.register(UserSession)