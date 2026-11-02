import secrets
from django.db import models
from datetime import timedelta
from django.utils import timezone
from django.conf import settings

SESSION_LIFETIME = timedelta(hours=24)

def generate_token():
    return secrets.token_hex(32)

def default_expiry():
    return timezone.now() + SESSION_LIFETIME

class Document(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="documents"
    )

    class Meta:
        ordering = ["-uploaded_at", "-id"]

    def __str__(self):
        return self.title

class QueryLog(models.Model):
    class Kind(models.TextChoices):
        QUERY = "query", "Query"          # Assignment 2's POST /query
        ASK_AI = "ask_ai", "Ask AI"       # POST /api/ask-ai
        RAG = "rag", "RAG query"          # POST /api/rag-query

    query_text = models.TextField()
    result_summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="query_logs",
    )
    kind = models.CharField(max_length=10, choices=Kind.choices, default=Kind.QUERY)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return self.query_text[:50]

class UserSession(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="sessions"
    )
    token = models.CharField(max_length=64, unique=True, default=generate_token)
    create_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(default=default_expiry)

    def is_expired(self):
        return timezone.now() >= self.expires_at

    def __str__(self):
        return f"Session for {self.user}" 
