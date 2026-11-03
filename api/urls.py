from django.urls import path
from . import views

urlpatterns = [
    path("auth/register", views.RegisterView.as_view()),
    path("auth/login", views.LoginView.as_view()),
    path("auth/logout", views.LogoutView.as_view()),
    path("documents", views.DocumentListCreateView.as_view()),
    path("documents/<int:pk>", views.DocumentDetailView.as_view()),
    path("query", views.QueryCreateView.as_view()),
    path("queries", views.QueryListView.as_view()),
    path("api/ask-ai", views.AskAIView.as_view()),
    path("api/ingest-doc", views.IngestDocView.as_view()),
    path("api/rag-query", views.RagQueryView.as_view()),
    path("api/query-history", views.AIQueryHistoryView.as_view()),
    path("api/chat", views.ChatView.as_view()),
]