import logging
from django.conf import settings
from django.contrib.auth import authenticate
from rest_framework import generics, status
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document, QueryLog, UserSession
from .serializers import DocumentSerializer, LoginSerializer, QueryLogSerializer, RegisterSerializer, AskAISerializer, IngestDocSerializer,RagQuerySerializer 
from .uploads import PayloadTooLarge, extract_text
from rest_framework.permissions import SAFE_METHODS, AllowAny, IsAuthenticated
from .permissions import IsOwner
from . import llm, rag

logger = logging.getLogger(__name__)

NO_DOCS_MESSAGE = (
    "I couldn't find any ingested documents to search. "
    "Upload a document and ingest it first."
)

def session_payload(session):
    return {"token": session.token, "expires_at": session.expires_at}

class RegisterView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        session = UserSession.objects.create(user=user)
        return Response(
            {"user": serializer.data, **session_payload(session=session)},
            status=status.HTTP_201_CREATED
        )

class LoginView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = authenticate(
            username=serializer.validated_data["username"],
            password=serializer.validated_data["password"]
        )

        if user is None:
            return Response(
                {"detail": "Invalid Username or password"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        session = UserSession.objects.create(user=user)
        return Response(session_payload(session=session))


class LogoutView(APIView):
    def post(self, request):
        request.auth.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class DocumentListCreateView(generics.ListCreateAPIView):
    serializer_class = DocumentSerializer
    parser_classes = [JSONParser, MultiPartParser]

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user)

    def create(self, request, *args, **kwargs):
        upload = request.FILES.get("file")
        if upload is not None:
            data = {
                "title": request.data.get("title") or upload.name[:255],
                "content": extract_text(upload),
            }
        else:
            data = request.data

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception = True)

        if len(serializer.validated_data["content"].encode("utf-8")) > settings.MAX_UPLOAD_BYTES:
            raise PayloadTooLarge("Content exceeds the maximum allowed size.")

        serializer.save(owner=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

class DocumentDetailView(generics.RetrieveDestroyAPIView):
    serializer_class = DocumentSerializer
    permission_classes = [IsAuthenticated, IsOwner]

    def get_queryset(self):
        qs = Document.objects.all()
        if self.request.method in SAFE_METHODS:
            return qs.filter(owner=self.request.user)
        return qs
    
    def perform_destroy(self, instance):
        owner_id, document_id = instance.owner_id, instance.id   # read these first
        instance.delete()
        try:
            rag.remove_document(owner_id, document_id)
        except Exception:
            logger.exception("Could not remove vectors for document %s", document_id)

class QueryListView(generics.ListAPIView):
    serializer_class = QueryLogSerializer

    def get_queryset(self):
        return QueryLog.objects.filter(user=self.request.user)

class QueryCreateView(generics.CreateAPIView):
    serializer_class = QueryLogSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class AskAIView(APIView):
    def post(self, request):
        serializer = AskAISerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        prompt = serializer.validated_data["prompt"]

        try:
            answer = llm.ask(prompt=prompt)
        except llm.LLMError:
            return Response(
                {"detail": "The AI service is unavailable. Please try again."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        QueryLog.objects.create(
            user=request.user,
            query_text=prompt,
            result_summary=answer,
            kind=QueryLog.Kind.ASK_AI
        )
        return Response({'response': answer})

class IngestDocView(APIView):
    def post(self, request):
        serializer = IngestDocSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        document_id = serializer.validated_data["document_id"]

        # Filter by owner: documents that aren't yours are "not found"
        try:
            doc = Document.objects.get(id=document_id, owner=request.user)
        except Document.DoesNotExist:
            return Response(
                {"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND
            )

        try:
            chunk_count = rag.index_document(
                request.user.id, doc.id, doc.title, doc.content
            )
        except Exception:
            logger.exception("Ingestion failed for document %s", doc.id)
            return Response(
                {"detail": "Could not index the document. Please try again."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if chunk_count == 0:
            return Response(
                {"detail": "The document has no text to index."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {"document_id": doc.id, "status": "ingested", "chunk_count": chunk_count}
        )

class RagQueryView(APIView):
    def post(self, request):
        serializer = RagQuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        query = serializer.validated_data["query"]

        # 1. Retrieve: only from this user's own index
        try:
            chunks = rag.search(request.user.id, query)
        except Exception:
            logger.exception("Retrieval failed")
            return Response(
                {"detail": "Could not search your documents. Please try again."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # 2. Generate: skip the model entirely if there is nothing to search
        if not chunks:
            answer = NO_DOCS_MESSAGE
        else:
            try:
                answer = llm.answer_from_context(query, chunks)
            except llm.LLMError:
                return Response(
                    {"detail": "The AI service is unavailable. Please try again."},
                    status=status.HTTP_502_BAD_GATEWAY,
                )

        QueryLog.objects.create(
            user=request.user,
            query_text=query,
            result_summary=answer,
            kind=QueryLog.Kind.RAG,
        )
        return Response({"answer": answer, "sources": chunks})

class AIQueryHistoryView(generics.ListAPIView):
    serializer_class = QueryLogSerializer

    def get_queryset(self):
        return QueryLog.objects.filter(
            user=self.request.user,
            kind__in=[QueryLog.Kind.ASK_AI, QueryLog.Kind.RAG],
        )
