from django.contrib.auth import get_user_model
from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import Document, QueryLog

User = get_user_model()

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])

    class Meta: 
        model =  User
        fields = ['id', 'username', 'email', 'password']
        extra_kwargs = {"email": {"required": False}}

    def create(self, validated_data):
        return User.objects.create_user(**validated_data) 

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'content', 'uploaded_at', 'owner']
        read_only_fields = ["id", "uploaded_at", "owner"]

class QueryLogSerializer(serializers.ModelSerializer):
    query_text = serializers.CharField(max_length=2000)

    class Meta:
        model = QueryLog
        fields = ["id", "query_text", "result_summary", "created_at", "user", "kind"]
        read_only_fields = ["id", "result_summary", "created_at", "user", "kind"]

class AskAISerializer(serializers.Serializer):
    prompt = serializers.CharField(max_length=4000)

class IngestDocSerializer(serializers.Serializer):
    document_id = serializers.IntegerField(min_value=1)

class RagQuerySerializer(serializers.Serializer):
    query = serializers.CharField(max_length=2000)

class ChatSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=2000)
    session_id = serializers.RegexField(r"^[A-Za-z0-9_-]{8,64}$", max_length=64)