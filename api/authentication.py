from rest_framework import authentication, exceptions

from .models import UserSession


class SessionTokenAuthentication(authentication.BaseAuthentication):
    """Authenticate requests carrying 'Authorization: Token <token>'."""

    keyword = "Token"

    def authenticate(self, request):
        parts = authentication.get_authorization_header(request).split()

        # No Authorization header, or a different scheme: not our job, skip.
        if not parts or parts[0].lower() != self.keyword.lower().encode():
            return None
        if len(parts) != 2:
            raise exceptions.AuthenticationFailed("Invalid token header.")

        try:
            token = parts[1].decode()
        except UnicodeError:
            raise exceptions.AuthenticationFailed("Invalid token header.")

        try:
            session = UserSession.objects.select_related("user").get(token=token)
        except UserSession.DoesNotExist:
            raise exceptions.AuthenticationFailed("Invalid token.")

        if session.is_expired():
            session.delete()
            raise exceptions.AuthenticationFailed("Token has expired.")
        if not session.user.is_active:
            raise exceptions.AuthenticationFailed("User is inactive.")

        return (session.user, session)   # becomes request.user, request.auth

    def authenticate_header(self, request):
        return self.keyword