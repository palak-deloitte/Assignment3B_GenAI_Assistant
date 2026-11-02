from rest_framework.permissions import BasePermission


class IsOwner(BasePermission):
    """Object-level check: only the owner may act on this object."""

    def has_object_permission(self, request, view, obj):
        return obj.owner_id == request.user.id