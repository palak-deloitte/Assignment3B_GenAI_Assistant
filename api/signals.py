import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from . import rag
from .models import Document

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=Document)
def remove_vectors_when_document_is_deleted(sender, instance, **kwargs):
    """Keep the vector store in step with the database, however the delete happened."""
    try:
        rag.remove_document(instance.owner_id, instance.id)
    except Exception:
        # A vector-store problem must never turn a successful delete into an error
        logger.exception("Could not remove vectors for document %s", instance.id)