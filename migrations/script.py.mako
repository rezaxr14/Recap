from logging.config import fileConfig
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recap.infrastructure.db.models import Base  # noqa: E402
