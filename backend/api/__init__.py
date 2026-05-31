"""HTTP route modules."""
from . import status, search, videos, settings, index, export, databases, setup, scene, tags, segment

ROUTERS = [
    status.router,
    setup.router,
    search.router,
    tags.router,
    videos.router,
    scene.router,
    segment.router,
    settings.router,
    index.router,
    export.router,
    databases.router,
]
