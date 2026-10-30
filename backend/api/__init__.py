"""HTTP route modules."""
from . import status, search, videos, settings, index, export, databases, setup, scene, tags, segment, derush, diagnostics, models, timeline
from . import searchplus, characters
from . import finish
from . import reembed
from . import autotune
from . import shutdown
from . import music
from . import roto

ROUTERS = [
    status.router,
    setup.router,
    search.router,
    tags.router,
    videos.router,
    scene.router,
    segment.router,
    derush.router,
    settings.router,
    index.router,
    export.router,
    timeline.router,
    databases.router,
    diagnostics.router,
    models.router,
    searchplus.router,
    characters.router,
    finish.router,
    reembed.router,
    autotune.router,
    shutdown.router,
    music.router,
    roto.router,
]
