"""Cross-video tag aggregations (used by autocomplete)."""
from fastapi import APIRouter, Query

from ..db import schema
from ..state import get_state

router = APIRouter()


@router.get("/api/tags/all")
def all_tags(limit: int = Query(default=500, ge=1, le=10_000)):
    state = get_state()
    with schema.get_conn(state.primary_db) as conn:
        rows = conn.execute(
            """SELECT tag, COUNT(*) AS c
                 FROM scene_tags
                GROUP BY tag
                ORDER BY c DESC
                LIMIT ?""",
            (limit,),
        ).fetchall()
    return {"tags": [{"tag": t, "count": c} for t, c in rows]}
