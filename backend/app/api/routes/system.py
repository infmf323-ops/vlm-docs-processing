from fastapi import APIRouter


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
def healthcheck():
    return {"status": "ok"}
