"""Short mobile requests submit long jobs; all work re-enters existing API guards."""
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from app.api.api import MatchResumeRequest
from app.core.schemas import InterviewRequest, LearningRequest, MatchRequest
from app.security.auth import CurrentUser
from app.services.mobile_jobs import cancel_job, load_job, start_job

router = APIRouter(prefix="/mobile", tags=["小程序"])


class MobileJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_request_id: UUID
    kind: Literal["match", "resume", "interview", "learning"]
    payload: dict


@router.post("/jobs", status_code=202)
async def create_mobile_job(request: Request, body: MobileJobRequest, user: CurrentUser):
    schema = {"match": MatchRequest, "resume": MatchResumeRequest,
              "interview": InterviewRequest, "learning": LearningRequest}[body.kind]
    try:
        payload = schema.model_validate(body.payload).model_dump(mode="json")
    except ValidationError:
        raise HTTPException(422, "任务参数不完整或超出范围") from None
    token = request.headers["authorization"].split(None, 1)[1]
    return await start_job(app=request.app, owner=user.id, identifier=body.client_request_id,
                           kind=body.kind, payload=payload, token=token)


@router.get("/jobs/{identifier}")
async def mobile_job(identifier: UUID, user: CurrentUser):
    return await load_job(user.id, identifier)


@router.delete("/jobs/{identifier}")
async def stop_mobile_job(identifier: UUID, user: CurrentUser):
    return await cancel_job(user.id, identifier)
