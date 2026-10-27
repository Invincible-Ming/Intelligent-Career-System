"""Transactional points ledger. No prompts, API keys or document text are stored here."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

import anyio
from langchain_core.callbacks import AsyncCallbackHandler
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.models import CreditEntry, CreditWallet

logger = logging.getLogger(__name__)
UNITS = 1000
BEIJING = timezone(timedelta(hours=8))


class InsufficientCredits(BaseException):
    """Must escape agent retries, optional-tool fallbacks and model-error wrappers."""

    def __init__(self):
        super().__init__("可用积分不足以预留本次模型调用，请签到领取积分后重试，或缩短输入。")


@dataclass(frozen=True)
class CreditActor:
    owner_id: uuid.UUID
    operation: str


credit_actor: ContextVar[CreditActor | None] = ContextVar("credit_actor", default=None)


@contextmanager
def billing_scope(owner_id, operation):
    token = credit_actor.set(CreditActor(uuid.UUID(str(owner_id)), operation[:160]))
    try:
        yield
    finally:
        credit_actor.reset(token)


async def ensure_wallet(session, owner_id):
    """Old and new accounts receive the gift exactly once, including across workers."""
    owner_id = uuid.UUID(str(owner_id))
    initial = settings.CREDITS_INITIAL_POINTS * UNITS
    created = (await session.execute(insert(CreditWallet).values(
        user_id=owner_id, balance_units=initial, reserved_units=0,
    ).on_conflict_do_nothing(index_elements=[CreditWallet.user_id]).returning(CreditWallet.user_id)))
    if created.scalar_one_or_none() is not None:
        session.add(CreditEntry(user_id=owner_id, idempotency_key=f"initial:{owner_id}",
                                kind="initial", status="settled", delta_units=initial,
                                balance_after_units=initial, settled_at=datetime.now(UTC)))
        await session.flush()
    return (await session.execute(select(CreditWallet).where(
        CreditWallet.user_id == owner_id).with_for_update())).scalar_one()


async def release_expired(session, wallet, now):
    # Every ledger writer first locks the wallet, then entries: one consistent lock order.
    entries = (await session.execute(select(CreditEntry).where(
        CreditEntry.user_id == wallet.user_id, CreditEntry.status == "pending", CreditEntry.expires_at <= now,
    ).with_for_update())).scalars().all()
    for entry in entries:
        wallet.reserved_units -= entry.reserved_units
        entry.status = "released"
        entry.settled_at = now
        entry.balance_after_units = wallet.balance_units


def wallet_response(wallet, *, now=None):
    now = now or datetime.now(UTC)
    today = now.astimezone(BEIJING).date()
    return {"balance": wallet.balance_units / UNITS, "reserved": wallet.reserved_units / UNITS,
            "available": (wallet.balance_units - wallet.reserved_units) / UNITS,
            "checked_in_today": wallet.last_checkin_date == today,
            "last_checkin_date": wallet.last_checkin_date.isoformat() if wallet.last_checkin_date else None,
            "rules": credit_rules()}


def credit_rules():
    return {"initial_points": settings.CREDITS_INITIAL_POINTS,
            "daily_checkin_points": settings.CREDITS_DAILY_CHECKIN_POINTS,
            "points_per_1000_tokens": settings.CREDITS_UNITS_PER_TOKEN,
            "timezone": "Asia/Shanghai", "precision": 3}


async def wallet_info(session, owner_id):
    wallet = await ensure_wallet(session, owner_id)
    await release_expired(session, wallet, datetime.now(UTC))
    result = wallet_response(wallet)
    await session.commit()
    return result


async def check_in(session, owner_id, *, now=None):
    now = now or datetime.now(UTC)
    today = now.astimezone(BEIJING).date()
    wallet = await ensure_wallet(session, owner_id)
    await release_expired(session, wallet, now)
    awarded = wallet.last_checkin_date != today
    if awarded:
        amount = settings.CREDITS_DAILY_CHECKIN_POINTS * UNITS
        wallet.balance_units += amount
        wallet.last_checkin_date = today
        session.add(CreditEntry(user_id=wallet.user_id, idempotency_key=f"checkin:{wallet.user_id}:{today}",
                                kind="checkin", status="settled", delta_units=amount,
                                balance_after_units=wallet.balance_units, settled_at=now))
    result = {**wallet_response(wallet, now=now), "awarded": awarded}
    await session.commit()
    return result


def estimate_tokens(messages):
    """Conservative text preflight only; provider usage replaces this at settlement."""
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") in ("image", "image_url"):
                    total += 16384  # OCR pixels are not base64 text tokens.
                else:
                    total += len(json.dumps(part, ensure_ascii=False).encode("utf-8"))
        else:
            total += len(str(content).encode("utf-8"))
        total += 32
    return total


def provider_usage(usage):
    """Normalize OpenAI, LangChain and DashScope; reasoning is already part of output."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return None
    input_count = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_count = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    values = [value for value in (input_count, output_count, total) if value is not None]
    if not values or any(type(value) is not int or value < 0 for value in values):
        return None
    if input_count is None:
        input_count = max(0, (total or 0) - (output_count or 0))
    if output_count is None:
        output_count = max(0, (total or input_count) - input_count)
    # Some providers omit reasoning from completion_tokens, but include it in total_tokens.
    output_count = max(output_count, (total or 0) - input_count)
    return input_count, output_count


@dataclass
class ModelCharge:
    entry_id: uuid.UUID | None = None
    owner_id: uuid.UUID | None = None
    input_estimate: int = 0
    output_text: str = ""
    usage: tuple[int, int] | None = None
    completed: bool = False

    @classmethod
    async def begin(cls, model, messages, output_cap, *, extra_input=0):
        charge = cls(input_estimate=estimate_tokens(messages) + extra_input)
        actor = credit_actor.get()
        if actor is None:
            return charge  # Health probes/maintenance are not silently assigned to a random user.
        charge.entry_id, charge.owner_id = uuid.uuid4(), actor.owner_id
        rate = settings.CREDITS_UNITS_PER_TOKEN
        held = (charge.input_estimate + output_cap) * rate
        async with AsyncSessionLocal() as session:
            wallet = await ensure_wallet(session, actor.owner_id)
            await release_expired(session, wallet, datetime.now(UTC))
            if wallet.balance_units - wallet.reserved_units < held:
                await session.commit()  # Persist first-time gift and stale-hold recovery even on refusal.
                raise InsufficientCredits()
            wallet.reserved_units += held
            session.add(CreditEntry(
                id=charge.entry_id, user_id=actor.owner_id, idempotency_key=f"model:{charge.entry_id}",
                kind="usage", status="pending", reserved_units=held, units_per_token=rate,
                model=str(model)[:160], operation=actor.operation,
                expires_at=datetime.now(UTC) + timedelta(seconds=max(settings.MODEL_TIMEOUT,
                    settings.MATCH_MODEL_TIMEOUT, settings.KNOWLEDGE_INDEX_TIMEOUT) + 120),
            ))
            await session.commit()
        return charge

    def record(self, *, usage=None, output="", completed=True):
        self.usage = provider_usage(usage) or self.usage
        self.output_text = output
        self.completed = completed

    async def settle(self):
        if self.entry_id is None:
            return
        async with AsyncSessionLocal() as session:
            wallet = (await session.execute(select(CreditWallet).where(
                CreditWallet.user_id == self.owner_id).with_for_update())).scalar_one()
            entry = await session.get(CreditEntry, self.entry_id)
            if entry.status != "pending":
                return  # Retry/disconnect cleanup cannot debit the same receipt twice.
            if self.usage is not None:
                inputs, outputs = self.usage
                source = "provider"
            elif self.completed or self.output_text:
                inputs, outputs = self.input_estimate, len(self.output_text.encode("utf-8"))
                source = "estimated"
            else:
                inputs, outputs, source = 0, 0, "unavailable"
            requested = (inputs + outputs) * entry.units_per_token
            # A provider can exceed its advertised cap (vision/reasoning). Never spend another
            # concurrent request's reservation or create debt; record full usage, cap this receipt.
            actual = min(requested, entry.reserved_units)
            if actual < requested:
                source += "_capped"
                logger.warning("模型用量超过预留上限，已限制本次扣分，entry_id=%s", entry.id)
            wallet.reserved_units -= entry.reserved_units
            wallet.balance_units -= actual
            entry.delta_units = -actual
            entry.input_tokens, entry.output_tokens, entry.usage_source = inputs, outputs, source
            entry.status = "settled" if (actual or self.completed) else "released"
            entry.settled_at = datetime.now(UTC)
            entry.balance_after_units = wallet.balance_units
            await session.commit()

    async def finish(self):
        # Settle partial streams even when the surrounding AnyIO/asyncio task was cancelled.
        with anyio.CancelScope(shield=True):
            task = asyncio.create_task(self.settle())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise


class CreditModelCallback(AsyncCallbackHandler):
    """The tool-calling ChatOpenAI path does not go through BailianService."""
    raise_error = True
    run_inline = True

    def __init__(self):
        self.charges = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        params = kwargs.get("invocation_params") or {}
        rows = [{"role": message.type, "content": message.content} for batch in messages for message in batch]
        # Tool schemas/call arguments are also input tokens, not just visible message text.
        extra = len(json.dumps(params.get("tools", []), ensure_ascii=False, default=str).encode())
        extra += sum(len(json.dumps(getattr(message, "tool_calls", []), ensure_ascii=False).encode())
                     for batch in messages for message in batch)
        self.charges[run_id] = await ModelCharge.begin(
            params.get("model_name", params.get("model", settings.BAILIAN_CHAT_MODEL)), rows,
            params.get("max_tokens") or settings.CHAT_MAX_OUTPUT_TOKENS, extra_input=extra,
        )

    async def on_llm_end(self, response, *, run_id, **kwargs):
        charge = self.charges.pop(run_id, None)
        if charge is None:
            return
        usage = (response.llm_output or {}).get("token_usage")
        texts = []
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                usage = usage or getattr(message, "usage_metadata", None)
                texts.append(str(getattr(generation, "text", "")))
                texts.append(json.dumps(getattr(message, "tool_calls", []), ensure_ascii=False))
        charge.record(usage=usage, output="".join(texts))
        await charge.finish()

    async def on_llm_error(self, error, *, run_id, **kwargs):
        charge = self.charges.pop(run_id, None)
        if charge is not None:
            await charge.finish()
