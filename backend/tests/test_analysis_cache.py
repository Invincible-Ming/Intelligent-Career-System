"""Real Redis, disposable user namespaces, no resumes or cloud model calls."""
import asyncio
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from redis.exceptions import ConnectionError as RedisConnectionError

from app.services.analysis_cache import AnalysisCache
from app.core.config import settings
from app.core.schemas import JobAnalysis, ResumeAnalysis


class AnalysisCacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cache = AnalysisCache()
        await self.cache.client.ping()
        self.owners = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.args = dict(owner_id=self.owners[0], kind='resume', text='Synthetic Python skills',
                         system_prompt='Extract facts only', prompt_version='test-v1', response_model=ResumeAnalysis)

    async def asyncTearDown(self):
        import hashlib
        for owner in self.owners:
            prefix=hashlib.sha256(owner.encode()).hexdigest()
            keys=[key async for key in self.cache.client.scan_iter(match=f'career:analysis:v1:{prefix}:*')]
            if keys:await self.cache.client.delete(*keys)
        await self.cache.close()

    async def test_same_user_hit_and_ttl(self):
        compute=AsyncMock(return_value=ResumeAnalysis(skills=['Python']))
        first,hit=await self.cache.get_or_compute(**self.args,compute=compute)
        self.assertFalse(hit)
        second,hit=await self.cache.get_or_compute(**self.args,compute=compute)
        self.assertTrue(hit);self.assertEqual(first,second);compute.assert_awaited_once()
        ttl=await self.cache.client.ttl(self.cache.key(**self.args))
        self.assertGreater(ttl,0);self.assertLessEqual(ttl,settings.ANALYSIS_CACHE_TTL)

    async def test_user_content_prompt_model_and_schema_separate(self):
        compute=AsyncMock(return_value=ResumeAnalysis(skills=['Python']))
        await self.cache.get_or_compute(**self.args,compute=compute)
        variants=[{'owner_id':self.owners[1]},{'text':'Different facts'},
                  {'prompt_version':'test-v2'},{'system_prompt':'Changed prompt'}]
        for variant in variants:
            _,hit=await self.cache.get_or_compute(**{**self.args,**variant},compute=compute)
            self.assertFalse(hit)
        with patch.object(settings,'BAILIAN_CHAT_MODEL','different-model'):
            _,hit=await self.cache.get_or_compute(**self.args,compute=compute)
            self.assertFalse(hit)
        self.assertNotEqual(self.cache.key(**self.args),self.cache.key(**{**self.args,'response_model':JobAnalysis}))
        self.assertEqual(compute.await_count,6)

    async def test_concurrent_requests_reuse_one_model_call(self):
        async def compute():
            await asyncio.sleep(.05)
            return ResumeAnalysis(skills=['Python'])
        compute=AsyncMock(side_effect=compute)
        results=await asyncio.gather(*(self.cache.get_or_compute(**self.args,compute=compute) for _ in range(3)))
        compute.assert_awaited_once()
        self.assertEqual(sum(hit for _,hit in results),2)

    async def test_corrupt_or_expired_cache_recomputed(self):
        key=self.cache.key(**self.args)
        await self.cache.client.set(key,'{"skills": 123}',ex=5)
        compute=AsyncMock(return_value=ResumeAnalysis(skills=['Python']))
        _,hit=await self.cache.get_or_compute(**self.args,compute=compute)
        self.assertFalse(hit)
        await self.cache.client.pexpire(key,1)
        await asyncio.sleep(.02)
        _,hit=await self.cache.get_or_compute(**self.args,compute=compute)
        self.assertFalse(hit);self.assertEqual(compute.await_count,2)

    async def test_unavailable_redis_falls_back_without_repeating_compute(self):
        fake=AsyncMock()
        fake.get.side_effect=RedisConnectionError('disconnected')
        compute=AsyncMock(return_value=ResumeAnalysis(skills=['Python']))
        with patch.object(self.cache,'_client',fake):
            value,hit=await self.cache.get_or_compute(**self.args,compute=compute)
        self.assertFalse(hit);self.assertEqual(value.skills,['Python']);compute.assert_awaited_once()

    async def test_write_failure_does_not_repeat_model_call(self):
        fake=AsyncMock()
        fake.get.return_value=None
        fake.set.side_effect=[True,RedisConnectionError('disconnected')]
        compute=AsyncMock(return_value=ResumeAnalysis(skills=['Python']))
        with patch.object(self.cache,'_client',fake):
            value,hit=await self.cache.get_or_compute(**self.args,compute=compute)
        self.assertFalse(hit);self.assertEqual(value.skills,['Python']);compute.assert_awaited_once()

    async def test_cancellation_releases_lock_and_does_not_cache_partial_result(self):
        entered=asyncio.Event()
        async def compute():
            entered.set();await asyncio.sleep(10)
        task=asyncio.create_task(self.cache.get_or_compute(**self.args,compute=compute))
        await asyncio.wait_for(entered.wait(),1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        key=self.cache.key(**self.args)
        self.assertIsNone(await self.cache.client.get(key))
        self.assertIsNone(await self.cache.client.get(key+':lock'))

    async def test_missing_owner_bypasses_shared_cache(self):
        compute=AsyncMock(return_value=ResumeAnalysis(skills=['Python']))
        _,hit=await self.cache.get_or_compute(**{**self.args,'owner_id':None},compute=compute)
        self.assertFalse(hit);compute.assert_awaited_once()

    async def test_reasoning_mode_change_does_not_reuse_old_analysis(self):
        compute=AsyncMock(return_value=ResumeAnalysis(skills=['Python']))
        with patch.object(settings,'BAILIAN_CHAT_MODEL','qwen3.8-flash'):
            with patch.object(settings,'MATCH_REASONING_EFFORT','none'):
                await self.cache.get_or_compute(**self.args,compute=compute)
            with patch.object(settings,'MATCH_REASONING_EFFORT','low'):
                _,hit=await self.cache.get_or_compute(**self.args,compute=compute)
        self.assertFalse(hit);self.assertEqual(compute.await_count,2)
