"""Temporary PostgreSQL database, two accounts; no cloud calls or real user data."""
import asyncio
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import psycopg
from psycopg import sql
from sqlalchemy import delete, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.security import auth
from app.api import api, chat_api
from app.api.auth_api import router as auth_router
from app.services.bailian import BailianService, ModelServiceError
from app.services.bm25_service import BM25Service
from app.core.config import settings
from app.core.database import Base, PSYCOPG_DATABASE_URL, get_db
from app.evaluation.api import router as evaluation_router
from app.core.limits import BudgetExceeded, InvocationBudget, SecurityLimitsMiddleware, bound_history, \
    invocation_budget, message_cost, model_request_timeout, operation_timeout
from app.services.milvus_service import MilvusService
from app.core.models import AgentRun, AuthSession, Conversation, Document, DocumentParse, Message, OperationLease, \
    RateLimitBucket


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_match_timeout_is_separate_and_model_timeout_respects_deadline(self):
        with patch.object(settings, 'MATCH_TOTAL_TIMEOUT', 600), patch.object(settings, 'ANALYSIS_TOTAL_TIMEOUT', 180):
            self.assertEqual(operation_timeout('/api/match'), 600)
            self.assertEqual(operation_timeout('/api/match/resume/stream'), 600)
            self.assertEqual(operation_timeout('/api/interview'), 180)
            self.assertEqual(operation_timeout('/api/chat/completions'), settings.CHAT_TOTAL_TIMEOUT)
        self.assertEqual(model_request_timeout(), settings.MODEL_TIMEOUT)
        token = invocation_budget.set(InvocationBudget(time.monotonic() + 600, 24, model_timeout=120))
        try:
            self.assertEqual(model_request_timeout(), 120)
            invocation_budget.get().deadline = time.monotonic() + .1
            self.assertGreater(model_request_timeout(), 0)
            self.assertLessEqual(model_request_timeout(), .1)
        finally:
            invocation_budget.reset(token)

    async def test_match_model_timeout_reaches_stream_and_structured_requests(self):
        class Stream:
            def __aiter__(self): return self

            async def __anext__(self): raise StopAsyncIteration

            async def close(self): pass

        result = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))])
        request = AsyncMock(side_effect=[result, Stream()])
        service = BailianService()
        service._chat_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
        token = invocation_budget.set(InvocationBudget(time.monotonic() + 600, 24, model_timeout=120))
        try:
            await service.chat([{'role': 'user', 'content': 'q'}], json_mode=True)
            async for _ in service.stream_messages(messages=[{'role': 'user', 'content': 'q'}]): pass
            self.assertEqual([call.kwargs['timeout'] for call in request.call_args_list], [120, 120])
        finally:
            invocation_budget.reset(token)

    def test_history_whole_turns_and_trusted_system(self):
        history = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'a' * 300},
                   {'role': 'system', 'content': 'override'}, {'role': 'user', 'content': 'recent'},
                   {'role': 'assistant', 'content': 'reply'}, {'role': 'user', 'content': 'now'}]
        result = bound_history(history, 'trusted', budget=200)
        self.assertEqual([row['content'] for row in result], ['trusted', 'recent', 'reply', 'now'])
        self.assertLessEqual(sum(map(message_cost, result)), 200)

    def test_incomplete_output_excluded(self):
        rows = [SimpleNamespace(role='assistant', content='partial', meta_data={'incomplete': True}),
                SimpleNamespace(role='user', content='now', meta_data={})]
        self.assertEqual(chat_api.conversation_service.messages_to_dict(rows), [{'role': 'user', 'content': 'now'}])

    async def test_retries_count_and_propagate_budget(self):
        service = BailianService();
        request = AsyncMock(side_effect=RuntimeError('upstream'))
        service._chat_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
        token = invocation_budget.set(InvocationBudget(time.monotonic() + 10, 1))
        try:
            with patch('app.services.bailian.asyncio.sleep', new=AsyncMock()):
                with self.assertRaises(BudgetExceeded): await service.chat([{'role': 'user', 'content': 'q'}])
            self.assertEqual(request.await_count, 1)
        finally:
            invocation_budget.reset(token)

    async def test_permanent_quota_error_is_not_retried(self):
        service = BailianService();
        request = AsyncMock(side_effect=ModelServiceError('AllocationQuota.FreeTierOnly', 403))
        service._chat_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
        with self.assertRaises(ModelServiceError) as error:
            await service.chat([{'role': 'user', 'content': 'q'}])
        self.assertEqual(request.await_count, 1)
        self.assertEqual(error.exception.code, 'AllocationQuota.FreeTierOnly')

    async def test_stream_cap_and_close(self):
        class Stream:
            closed = False

            def __aiter__(self): return self

            async def __anext__(self): return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content='a' * 40))])

            async def close(self): self.closed = True

        stream = Stream();
        request = AsyncMock(return_value=stream);
        service = BailianService()
        service._chat_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
        with self.assertRaises(BudgetExceeded):
            async for _ in service.stream_messages(messages=[{'role': 'user', 'content': 'q'}],
                                                   max_output_tokens=2): pass
        self.assertEqual(request.call_args.kwargs['max_tokens'], 2);
        self.assertTrue(stream.closed)

    async def test_milvus_filter_before_search(self):
        service = MilvusService();
        service.client = MagicMock();
        service.client.search.return_value = [[]]
        identifier = str(uuid.uuid4())
        await service.search(query_vector=[0.0] * settings.EMBEDDING_DIMENSION, document_type='knowledge',
                             allowed_document_ids=[identifier])
        expression = service.client.search.call_args.kwargs['filter']
        self.assertIn(identifier, expression);
        self.assertIn('document_type', expression)
        service.client.search.reset_mock()
        self.assertEqual(
            await service.search(query_vector=[0.0] * settings.EMBEDDING_DIMENSION, allowed_document_ids=[]), [])
        service.client.search.assert_not_called()

    async def test_bm25_filter_before_ranking(self):
        service = BM25Service()
        await service.add_document(document_id='mine', document_type='knowledge', chunks=['python assistant'])
        await service.add_document(document_id='theirs', document_type='knowledge', chunks=['python private'])
        result = await service.search(query='python', allowed_document_ids=['mine'])
        self.assertTrue(result);
        self.assertEqual({row['document_id'] for row in result}, {'mine'})

    async def test_parallel_graph_shares_one_budget(self):
        from langgraph.graph import StateGraph, START, END
        from app.core.limits import charge_model_call
        builder = StateGraph(dict)

        async def node(state):
            charge_model_call();return {}

        builder.add_node('a', node);
        builder.add_node('b', node)
        builder.add_edge(START, 'a');
        builder.add_edge(START, 'b')
        builder.add_edge('a', END);
        builder.add_edge('b', END)
        token = invocation_budget.set(InvocationBudget(time.monotonic() + 10, 1))
        try:
            with self.assertRaises(BudgetExceeded):
                await builder.compile().ainvoke({})
            self.assertEqual(invocation_budget.get().calls, 1)
        finally:
            invocation_budget.reset(token)

    async def test_parser_text_and_archive_expansion_limits(self):
        import io, zipfile
        from app.services.document_pipeline import parse_blocks, check_archive
        with self.assertRaises(ValueError):
            await parse_blocks(b'a' * 101, '.txt', AsyncMock(), max_chars=100)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
            output.writestr('oversized.xml', b'a' * (32 * 1024 * 1024 + 1))
        with self.assertRaises(ValueError): check_archive(archive.getvalue())


class AuthIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dbname = 'career_security_test_' + uuid.uuid4().hex[:12]
        self.admin = await psycopg.AsyncConnection.connect(PSYCOPG_DATABASE_URL, autocommit=True)
        await self.admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(self.dbname)))
        self.engine = create_async_engine(make_url(settings.DATABASE_URL).set(database=self.dbname))
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.patches = [patch.object(auth, 'AsyncSessionLocal', self.factory),
                        patch.object(chat_api, 'AsyncSessionLocal', self.factory),
                        patch.object(chat_api, 'engine', self.engine),
                        patch.object(api, 'AsyncSessionLocal', self.factory)]
        for item in self.patches: item.start()
        self.app = FastAPI();
        self.app.add_middleware(SecurityLimitsMiddleware)
        for router in (auth_router, api.router, chat_api.router, evaluation_router): self.app.include_router(router,
                                                                                                             prefix='/api')

        async def db():
            async with self.factory() as session: yield session

        self.app.dependency_overrides[get_db] = db

        @self.app.exception_handler(RequestValidationError)
        async def validation(request, exc):
            return JSONResponse(
                {'detail': [{'loc': r['loc'], 'msg': r['msg'], 'type': r['type']} for r in exc.errors()]},
                status_code=422)

        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')
        self.accounts = []
        for username in ('alice', 'bob'):
            credentials = {'username': username, 'password': 'test-password-long'}
            response = await self.client.post('/api/auth/register', json=credentials)
            self.assertEqual(response.status_code, 201, response.text)
            response = await self.client.post('/api/auth/login', json=credentials)
            self.assertEqual(response.status_code, 200, response.text)
            self.accounts.append((uuid.UUID(response.json()['user']['id']),
                                  {'Authorization': 'Bearer ' + response.json()['access_token']}))
        self.alice, self.ah = self.accounts[0];
        self.bob, self.bh = self.accounts[1]

    async def asyncTearDown(self):
        await self.client.aclose()
        for item in reversed(self.patches): item.stop()
        await self.engine.dispose()
        await self.admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(self.dbname)))
        await self.admin.close()

    async def reset_counters(self):
        async with self.factory() as session: await session.execute(delete(RateLimitBucket));await session.commit()

    async def seed(self):
        async with self.factory() as session:
            da = Document(owner_id=self.alice, filename='alice.txt', document_type='knowledge', minio_object_key='a',
                          status='ready')
            db = Document(owner_id=self.bob, filename='bob.txt', document_type='resume', minio_object_key='b',
                          status='ready')
            run = AgentRun(owner_id=self.bob, task_type='match', status='paused', input_data={})
            ownrun = AgentRun(owner_id=self.alice, task_type='match', status='paused', input_data={})
            ca = Conversation(owner_id=self.alice, title='a', meta_data={});
            cb = Conversation(owner_id=self.bob, title='b', meta_data={})
            session.add_all([da, db, run, ownrun, ca, cb]);
            await session.commit()
            return da.id, db.id, run.id, ownrun.id, ca.id, cb.id

    async def test_login_register_logout_expiry(self):
        self.assertEqual((await self.client.get('/api/documents')).status_code, 401)
        response = await self.client.post('/api/auth/login',
                                          json={'username': 'alice', 'password': 'wrong-password-long'})
        self.assertEqual(response.status_code, 401)
        response = await self.client.post('/api/auth/register',
                                          json={'username': 'mallory', 'password': 'test-password-long',
                                                'is_admin': True})
        self.assertEqual(response.status_code, 422);
        self.assertNotIn('test-password-long', response.text)
        self.assertEqual((await self.client.get('/api/auth/me', headers=self.ah)).json()['username'], 'alice')
        self.assertEqual((await self.client.get('/api/evaluation/experiments', headers=self.ah)).status_code, 403)
        await self.client.post('/api/auth/logout', headers=self.ah)
        self.assertEqual((await self.client.get('/api/auth/me', headers=self.ah)).status_code, 401)
        async with self.factory() as session:
            for row in (await session.execute(select(AuthSession))).scalars(): row.expires_at = datetime.now(
                timezone.utc) - timedelta(seconds=1)
            await session.commit()
        self.assertEqual((await self.client.get('/api/auth/me', headers=self.bh)).status_code, 401)

    async def test_browser_cookie_login_restore_logout_and_csrf(self):
        endpoint = 'http://localhost:8000/api/auth'
        form = {'username': 'alice', 'password': 'test-password-long', 'return_to': 'http://localhost:8501/'}
        headers = {'Origin': 'http://localhost:8000'}
        with patch.object(settings, 'AUTH_COOKIE_SECURE', False):
            # Bad origin and redirect are rejected before issuing a session.
            bad = await self.client.post(endpoint + '/browser-login', data=form,
                                         headers={'Origin': 'http://attacker.invalid'})
            self.assertEqual(bad.status_code, 403)
            bad = await self.client.post(endpoint + '/browser-login',
                                         data={**form, 'return_to': 'https://attacker.invalid/'}, headers=headers)
            self.assertEqual(bad.status_code, 400)
            bad = await self.client.post(endpoint + '/browser-login',
                                         data={**form, 'return_to': 'http://127.0.0.1:8501/'}, headers=headers)
            self.assertEqual(bad.status_code, 400)
            bad = await self.client.post(endpoint + '/browser-login', data={**form, 'password': 'wrong-password-long'},
                                         headers=headers)
            self.assertEqual(bad.status_code, 401)
            self.assertNotIn('wrong-password-long', bad.text)
            self.assertNotIn('set-cookie', bad.headers)
            result = await self.client.post(endpoint + '/browser-login', data=form,
                                            headers={'Origin': 'http://localhost:8501', 'Accept': 'application/json'})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json(), {'logged_in': True})
            cookie = result.headers['set-cookie']
            self.assertIn('HttpOnly', cookie);
            self.assertIn('SameSite=lax', cookie)
            self.assertIn('Path=/', cookie);
            self.assertNotIn('Domain=', cookie)
            self.assertIn(f'Max-Age={settings.AUTH_SESSION_HOURS * 3600}', cookie)
            token = result.cookies[settings.AUTH_COOKIE_NAME]
            self.assertNotIn(token, result.text)
            self.assertEqual((await self.client.get(endpoint + '/me')).status_code,
                             401)  # Cookie cannot bypass Bearer auth.
            auth_header = {'Authorization': 'Bearer ' + token}
            self.assertEqual((await self.client.get(endpoint + '/me', headers=auth_header)).json()['username'], 'alice')
            # Simulates a new Streamlit connection restoring only the cookie token.
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url=endpoint) as restored:
                self.assertEqual((await restored.get(endpoint + '/me', headers=auth_header)).status_code, 200)
            rejected = await self.client.post(endpoint + '/browser-logout', data={'return_to': form['return_to']},
                                              headers={'Origin': 'https://attacker.invalid'})
            self.assertEqual(rejected.status_code, 403)
            result = await self.client.post(endpoint + '/browser-logout', data={'return_to': form['return_to']},
                                            headers={'Origin': 'http://localhost:8501', 'Accept': 'application/json'})
            self.assertEqual(result.json(), {'cleared': True})
            self.assertIn('Max-Age=0', result.headers['set-cookie'])
            self.assertEqual((await self.client.get(endpoint + '/me', headers=auth_header)).status_code, 401)
            # Revoked/expired sessions still allow clearing their cookie.
            result = await self.client.post(endpoint + '/browser-logout', data={'return_to': form['return_to']},
                                            headers=headers)
            self.assertEqual(result.status_code, 200)

    async def test_browser_cookie_secure_over_https(self):
        with patch.object(settings, 'CORS_ORIGINS', 'https://career.example'), patch.object(settings,
                                                                                            'AUTH_COOKIE_SECURE', True):
            result = await self.client.post('https://career.example/api/auth/browser-login',
                                            data={'username': 'alice', 'password': 'test-password-long',
                                                  'return_to': 'https://career.example/'},
                                            headers={'Origin': 'https://career.example', 'Accept': 'application/json'})
            self.assertEqual(result.status_code, 200)
            self.assertIn('Secure', result.headers['set-cookie'])

    async def test_foreign_resources_denied(self):
        da, db, run, ownrun, ca, cb = await self.seed()
        self.assertEqual({r['id'] for r in (await self.client.get('/api/documents', headers=self.ah)).json()},
                         {str(da)})
        self.assertEqual({r['run_id'] for r in (await self.client.get('/api/runs', headers=self.ah)).json()},
                         {str(ownrun)})
        self.assertEqual({r['id'] for r in (await self.client.get('/api/chat/conversations', headers=self.ah)).json()},
                         {str(ca)})
        cases = [('GET', f'/api/runs/{run}', None), ('DELETE', f'/api/documents/{db}', None),
                 ('GET', f'/api/chat/conversations/{cb}/messages', None),
                 ('PATCH', f'/api/chat/conversations/{cb}?title=stolen', None),
                 ('DELETE', f'/api/chat/conversations/{cb}', None),
                 ('POST', '/api/chat/completions', {'conversation_id': str(cb), 'message': 'hi', 'stream': False}),
                 ('POST', '/api/match', {'resume_document_id': str(db), 'jd_text': 'python'}),
                 ('POST', '/api/match/resume', {'thread_id': str(run)}),
                 ('POST', '/api/match/stream/resume', {'thread_id': str(run)}),
                 ('POST', '/api/interview', {'match_run_id': str(run)}),
                 ('POST', '/api/learning-plan', {'match_run_id': str(run)})]
        for method, path, body in cases:
            response = await self.client.request(method, path, headers=self.ah, json=body)
            self.assertEqual(response.status_code, 404, (path, response.text));
            await self.reset_counters()
        response = await self.client.post('/api/match/resume', headers=self.ah,
                                          json={'thread_id': str(run), 'run_id': str(ownrun)})
        self.assertEqual(response.status_code, 400)
        async with self.factory() as session: self.assertEqual((await session.get(AgentRun, ownrun)).status, 'paused')

    async def test_search_ownership_filters(self):
        da, db, *_ = await self.seed()
        mine = {'document_id': str(da), 'document_type': 'knowledge', 'content': 'mine', 'score': 1, 'source': 'dense'}
        theirs = {'document_id': str(db), 'document_type': 'resume', 'content': 'secret', 'score': 1, 'source': 'dense'}
        for path, service in [('/api/search', api.milvus_service), ('/api/search/bm25', api.bm25_service),
                              ('/api/search/hybrid', api.hybrid_search_service)]:
            search = AsyncMock(return_value=[mine, theirs])
            with patch.object(service, 'search', search), patch.object(api.bailian_service, 'embed_query',
                                                                       AsyncMock(return_value=[0])):
                response = await self.client.post(path, headers=self.ah, json={'query': 'python'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual({r['document_id'] for r in response.json()}, {str(da)})
            self.assertEqual(search.call_args.kwargs['allowed_document_ids'], [str(da)]);
            self.assertNotIn('secret', response.text)

    async def test_inputs_and_body_caps(self):
        for body in [{'message': 'a' * (settings.CHAT_MAX_INPUT_CHARS + 1)},
                     {'message': 'hi', 'user_id': str(self.bob)}, {'message': ' '}]:
            response = await self.client.post('/api/chat/completions', headers=self.ah, json=body);
            self.assertEqual(response.status_code, 422, response.text)
        response = await self.client.post('/api/chat/completions', headers=self.ah,
                                          content=b'x' * (settings.JSON_BODY_MAX_BYTES + 1))
        self.assertEqual(response.status_code, 413)
        self.assertEqual((await self.client.get('/api/runs?limit=100000', headers=self.ah)).status_code, 422)

    async def test_atomic_rate_limit_concurrent(self):
        with patch.object(settings, 'GENERATION_REQUESTS_PER_MINUTE', 2):
            responses = await asyncio.gather(
                *[self.client.post('/api/search/bm25', headers=self.ah, json={'query': 'q'}) for _ in range(5)])
        self.assertEqual(sorted(r.status_code for r in responses), [200, 200, 429, 429, 429])
        self.assertTrue(all('retry-after' in r.headers for r in responses if r.status_code == 429))

    async def test_conversation_lock_and_operation_limit(self):
        *_, ca, cb = await self.seed();
        started = asyncio.Event();
        finish = asyncio.Event()

        async def chat(**kwargs): started.set();await finish.wait();return 'answer'

        with patch.object(chat_api.bailian_service, 'chat', chat):
            task = asyncio.create_task(self.client.post('/api/chat/completions', headers=self.ah,
                                                        json={'conversation_id': str(ca), 'message': 'q',
                                                              'stream': False}))
            await asyncio.wait_for(started.wait(), 2)
            response = await self.client.post('/api/chat/completions', headers=self.ah,
                                              json={'conversation_id': str(ca), 'message': 'q2', 'stream': False})
            self.assertEqual(response.status_code, 409, response.text)
            with patch.object(settings, 'MAX_CONCURRENT_OPERATIONS', 1):
                response = await self.client.post('/api/search/bm25', headers=self.ah, json={'query': 'q'})
            self.assertEqual(response.status_code, 429)
            finish.set();
            self.assertEqual((await task).status_code, 200)
        async with self.engine.connect() as connection:
            key = 'chat:' + str(ca)
            self.assertTrue((await connection.execute(text('SELECT pg_try_advisory_lock(hashtextextended(:k,0))'),
                                                      {'k': key})).scalar())
            await connection.execute(text('SELECT pg_advisory_unlock(hashtextextended(:k,0))'), {'k': key})

    async def test_timeout_cleans_stream_and_leases(self):
        stopped = asyncio.Event()

        async def stream(**kwargs):
            try:
                yield 'partial';await asyncio.sleep(10)
            finally:
                stopped.set()

        with patch.object(settings, 'CHAT_TOTAL_TIMEOUT', .25), patch.object(chat_api.bailian_service,
                                                                             'stream_messages', stream):
            response = await self.client.post('/api/chat/completions', headers=self.ah, json={'message': 'q'})
        self.assertEqual(response.status_code, 200);
        self.assertIn('总耗时', response.text);
        self.assertIn('[DONE]', response.text);
        self.assertTrue(stopped.is_set())
        await asyncio.sleep(.05)
        async with self.factory() as session:
            self.assertEqual((await session.execute(select(func.count()).select_from(OperationLease))).scalar(), 0)
            rows = (await session.execute(select(Message).where(Message.role == 'assistant'))).scalars().all()
            self.assertTrue(rows);
            self.assertTrue(rows[0].meta_data['incomplete'])

    async def test_budget_error_reaches_sse_client(self):
        async def stream(**kwargs): yield 'partial';raise BudgetExceeded('已达到调用次数上限')

        with patch.object(chat_api.bailian_service, 'stream_messages', stream):
            response = await self.client.post('/api/chat/completions', headers=self.ah, json={'message': 'q'})
        self.assertEqual(response.status_code, 200);
        self.assertIn('调用次数', response.text);
        self.assertIn('[DONE]', response.text)

    async def test_upload_deduplication_is_per_owner(self):
        import app.services.document_service as service_module
        module = service_module

        async def upload(**kwargs): return 'documents/' + kwargs['document_id'] + '/test.txt'

        with patch.object(module.minio_service, 'upload', upload), \
                patch.object(module.bailian_service, 'embed',
                             AsyncMock(return_value=[[0.0] * settings.EMBEDDING_DIMENSION])), \
                patch.object(module.milvus_service, 'insert', AsyncMock(return_value=1)), \
                patch.object(module.bm25_service, 'add_document', AsyncMock()):
            results = []
            for headers in (self.ah, self.ah, self.bh):
                response = await self.client.post('/api/documents/upload', headers=headers,
                                                  files={'file': ('test.txt', b'Python experience', 'text/plain')},
                                                  data={'document_type': 'resume'})
                self.assertEqual(response.status_code, 200, response.text);
                results.append(response.json())
        self.assertEqual(results[0]['id'], results[1]['id'])
        self.assertNotEqual(results[0]['id'], results[2]['id'])
        async with self.factory() as session:
            rows = (await session.execute(select(Document))).scalars().all()
            self.assertEqual({row.owner_id for row in rows}, {self.alice, self.bob})

    async def test_resume_uses_owned_thread_and_rejects_replay(self):
        *_, ownrun, ca, cb = await self.seed()
        report = SimpleNamespace(model_dump=lambda **kwargs: {'total_score': 100})
        resume = AsyncMock(return_value=report)
        with patch.object(api, 'resume_match', resume):
            response = await self.client.post('/api/match/resume', headers=self.ah, json={'thread_id': str(ownrun)})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(resume.call_args.kwargs['thread_id'], str(ownrun))
            response = await self.client.post('/api/match/resume', headers=self.ah, json={'thread_id': str(ownrun)})
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(resume.await_count, 1)

    async def test_workflow_stream_timeout_stops_and_marks_run_failed(self):
        stopped = asyncio.Event()
        async with self.factory() as session:
            document = Document(owner_id=self.alice, filename='resume.txt', document_type='resume', status='ready',
                                minio_object_key='local')
            session.add(document);
            await session.flush();
            identifier = document.id
            session.add(
                DocumentParse(document_id=identifier, pipeline_version='test', text='Python developer', blocks=[],
                              chunks=[]))
            await session.commit()

        async def workflow(**kwargs):
            try:
                self.assertEqual(invocation_budget.get().model_timeout, settings.MATCH_MODEL_TIMEOUT)
                yield 'event: node_update\ndata: {"message":"started"}\n\n'
                await asyncio.sleep(10)
            finally:
                stopped.set()

        with patch.object(settings, 'MATCH_TOTAL_TIMEOUT', 2), patch.object(api, 'run_match_stream', workflow):
            response = await self.client.post('/api/match/stream', headers=self.ah,
                                              json={'resume_document_id': str(identifier), 'jd_text': 'Python'})
        self.assertEqual(response.status_code, 200, response.text);
        self.assertIn('总耗时', response.text)
        self.assertTrue(stopped.is_set())
        async with self.factory() as session:
            runs = (await session.execute(select(AgentRun))).scalars().all()
            self.assertEqual(len(runs), 1);
            self.assertEqual(runs[0].status, 'failed')
            self.assertEqual((await session.execute(select(func.count()).select_from(OperationLease))).scalar(), 0)


if __name__ == '__main__': unittest.main()
