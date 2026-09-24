"""Render real Streamlit forms with fake HTTP; verify no private load before login."""
import unittest
from pathlib import Path
import json
from unittest.mock import PropertyMock, patch
from streamlit.testing.v1 import AppTest
import requests
import streamlit as st

calls=[]
options={'registration_enabled':True,'chat_max_input_chars':4000,'chat_max_output_tokens':2048,'chat_total_timeout':90,
         'analysis_total_timeout':180,'max_upload_size_mb':20,'search_max_input_chars':2000,'jd_max_input_chars':30000,'feedback_max_input_chars':2000}
def response(value,status=200):
 result=requests.Response();result.status_code=status;result._content=json.dumps(value).encode();return result
def api_request(method,url,**kwargs):
 calls.append((method,url,kwargs.get('headers',{})))
 if url.endswith('/api/auth/options'):return response(options)
 if url.endswith('/api/auth/login'):return response({'access_token':'test-session-token','user':{'username':'alice','is_admin':False}})
 if url.endswith('/api/auth/me'):return response({'username':'alice','is_admin':False})
 if url.endswith('/api/auth/logout'):return response({'message':'bye'})
 return response([])
class FrontendAuthTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()  # AppTest creates its own component registry.

    def test_login_guard_and_logout_clear_private_ui(self):
        calls.clear()
        cookies={}
        token='test-session-token'+'x'*32
        with patch('requests.request',side_effect=api_request), \
             patch.object(type(st.context),'cookies',new_callable=PropertyMock,return_value=cookies), \
             patch('streamlit.components.v2.component',return_value=lambda **kwargs: None):
            path=Path(__file__).resolve().parents[2]/'frontend'/'app.py'
            app=AppTest.from_file(str(path),default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertEqual([x.label for x in app.tabs],['登录','注册'])
            self.assertTrue(all('/api/auth/' in url for _,url,_ in calls))
            cookies['career_session']=token
            # A fresh browser connection has no session_state, only the cookie.
            app=AppTest.from_file(str(path),default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.tabs),0)
            self.assertEqual(len(app.radio),0)
            self.assertEqual(app.session_state['active_page'],'chat')
            app.button(key='nav_knowledge').click().run()
            self.assertTrue(any('/api/documents' in url and headers.get('Authorization')=='Bearer '+token for _,url,headers in calls))
            app.session_state['match_report']={'private':'alice report'}
            next(button for button in app.button if button.label=='退出登录').click().run();app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state['clear_browser_session'])
            self.assertNotIn('access_token',app.session_state.filtered_state)
            self.assertNotIn('match_report',app.session_state.filtered_state)
            self.assertFalse(app.chat_message)
            self.assertFalse(any('alice' in str(row.value) for row in app.sidebar.markdown))
            cookies.clear()  # Browser component clears Cookie and reloads.
            app=AppTest.from_file(str(path),default_timeout=15).run()
            self.assertEqual([x.label for x in app.tabs],['登录','注册'])

    def test_invalid_cookie_cannot_render_private_content(self):
        calls.clear()
        def invalid_me(method,url,**kwargs):
            if url.endswith('/api/auth/me'):return response({'detail':'expired'},401)
            return api_request(method,url,**kwargs)
        with patch('requests.request',side_effect=invalid_me), \
             patch.object(type(st.context),'cookies',new_callable=PropertyMock,return_value={'career_session':'x'*43}), \
             patch('streamlit.components.v2.component',return_value=lambda **kwargs: None):
            path=Path(__file__).resolve().parents[2]/'frontend'/'app.py'
            app=AppTest.from_file(str(path),default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state['clear_browser_session'])
            self.assertNotIn('access_token',app.session_state.filtered_state)
            self.assertTrue(all('/api/auth/' in url for _,url,_ in calls))

    def test_match_progress_and_structured_report_are_rendered_before_review(self):
        report={'total_score':78,'match_level':'较高匹配','scores':{key:80 for key in ['skill','experience','responsibility','education','bonus']},
                'matched_skills':['Python'],'missing_skills':['Testing'],'strengths':['Synthetic evidence'],
                'risks':['Synthetic gap'],'suggestions':['Add synthetic tests'],
                'resume_analysis':{'skills':['Python']},'job_analysis':{'job_title':'Synthetic developer'}}
        document={'id':'synthetic-resume','filename':'synthetic.txt','document_type':'resume','status':'ready','chunk_count':1}
        def http_request(method,url,**kwargs):
            if url.endswith('/api/documents'):return response([document])
            return api_request(method,url,**kwargs)
        events=[('init',{'run_id':'synthetic-run'}),('start',{}),
                ('node_update',{'node':'resume_agent','status':'completed','message':'简历分析完成（复用已有结果）','cache_hit':True}),
                ('node_update',{'node':'jd_agent','status':'completed','message':'岗位分析完成（复用已有结果）','cache_hit':True}),
                ('progress',{'node':'match_agent','message':'正在生成匹配报告…'}),
                ('heartbeat',{'elapsed_seconds':5}),
                ('node_update',{'node':'match_agent','status':'completed','score':78}),
                ('interrupt',{'thread_id':'synthetic-run','current_report':report})]
        stream=response({})
        stream.close=lambda: None
        stream.iter_lines=lambda **kwargs: iter(line for kind,data in events for line in
            [f'event: {kind}','data: '+json.dumps(data),''])
        with patch('requests.request',side_effect=http_request),patch('requests.post',return_value=stream), \
             patch.object(type(st.context),'cookies',new_callable=PropertyMock,return_value={'career_session':'x'*43}), \
             patch('streamlit.components.v2.component',return_value=lambda **kwargs: None), \
             patch('streamlit.progress',wraps=st.progress) as progress:
            path=Path(__file__).resolve().parents[2]/'frontend'/'app.py'
            app=AppTest.from_file(str(path),default_timeout=15).run()
            app.button(key='nav_match').click().run()
            next(field for field in app.text_area if field.label=='输入岗位描述 (JD)').set_value('Synthetic JD')
            next(button for button in app.button if button.label=='🚀 开始岗位匹配').click().run()
            self.assertFalse(app.exception)
            progress.assert_called()
            self.assertTrue(app.session_state['match_paused'])
            self.assertEqual(app.session_state['preliminary_report']['total_score'],78)
            self.assertTrue(any(metric.label=='综合加权匹配分' for metric in app.metric))
            self.assertTrue(any('Add synthetic tests' in row.value for row in app.markdown))
            self.assertFalse(any('思考中' in row.value for row in app.markdown))
