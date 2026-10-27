const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const path=require('node:path');const vm=require('node:vm');const ts=require('typescript');
function runtime(){
 const storage=new Map(), calls=[], cache=new Map();let respond=()=>({statusCode:200,data:{}}),env='develop';
 const wx={getStorageSync:k=>storage.get(k),setStorageSync:(k,v)=>storage.set(k,v),removeStorageSync:k=>storage.delete(k),getAccountInfoSync:()=>({miniProgram:{envVersion:env}}),reLaunch:options=>calls.push({redirect:options.url}),request:options=>{calls.push(options);queueMicrotask(()=>{const value=respond(options);value.fail?options.fail({}):options.success(value)});return {abort(){}}}};
 function load(name){let p=path.resolve(__dirname,'../miniprogram',name);if(!p.endsWith('.ts'))p+='.ts';if(cache.has(p))return cache.get(p).exports;const m={exports:{}};cache.set(p,m);const code=ts.transpileModule(fs.readFileSync(p,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;const context={module:m,exports:m.exports,require:n=>load(path.relative(path.resolve(__dirname,'../miniprogram'),path.resolve(path.dirname(p),n))),wx,getCurrentPages:()=>[{route:'pages/chat/index'}],console,Date,Math,Error,Number,ArrayBuffer,Uint8Array,Promise,setTimeout,clearTimeout,Page:definition=>context.page=definition};vm.runInNewContext(code,context,{filename:p});return context.page||m.exports}
 const auth={access_token:'synthetic-session-token-'.padEnd(43,'x'),expires_at:new Date(Date.now()+3600000).toISOString(),user:{id:'owner-a',username:'alice'}};
 return {wx,storage,calls,load,auth,respond:fn=>respond=fn,environment:value=>env=value};
}
test('UTF-8 and SSE survive every split point, including Chinese, emoji and CRLF',()=>{
 const {SseDecoder}=runtime().load('services/stream');const text='data: '+JSON.stringify({choices:[{delta:{content:'你好🙂，简面职说'}}]})+'\r\n\r\ndata: [DONE]\n\n';const bytes=Buffer.from(text);
 for(let split=0;split<=bytes.length;split++){const decoder=new SseDecoder();const frames=[...decoder.feed(Uint8Array.from(bytes.subarray(0,split)).buffer),...decoder.feed(Uint8Array.from(bytes.subarray(split)).buffer)];assert.equal(frames.length,2);assert.equal(JSON.parse(frames[0]).choices[0].delta.content,'你好🙂，简面职说');assert.equal(frames[1],'[DONE]')}
});
test('expired or malformed sessions cannot authorize a request',async()=>{
 const r=runtime(),session=r.load('services/session');session.saveSession({...r.auth,expires_at:'bad-date'});assert.equal(session.session(),null);await assert.rejects(r.load('services/api').request('/documents'),{status:401});assert.equal(r.calls.length,0)
});
test('authenticated requests attach bearer token; 401 clears the local session',async()=>{
 const r=runtime();const session=r.load('services/session');session.saveSession(r.auth);r.respond(()=>({statusCode:401,data:{detail:'登录已过期'}}));await assert.rejects(r.load('services/api').request('/credits/me'),{status:401});assert.equal(r.calls[0].header.Authorization,'Bearer '+r.auth.access_token);assert.equal(session.session(),null);assert.equal(r.calls[1].redirect,'/pages/login/index')
});
test('ambiguous paid submission reuses the idempotency id across page reloads',async()=>{
 const r=runtime();r.load('services/session').saveSession(r.auth);r.respond(()=>({fail:true}));const jobs=r.load('services/jobs');await assert.rejects(jobs.submitJob('match',{jd_text:'same'},'first-id'));r.respond(()=>({statusCode:202,data:{job_id:'first-id',status:'queued'}}));const result=await jobs.submitJob('match',{jd_text:'same'},'second-id');assert.equal(result.job_id,'first-id');assert.equal(r.calls[1].data.client_request_id,'first-id');assert.equal(r.calls.length,2)
});
test('production cannot silently use the development tunnel',()=>{const r=runtime();r.environment('release');assert.throws(()=>r.load('config').apiBase(),/配置小程序服务地址/)});
test('quota page maps the real server shape and formats fractional points',async()=>{
 const r=runtime();r.load('services/session').saveSession(r.auth);r.respond(options=>({statusCode:200,data:options.url.endsWith('/credits/me')?{available:68.686,checked_in_today:true,rules:{daily_checkin_points:10}}:options.url.endsWith('/auth/usage')?{is_admin:false,generation_daily:{used:3,limit:200},generation_minute:{used:1,limit:10}}:{items:[],has_more:false}}));const page=r.load('pages/account/index');page.data=structuredClone(page.data);page.setData=patch=>Object.assign(page.data,patch);await page.refresh();assert.equal(page.data.available,'68.686');assert.equal(page.data.usage.daily_used,3);assert.equal(page.data.usage.daily_limit,200);assert.equal(page.data.usage.minute_used,1)
});
test('rendered model text cannot inject executable HTML or remote images',()=>{const {markdown}=runtime().load('services/markdown');const html=markdown('# 标题\n**优势**\n<img src="https://attacker.example/x">\n[链接](javascript:alert(1))\n```\n<script>bad()</script>\n```');assert.ok(html.includes('<strong>优势</strong>'));assert.ok(!html.includes('<img'));assert.ok(!html.includes('<script'));assert.ok(!html.includes('href='));assert.ok(html.includes('&lt;script&gt;'))});
