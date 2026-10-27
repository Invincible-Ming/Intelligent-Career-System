const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm'), ts=require('typescript');
const source=ts.transpileModule(fs.readFileSync(path.resolve(__dirname,'../miniprogram/pages/chat/index.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;
function deferred(){let resolve,reject;const promise=new Promise((ok,no)=>{resolve=ok;reject=no});return{promise,resolve,reject}}
function fixture(handle=()=>[], start=()=>Promise.resolve()){
 let page;const calls=[],streams=[],errors=[];
 const request=async(url,...args)=>{calls.push(url);if(url==='/auth/options')return{chat_max_input_chars:4000};if(url==='/chat/conversations'&&!args.length)return[];return handle(url,...args)};
 const wx={nextTick:fn=>fn()};
 vm.runInNewContext(source,{exports:{},Page:p=>page=p,wx,console,require:name=>name.endsWith('/api')?{request,errorMessage:e=>e.message,notifyError:e=>errors.push(e)}:name.endsWith('/session')?{requireSession:()=>true,session:()=>({user:{username:'alice'}})}:name.endsWith('/markdown')?{markdown:s=>s}:name.endsWith('/stream')?{streamChat:payload=>{const stream={payload,done:null,stopped:false,stop(){this.stopped=true}};streams.push(stream);stream.done=start(stream);return stream}}:{}});
 page.data=structuredClone(page.data);
 page.setData=patch=>{for(const [key,value]of Object.entries(patch)){const match=/^messages\[(\d+)\]\.(.+)$/.exec(key);if(match)page.data.messages[Number(match[1])][match[2]]=value;else page.data[key]=value}};
 return{page,calls,streams,errors,async show(){page.onLoad();await page.onShow()}};
}
const touch=id=>({currentTarget:{dataset:{id}}});

test('hide then show during conversation creation cannot revive generation; unsent question is restored',async()=>{
 const create=deferred(),r=fixture(()=>create.promise);await r.show();r.page.data.draft='我的简历怎么写？';const sending=r.page.send();
 r.page.onHide();await r.page.onShow();create.resolve({id:'late-created'});await sending;
 assert.equal(r.streams.length,0);assert.equal(r.page.data.sending,false);assert.equal(r.page.data.draft,'我的简历怎么写？');assert.equal(r.page.data.conversationId,'');
});
test('explicit stop or unload before creation finishes prevents any model request',async()=>{
 for(const action of ['stop','onUnload']){const create=deferred(),r=fixture(()=>create.promise);await r.show();r.page.data.draft='test';const sending=r.page.send();r.page[action]();create.resolve({id:'late'});await sending;assert.equal(r.streams.length,0)}
});
test('latest-first pagination and post-generation reload keep new messages and correct older-page offset',async()=>{
 const history=Array.from({length:300},(_,i)=>({id:String(i),role:i%2?'assistant':'user',content:'message-'+i}));
 const r=fixture(url=>{const u=new URL(url,'https://synthetic.example');const offset=Number(u.searchParams.get('offset')),limit=Number(u.searchParams.get('limit'));assert.ok(url.includes('/messages?'));return history.slice(Math.max(0,history.length-offset-limit),history.length-offset)},()=>{history.push({id:'300',role:'user',content:'new question'},{id:'301',role:'assistant',content:'new answer'});return Promise.resolve()});
 await r.show();await r.page.openConversation(touch('long-conversation'));
 assert.equal(r.page.data.messages[0].id,'250');assert.equal(r.page.data.messages.at(-1).id,'299');assert.equal(r.page.data.historyOffset,50);
 await r.page.loadMore();assert.equal(r.page.data.messages[0].id,'200');assert.equal(r.page.data.historyOffset,100);
 r.page.data.draft='new question';await r.page.send();
 assert.equal(r.page.data.messages[0].id,'252');assert.equal(r.page.data.messages.at(-1).content,'new answer');assert.equal(r.page.data.historyOffset,50);
 await r.page.loadMore();assert.equal(r.page.data.messages[0].id,'202');assert.equal(r.page.data.messages.at(-1).id,'301');assert.equal(r.page.data.historyOffset,100);
 assert.equal(new Set(r.page.data.messages.map(x=>x.id)).size,100);assert.ok(r.calls.every(url=>!url.includes('limit=200')));assert.deepEqual(r.errors,[]);
});
test('exactly one page ends cleanly when the next older page is empty',async()=>{
 const messages=Array.from({length:50},(_,i)=>({id:String(i),role:'user',content:String(i)}));const r=fixture(url=>url.includes('offset=0')?messages:[]);await r.show();await r.page.openConversation(touch('id'));await r.page.loadMore();assert.equal(r.page.data.messages.length,50);assert.equal(r.page.data.canLoadMore,false);assert.equal(r.page.data.historyOffset,50);
});
test('a delayed history response cannot overwrite a newly created empty chat',async()=>{
 const history=deferred(),r=fixture(()=>history.promise);await r.show();const loading=r.page.openConversation(touch('old'));r.page.newChat();history.resolve([{id:'old-msg',role:'user',content:'old'}]);await loading;assert.equal(r.page.data.messages.length,0);assert.equal(r.page.data.conversationId,'');assert.equal(r.page.data.loadingHistory,false);
});
test('hiding an active stream aborts it and returning reloads persisted messages',async()=>{
 const done=deferred();const r=fixture(()=>[{id:'persisted',role:'assistant',content:'partial',incomplete:true}],()=>done.promise);await r.show();r.page.data.conversationId='id';r.page.data.draft='question';const sending=r.page.send();assert.equal(r.streams.length,1);r.page.onHide();assert.equal(r.streams[0].stopped,true);done.resolve();await sending;await r.page.onShow();assert.equal(r.page.data.messages[0].id,'persisted');assert.equal(r.page.data.sending,false);
});
