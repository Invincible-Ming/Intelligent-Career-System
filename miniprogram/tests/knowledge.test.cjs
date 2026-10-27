const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),ts=require('typescript');
function fixture(){
 const first='11111111-1111-4111-8111-111111111111',second='22222222-2222-4222-8222-222222222222';
 let documents=[{id:first,filename:'我的简历.pdf',document_type:'resume',status:'ready',chunk_count:6,created_at:'2026-10-27T01:00:00Z'},{id:second,filename:'岗位要求.md',document_type:'job_description',status:'failed',chunk_count:0}];
 const calls=[],modals=[],downloads=[],uploads=[],navigations=[],cache=new Map();let modal={confirm:false},choice=null;
 const wx={showToast(){},showActionSheet:async()=>{if(!choice)throw new Error('cancel');return choice},chooseMessageFile:async()=>({tempFiles:[{path:'synthetic.pdf',size:12}]}),showModal:async options=>{modals.push(options);return modal},navigateTo:options=>navigations.push(options.url)};
 const api={request:async(url,method='GET',body)=>{calls.push({url,method,body});if(url==='/documents')return structuredClone(documents);if(url==='/auth/options')return{knowledge_provider:'credits',max_upload_size_mb:50};if(url==='/auth/me')return{knowledge_provider:'milvus'};if(url==='/search/hybrid')return[];if(method==='DELETE')documents=documents.filter(doc=>url!=='/documents/'+doc.id);if(url.endsWith('/rename'))documents=documents.map(doc=>url==='/documents/'+doc.id+'/rename'?{...doc,filename:body.filename}:doc);return{}},download:async id=>downloads.push(id),upload:async(file,type)=>{uploads.push({file,type});return{deduplicated:false}},errorMessage:error=>error.message,notifyError:error=>{throw error}};
 function load(name){const filename=path.resolve(__dirname,'../miniprogram',name+'.ts');if(cache.has(filename))return cache.get(filename);let page;const module={exports:{}};const code=ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;vm.runInNewContext(code,{module,exports:module.exports,Page:p=>page=p,wx,Date,Number,console,require:imported=>imported.endsWith('/api')?api:imported.endsWith('/session')?{requireSession:()=>true,updateUser(){}}:load(path.relative(path.resolve(__dirname,'../miniprogram'),path.resolve(path.dirname(filename),imported)))});const result=page||module.exports;cache.set(filename,result);return result}
 const page=load('pages/knowledge/index');page.data=structuredClone(page.data);page.setData=patch=>Object.assign(page.data,patch);
 return{page,calls,modals,downloads,uploads,navigations,first,second,modal:value=>modal=value,choice:value=>choice=value};
}
const touch=(id,action)=>({currentTarget:{dataset:{id,action}}});
test('file types filter locally, counts are correct and search does not run during navigation',async()=>{
 const r=fixture();await r.page.refresh();assert.equal(r.page.data.visibleDocuments.length,2);assert.equal(r.page.data.documents[0].extension,'PDF');assert.equal(r.page.data.filters.find(x=>x.id==='knowledge').count,0);const requests=r.calls.length;r.page.filterChange(touch('resume'));assert.equal(r.page.data.visibleDocuments.length,1);assert.equal(r.page.data.visibleDocuments[0].id,r.first);r.page.tabChange({currentTarget:{dataset:{tab:'search'}}});assert.equal(r.calls.length,requests);assert.equal(r.page.data.tab,'search');
});
test('management expands the selected card and download is a direct labelled action',async()=>{
 const r=fixture();await r.page.refresh();r.page.manage(touch(r.first));assert.equal(r.page.data.expandedId,r.first);r.page.manage(touch(r.second));assert.equal(r.page.data.expandedId,r.second);await r.page.fileAction(touch(r.first,'download'));assert.deepEqual(r.downloads,[r.first]);assert.equal(r.modals.length,0);r.page.manage(touch(r.second));assert.equal(r.page.data.expandedId,'');
});
test('deleting requires confirmation and removes only the chosen synthetic file',async()=>{
 const r=fixture();await r.page.refresh();await r.page.fileAction(touch(r.first,'delete'));assert.equal(r.calls.filter(x=>x.method==='DELETE').length,0);assert.equal(r.page.data.busyFileId,'');r.modal({confirm:true});await r.page.fileAction(touch(r.first,'delete'));assert.equal(r.calls.filter(x=>x.method==='DELETE')[0].url,'/documents/'+r.first);assert.equal(r.page.data.documents.length,1);assert.equal(r.page.data.documents[0].id,r.second);
});
test('rename starts with the existing filename and refreshes the filtered list',async()=>{
 const r=fixture();await r.page.refresh();r.page.filterChange(touch('resume'));r.modal({confirm:true,content:'新版简历.pdf'});await r.page.fileAction(touch(r.first,'rename'));assert.equal(r.modals[0].content,'我的简历.pdf');assert.equal(r.page.data.visibleDocuments[0].filename,'新版简历.pdf');assert.equal(r.calls.find(x=>x.url.endsWith('/rename')).method,'POST');
});
test('cancel upload type selection has no effect; choosing resume sends the correct document type',async()=>{
 const r=fixture();await r.page.chooseUpload();assert.equal(r.uploads.length,0);r.choice({tapIndex:1});await r.page.chooseUpload();assert.equal(r.uploads.length,1);assert.equal(r.uploads[0].type,'resume');assert.equal(r.page.data.tab,'files');assert.equal(r.page.data.filter,'all');
});
