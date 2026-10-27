import { download, errorMessage, notifyError, request, upload } from '../../services/api'
import { requireSession, updateUser } from '../../services/session'
import { dateLabel, fileExtension } from '../../services/presentation'
import { DocumentItem, Options, SearchResult, User } from '../../services/types'
const names:Record<string,string>={resume:'简历',job_description:'岗位描述',knowledge:'知识资料',ready:'已就绪',processing:'处理中',failed:'处理失败',pending:'等待处理'}
interface FileRow extends DocumentItem { extension:string; timeText:string }
const categories=[{id:'all',label:'全部'},{id:'resume',label:'简历'},{id:'job_description',label:'岗位'},{id:'knowledge',label:'知识'}]
Page({
  _choosing:false,
  data:{documents:[] as FileRow[],visibleDocuments:[] as FileRow[],filters:categories.map(item=>({...item,count:0})),filter:'all',tab:'files',expandedId:'',busyFileId:'',busyAction:'',results:[] as SearchResult[],query:'',searched:false,loading:false,uploading:false,searching:false,error:'',provider:'ima',providerLabels:['ima + 本地关键词','混合检索 + 重排序'],providerIndex:0,canChoose:false,typeLabels:['知识资料','简历','岗位描述'],typeIndex:0,maxMb:20},
  async onShow(){if(!requireSession())return;await this.refresh()},
  async refresh(){
    this.setData({loading:true,error:''})
    try{
      const [docs,options,user]=await Promise.all([request<DocumentItem[]>('/documents'),request<Options>('/auth/options'),request<User>('/auth/me')])
      updateUser(user)
      const documents=docs.map(item=>({...item,label:names[item.document_type]||item.document_type,statusLabel:names[item.status]||item.status,extension:fileExtension(item.filename),timeText:dateLabel(item.created_at)}))
      this.setData({documents,visibleDocuments:this.filtered(documents,this.data.filter),filters:categories.map(item=>({...item,count:documents.filter(doc=>item.id==='all'||doc.document_type===item.id).length})),expandedId:documents.some(item=>item.id===this.data.expandedId)?this.data.expandedId:'',provider:user.knowledge_provider,providerIndex:user.knowledge_provider==='ima'?0:1,canChoose:options.knowledge_provider==='credits',maxMb:options.max_upload_size_mb})
    }catch(error){this.setData({error:errorMessage(error)})}
    finally{this.setData({loading:false})}
  },
  filtered(documents:FileRow[],filter:string):FileRow[]{return documents.filter(item=>filter==='all'||item.document_type===filter)},
  tabChange(e:WechatMiniprogram.TouchEvent){this.setData({tab:String(e.currentTarget.dataset.tab),error:''})},
  filterChange(e:WechatMiniprogram.TouchEvent){const filter=String(e.currentTarget.dataset.id);this.setData({filter,visibleDocuments:this.filtered(this.data.documents,filter),expandedId:''})},
  async providerChange(e:WechatMiniprogram.PickerChange){const index=Number(e.detail.value);if(index===this.data.providerIndex)return;try{await request('/knowledge/provider','PUT',{provider:index===0?'ima':'milvus'});await this.refresh();this.setData({results:[],searched:false})}catch(error){notifyError(error)}},
  input(e:WechatMiniprogram.Input){this.setData({query:e.detail.value})},
  async chooseUpload(){
    if(this.data.uploading||this._choosing)return
    this._choosing=true
    try{
      const choice=await wx.showActionSheet({itemList:this.data.typeLabels.map(label=>'上传'+label)}).catch(()=>null)
      if(!choice)return
      this.setData({typeIndex:choice.tapIndex})
      await this.chooseFile()
    }finally{this._choosing=false}
  },
  async chooseFile(){
    if(this.data.uploading)return
    const type=['knowledge','resume','job_description'][this.data.typeIndex]
    try{
      const choice=await wx.chooseMessageFile({count:1,type:'file',extension:['pdf','docx','txt','md','xlsx','xls','csv','png','jpg','jpeg','webp']}).catch(()=>null)
      if(!choice?.tempFiles.length)return
      const file=choice.tempFiles[0]
      if(file.size>this.data.maxMb*1024*1024){notifyError(new Error('文件不能超过 '+this.data.maxMb+' MB'));return}
      this.setData({uploading:true,error:''})
      const result=await upload(file.path,type)
      wx.showToast({title:result.deduplicated?'已复用相同文件':'文件上传成功',icon:'none'})
      this.setData({tab:'files',filter:'all'})
      await this.refresh()
    }catch(error){this.setData({error:errorMessage(error)})}finally{this.setData({uploading:false})}
  },
  async search(){if(!this.data.query.trim()||this.data.searching)return;this.setData({searching:true,error:'',results:[],searched:false});try{this.setData({results:await request<SearchResult[]>('/search/hybrid','POST',{query:this.data.query.trim(),top_k:5}),searched:true})}catch(error){this.setData({error:errorMessage(error)})}finally{this.setData({searching:false})}},
  preview(e:WechatMiniprogram.TouchEvent){if(!/^[0-9a-f-]{36}$/.test(String(e.currentTarget.dataset.id))){wx.showToast({title:'外部资料正文已显示在检索结果中',icon:'none'});return}wx.navigateTo({url:'/pages/document/index?id='+encodeURIComponent(String(e.currentTarget.dataset.id))})},
  manage(e:WechatMiniprogram.TouchEvent){const id=String(e.currentTarget.dataset.id);this.setData({expandedId:this.data.expandedId===id?'':id})},
  async fileAction(e:WechatMiniprogram.TouchEvent){
    if(this.data.busyFileId)return
    const id=String(e.currentTarget.dataset.id),action=String(e.currentTarget.dataset.action),document=this.data.documents.find(item=>item.id===id)
    if(!document)return
    this.setData({busyFileId:id,busyAction:action})
    try{
      if(action==='download'){await download(id);return}
      if(action==='rename'){
        const value=await wx.showModal({title:'重命名文件',editable:true,content:document.filename,placeholderText:'输入文件名称',confirmText:'保存'})
        if(!value.confirm||!value.content?.trim()||value.content.trim()===document.filename)return
        await request('/documents/'+id+'/rename','POST',{filename:value.content.trim()})
        wx.showToast({title:'名称已更新',icon:'success'})
      }else if(action==='delete'){
        const result=await wx.showModal({title:'删除这个文件？',content:'「'+document.filename+'」与检索分块将被移除，删除后无法恢复。',confirmText:'删除',confirmColor:'#b4453d'})
        if(!result.confirm)return
        await request('/documents/'+id,'DELETE')
        wx.showToast({title:'文件已删除',icon:'success'})
      }else if(action==='retry'){await request('/documents/'+id+'/retry','POST')}
      else return
      this.setData({expandedId:''})
      await this.refresh()
    }catch(error){notifyError(error)}finally{this.setData({busyFileId:'',busyAction:''})}
  },
})
