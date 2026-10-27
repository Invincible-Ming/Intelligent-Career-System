import { dateLabel } from '../../services/presentation'
import { errorMessage, notifyError, request, uuid } from '../../services/api'
import { requireSession, session } from '../../services/session'
import { submitJob } from '../../services/jobs'
import { DocumentItem, Options, Run } from '../../services/types'
const states:Record<string,string>={running:'生成中',paused:'待审核',completed:'已完成',failed:'未完成'}
Page({
  data:{resumes:[] as DocumentItem[],jobs:[] as DocumentItem[],runs:[] as Run[],resumeIndex:0,jobIndex:0,jdSource:0,jdText:'',research:false,loading:false,submitting:false,error:'',provider:'',maxChars:30000,pendingId:'',pendingPayload:''},
  async onShow(){if(!requireSession())return;await this.refresh()},
  async refresh(){this.setData({loading:true,error:''});try{const [docs,runs,options]=await Promise.all([request<DocumentItem[]>('/documents'),request<Run[]>('/runs?task_type=match&limit=30'),request<Options>('/auth/options')]);const ready=docs.filter(item=>item.status==='ready');this.setData({resumes:ready.filter(item=>item.document_type==='resume'),jobs:ready.filter(item=>item.document_type==='job_description'),runs:runs.map(run=>({...run,label:states[run.status]||run.status,timeText:dateLabel(run.created_at)})),provider:session()?.user.knowledge_provider==='ima'?'ima + 本地关键词':'混合检索 + BGE 重排序',maxChars:options.jd_max_input_chars})}catch(error){this.setData({error:errorMessage(error)})}finally{this.setData({loading:false})}},
  resumeChange(e:WechatMiniprogram.PickerChange){this.setData({resumeIndex:Number(e.detail.value)})},
  jobChange(e:WechatMiniprogram.PickerChange){this.setData({jobIndex:Number(e.detail.value)})},
  sourceChange(e:WechatMiniprogram.TouchEvent){this.setData({jdSource:Number(e.currentTarget.dataset.index)})},
  input(e:WechatMiniprogram.Input){this.setData({jdText:e.detail.value})},
  researchChange(e:WechatMiniprogram.SwitchChange){this.setData({research:e.detail.value})},
  upload(){wx.switchTab({url:'/pages/knowledge/index'})},
  async start(){
    if(this.data.submitting)return
    const resume=this.data.resumes[this.data.resumeIndex],job=this.data.jobs[this.data.jobIndex]
    if(!resume){notifyError(new Error('请先上传一份简历'));return}
    if(this.data.jdSource===0&&!this.data.jdText.trim()||this.data.jdSource===1&&!job){notifyError(new Error('请补充岗位描述'));return}
    const payload={resume_document_id:resume.id,research_approved:this.data.research,...(this.data.jdSource===0?{jd_text:this.data.jdText.trim()}:{jd_document_id:job.id})}
    const signature=JSON.stringify(payload)
    const id=this.data.pendingPayload===signature&&this.data.pendingId?this.data.pendingId:uuid()
    this.setData({submitting:true,error:'',pendingId:id,pendingPayload:signature})
    try{const task=await submitJob('match',payload,id);this.setData({pendingId:'',pendingPayload:''});wx.navigateTo({url:'/pages/task/index?jobId='+task.job_id})}catch(error){this.setData({error:errorMessage(error)})}finally{this.setData({submitting:false})}
  },
  openRun(e:WechatMiniprogram.TouchEvent){wx.navigateTo({url:'/pages/task/index?runId='+String(e.currentTarget.dataset.id)})},
})
