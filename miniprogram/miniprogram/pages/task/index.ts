import { errorMessage, notifyError, request, uuid } from '../../services/api'
import { requireSession } from '../../services/session'
import { submitJob } from '../../services/jobs'
import { Job, MatchReport, Run } from '../../services/types'
interface Question {question:string;purpose:string;answer_points:string[]}
interface LearningItem {skill:string;priority:string;reason:string;estimated_days:number;tasks:string[]}
Page({
  _timer:undefined as ReturnType<typeof setTimeout>|undefined, _visible:false,
  data:{jobId:'',runId:'',kind:'match',status:'running',message:'正在载入任务',error:'',report:null as MatchReport|null,scoreRows:[] as {label:string;value:number}[],feedback:'',submitting:false,elapsed:0,started:Date.now(),questions:[] as Question[],focusAreas:[] as string[],learningItems:[] as LearningItem[],weeklyPlan:[] as string[],planTitle:'',planSummary:'',requestId:''},
  onLoad(query:Record<string,string|undefined>){if(!requireSession())return;this.setData({jobId:query.jobId||'',runId:query.runId||'',started:Date.now()})},
  onShow(){this._visible=true;if(requireSession())void this.refresh()},
  onHide(){this._visible=false;if(this._timer)clearTimeout(this._timer)},
  onUnload(){this._visible=false;if(this._timer)clearTimeout(this._timer)},
  async refresh(){
    if(this._timer)clearTimeout(this._timer)
    if(!this._visible)return
    try{
      if(this.data.jobId){
        const task=await request<Job>('/mobile/jobs/'+encodeURIComponent(this.data.jobId))
        this.setData({status:task.status,message:task.message,error:task.error,kind:task.kind,runId:task.run_id||this.data.runId,elapsed:Math.floor((Date.now()-this.data.started)/1000)})
        if(task.result){const result=task.result;const report=(result.current_report||result.report) as MatchReport|undefined;if(report)this.showReport(report);if(task.kind==='interview'||task.kind==='learning')this.showPlan((result.result||result) as Record<string,unknown>)}
        if(task.status==='completed'||task.status==='paused'){if(this.data.runId)await this.loadRun();return}
        if(task.status==='failed'||task.status==='cancelled')return
      }else if(this.data.runId){await this.loadRun();if(this.data.status!=='running')return}
      else{this.setData({status:'failed',error:'缺少任务编号'});return}
      if(this._visible)this._timer=setTimeout(()=>void this.refresh(),3000)
    }catch(error){if(this._visible)this.setData({error:errorMessage(error),message:'进度连接中断，可手动刷新；任务不会因此重新执行'})}
  },
  async loadRun(){const run=await request<Run>('/runs/'+encodeURIComponent(this.data.runId));this.setData({status:run.status,kind:run.task_type,error:run.error_message||''});if(run.result){if(run.task_type==='match'){const report=(run.result.current_report||run.result.report||run.result) as unknown as MatchReport;this.showReport(report)}else this.showPlan(run.result)}},
  showReport(report:MatchReport){const names:Record<string,string>={skill:'技能匹配',experience:'经历匹配',responsibility:'职责匹配',education:'学历要求',bonus:'加分项'};this.setData({report,scoreRows:Object.keys(names).map(key=>({label:names[key],value:report.scores?.[key]||0}))})},
  showPlan(result:Record<string,unknown>){this.setData({planTitle:String(result.job_title||result.target_role||''),planSummary:String(result.summary||''),questions:(result.questions||[]) as Question[],focusAreas:(result.focus_areas||[]) as string[],learningItems:(result.items||[]) as LearningItem[],weeklyPlan:(result.weekly_plan||[]) as string[]})},
  input(e:WechatMiniprogram.Input){this.setData({feedback:e.detail.value,requestId:''})},
  async resume(){
    if(this.data.submitting)return
    const id=this.data.requestId||uuid();this.setData({submitting:true,error:'',requestId:id})
    try{const job=await submitJob('resume',{thread_id:this.data.runId,run_id:this.data.runId,human_feedback:this.data.feedback.trim()||null},id);this.setData({jobId:job.job_id,status:'running',message:'正在恢复任务',requestId:'',started:Date.now()});await this.refresh()}catch(error){notifyError(error)}finally{this.setData({submitting:false})}
  },
  async stop(){if(!this.data.jobId)return;try{await request('/mobile/jobs/'+this.data.jobId,'DELETE');wx.showToast({title:'已请求停止',icon:'none'});await this.refresh()}catch(error){notifyError(error)}},
  plan(e:WechatMiniprogram.TouchEvent){wx.navigateTo({url:'/pages/plan/index?runId='+this.data.runId+'&kind='+String(e.currentTarget.dataset.kind)})},
})
