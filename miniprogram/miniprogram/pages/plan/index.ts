import { dateLabel } from '../../services/presentation'
import { errorMessage, request, uuid } from '../../services/api'
import { requireSession } from '../../services/session'
import { submitJob } from '../../services/jobs'
import { Run } from '../../services/types'
Page({
  data:{runId:'',kind:'interview',difficultyIndex:1,difficulties:['入门','进阶','深入'],questionCount:8,weeks:4,hours:10,submitting:false,error:'',requestId:'',history:[] as Run[]},
  async onLoad(query:Record<string,string|undefined>){if(!requireSession())return;const kind=query.kind==='learning'?'learning':'interview';this.setData({runId:query.runId||'',kind});wx.setNavigationBarTitle({title:kind==='interview'?'面试准备':'学习计划'});try{this.setData({history:await request<Run[]>('/runs?task_type='+(kind==='learning'?'learning_plan':'interview')+'&status=completed&limit=20').then(rows=>rows.map(item=>({...item,timeText:dateLabel(item.created_at)})))})}catch(error){this.setData({error:errorMessage(error)})}},
  difficultyChange(e:WechatMiniprogram.PickerChange){this.setData({difficultyIndex:Number(e.detail.value),requestId:''})},
  countChange(e:WechatMiniprogram.SliderChange){this.setData({questionCount:e.detail.value,requestId:''})},
  weeksChange(e:WechatMiniprogram.SliderChange){this.setData({weeks:e.detail.value,requestId:''})},
  hoursChange(e:WechatMiniprogram.SliderChange){this.setData({hours:e.detail.value,requestId:''})},
  async submit(){if(this.data.submitting)return;const payload=this.data.kind==='interview'?{match_run_id:this.data.runId,difficulty:['junior','intermediate','senior'][this.data.difficultyIndex],question_count:this.data.questionCount}:{match_run_id:this.data.runId,available_weeks:this.data.weeks,hours_per_week:this.data.hours};const id=this.data.requestId||uuid();this.setData({submitting:true,error:'',requestId:id});try{const job=await submitJob(this.data.kind,payload,id);this.setData({requestId:''});wx.redirectTo({url:'/pages/task/index?jobId='+job.job_id})}catch(error){this.setData({error:errorMessage(error)})}finally{this.setData({submitting:false})}},
  openRun(e:WechatMiniprogram.TouchEvent){wx.navigateTo({url:'/pages/task/index?runId='+String(e.currentTarget.dataset.id)})},
})
