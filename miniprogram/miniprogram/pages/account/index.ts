import { dateLabel } from '../../services/presentation'
import { errorMessage, notifyError, request } from '../../services/api'
import { clearSession, requireSession, session } from '../../services/session'
import { CreditEntry, Wallet } from '../../services/types'
const kinds:Record<string,string>={initial:'初始赠送',checkin:'签到奖励',usage:'模型用量'}
Page({
  data:{username:'',initial:'',wallet:null as Wallet|null,available:'—',entries:[] as CreditEntry[],hasMore:false,loading:false,checking:false,error:'',usage:{daily_used:0,daily_limit:0,minute_used:0,minute_limit:0},isAdmin:false,rulesOpen:false,aboutOpen:false,ledgerOpen:false},
  async onShow(){if(!requireSession())return;const username=session()?.user.username||'';this.setData({username,initial:username.slice(0,1).toUpperCase()});await this.refresh()},
  async refresh(){this.setData({loading:true,error:''});try{const [wallet,usage]=await Promise.all([request<Wallet>('/credits/me'),request<{is_admin:boolean;generation_daily:{used:number;limit:number}|null;generation_minute:{used:number;limit:number}|null}>('/auth/usage')]);this.setData({wallet,available:wallet.available.toFixed(3),isAdmin:usage.is_admin,usage:{daily_used:usage.generation_daily?.used||0,daily_limit:usage.generation_daily?.limit||0,minute_used:usage.generation_minute?.used||0,minute_limit:usage.generation_minute?.limit||0}});await this.loadEntries(true)}catch(error){this.setData({error:errorMessage(error)})}finally{this.setData({loading:false})}},
  async loadEntries(reset=false){try{const offset=reset?0:this.data.entries.length;const result=await request<{items:CreditEntry[];has_more:boolean}>('/credits/entries?limit=20&offset='+offset);const rows=result.items.map(entry=>({...entry,label:kinds[entry.kind]||entry.kind,pointsText:(entry.points>0?'+':'')+entry.points.toFixed(3),timeText:dateLabel(entry.created_at)}));this.setData({entries:reset?rows:this.data.entries.concat(rows),hasMore:result.has_more})}catch(error){notifyError(error)}},
  more(){void this.loadEntries()},
  async checkIn(){if(this.data.checking||this.data.wallet?.checked_in_today)return;this.setData({checking:true});try{await request('/credits/check-in','POST');wx.showToast({title:'签到成功',icon:'success'});await this.refresh()}catch(error){notifyError(error)}finally{this.setData({checking:false})}},
  toggleLedger(){this.setData({ledgerOpen:!this.data.ledgerOpen,rulesOpen:false,aboutOpen:false})},
  toggleRules(){this.setData({rulesOpen:!this.data.rulesOpen,ledgerOpen:false,aboutOpen:false})},
  toggleAbout(){this.setData({aboutOpen:!this.data.aboutOpen,rulesOpen:false,ledgerOpen:false})},
  async logout(){const response=await wx.showModal({title:'退出登录',content:'确定退出当前账号吗？',confirmText:'退出'});if(!response.confirm)return;try{await request('/auth/logout','POST')}catch(_){/* Local sign-out is still possible when offline. */}clearSession();wx.reLaunch({url:'/pages/login/index'})},
})
