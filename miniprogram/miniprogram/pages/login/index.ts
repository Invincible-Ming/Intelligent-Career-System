import { request, errorMessage } from '../../services/api'
import { saveSession, session } from '../../services/session'
import { Auth, Options } from '../../services/types'
Page({
  data: { username:'', password:'', confirmation:'', register:false, registrationEnabled:false, loading:false, error:'' },
  async onLoad() {
    if (session()) { wx.switchTab({url:'/pages/chat/index'}); return }
    try { const options=await request<Options>('/auth/options','GET',undefined,false); this.setData({registrationEnabled:options.registration_enabled}) } catch(error) { this.setData({error:errorMessage(error)}) }
  },
  onUsername(e: WechatMiniprogram.Input) { this.setData({username:e.detail.value}) },
  onPassword(e: WechatMiniprogram.Input) { this.setData({password:e.detail.value}) },
  onConfirmation(e: WechatMiniprogram.Input) { this.setData({confirmation:e.detail.value}) },
  toggleMode() { this.setData({register:!this.data.register,error:''}) },
  async submit() {
    if(this.data.loading)return
    const {username,password,register,confirmation}=this.data
    if(!/^[a-zA-Z0-9_]{3,32}$/.test(username)||password.length<12||password.length>128){this.setData({error:'用户名为 3–32 位字母、数字或下划线，密码为 12–128 位'});return}
    if(register&&confirmation!==password){this.setData({error:'两次密码不一致'});return}
    this.setData({loading:true,error:''})
    try {
      if(register){await request('/auth/register','POST',{username,password},false);this.setData({register:false,confirmation:''});wx.showToast({title:'注册成功，请登录',icon:'none'});return}
      saveSession(await request<Auth>('/auth/login','POST',{username,password},false))
      this.setData({password:'',confirmation:''})
      wx.switchTab({url:'/pages/chat/index'})
    }catch(error){this.setData({error:errorMessage(error)})}finally{this.setData({loading:false})}
  },
})
