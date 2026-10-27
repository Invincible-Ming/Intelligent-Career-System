import { download, errorMessage, notifyError, request } from '../../services/api'
import { requireSession } from '../../services/session'
Page({
  data:{id:'',filename:'',content:'',truncated:false,loading:true,error:''},
  async onLoad(query:Record<string,string|undefined>){if(!requireSession())return;const id=query.id||'';this.setData({id});try{const data=await request<{filename:string;content:string;truncated:boolean}>('/documents/'+encodeURIComponent(id)+'/text');this.setData(data);wx.setNavigationBarTitle({title:data.filename})}catch(error){this.setData({error:errorMessage(error)})}finally{this.setData({loading:false})}},
  async download(){try{await download(this.data.id)}catch(error){notifyError(error)}},
})
