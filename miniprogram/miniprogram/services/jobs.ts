import { ApiError, request, uuid } from './api'
import { session } from './session'
import { Job } from './types'
export function fingerprint(value:string):string {
  let hash=2166136261
  for(let i=0;i<value.length;i++)hash=Math.imul(hash^value.charCodeAt(i),16777619)
  return (hash>>>0).toString(16)
}
export async function submitJob(kind:string,payload:object,id=uuid()):Promise<Job>{
  const owner=session()?.user.id
  if(!owner)throw new ApiError('请先登录',401)
  const key='jmzs.pending-job.'+owner
  const signature=fingerprint(JSON.stringify([kind,payload]))
  const previous=wx.getStorageSync(key) as {id:string;signature:string}|undefined
  const identifier=previous?.signature===signature?previous.id:id
  wx.setStorageSync(key,{id:identifier,signature})
  try{
    const job=await request<Job>('/mobile/jobs','POST',{client_request_id:identifier,kind,payload})
    wx.removeStorageSync(key)
    return job
  }catch(error){
    if(error instanceof ApiError&&error.status>=400&&error.status<500&&error.status!==409)wx.removeStorageSync(key)
    throw error
  }
}
