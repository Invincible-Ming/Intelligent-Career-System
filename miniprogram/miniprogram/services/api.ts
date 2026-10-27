import { apiBase } from '../config'
import { clearSession, session } from './session'
export class ApiError extends Error { constructor(message: string, public status = 0) { super(message) } }
export function errorMessage(value: unknown): string { return value instanceof Error ? value.message : '操作未完成，请稍后重试' }
export function notifyError(value: unknown): void { wx.showToast({ title: errorMessage(value), icon: 'none', duration: 3000 }) }
function authHeaders(auth = true): Record<string, string> {
  const token = session()?.token
  if (auth && !token) throw new ApiError('请先登录', 401)
  return token && auth ? { Authorization: `Bearer ${token}` } : {}
}
export function checkResponse(status: number, data: unknown, auth = true): void {
  if (status >= 200 && status < 300) return
  if (status === 401 && auth) {
    clearSession()
    wx.reLaunch({ url: '/pages/login/index' })
  }
  const detail = data && typeof data === 'object' ? (data as { detail?: unknown }).detail : null
  throw new ApiError(typeof detail === 'string' ? detail : status === 422 ? '请检查输入内容和长度' : '服务暂不可用，请稍后重试', status)
}
export function request<T>(path: string, method: WechatMiniprogram.RequestOption['method'] = 'GET', data?: object, auth = true): Promise<T> {
  return new Promise((resolve, reject) => {
    try {
      wx.request({ url: apiBase() + path, method, data, timeout: 30000,
        header: { 'Content-Type': 'application/json', ...authHeaders(auth) },
        success: response => { try { checkResponse(response.statusCode, response.data, auth); resolve(response.data as T) } catch (error) { reject(error) } },
        fail: () => reject(new ApiError('网络连接失败，请检查网络后重试')),
      })
    } catch (error) { reject(error) }
  })
}
export function upload(path: string, type: string): Promise<{ document: import('./types').DocumentItem; deduplicated: boolean }> {
  return new Promise((resolve, reject) => {
    try {
      wx.uploadFile({ url: apiBase() + '/documents/upload', filePath: path, name: 'file', timeout: 120000,
        formData: { document_type: type }, header: authHeaders(),
        success: response => { try { const body = JSON.parse(response.data); checkResponse(response.statusCode, body); resolve(body) } catch (error) { reject(error) } },
        fail: () => reject(new ApiError('上传连接中断，请刷新文件列表确认结果后再试')),
      })
    } catch (error) { reject(error) }
  })
}
export function download(id: string): Promise<void> {
  return new Promise((resolve, reject) => {
    try {
      wx.downloadFile({ url: apiBase() + '/documents/' + encodeURIComponent(id) + '/download', header: authHeaders(), timeout: 120000,
        success: response => {
          try { checkResponse(response.statusCode, null) } catch (error) { reject(error); return }
          wx.openDocument({ filePath: response.tempFilePath, showMenu: true, success: () => resolve(), fail: () => reject(new ApiError('此文件暂不支持直接预览，请在 Web 端下载')) })
        }, fail: () => reject(new ApiError('文件下载失败，请稍后重试')),
      })
    } catch (error) { reject(error) }
  })
}
export function uuid(): string { return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, char => { const r = Math.floor(Math.random() * 16); return (char === 'x' ? r : (r & 3) | 8).toString(16) }) }
