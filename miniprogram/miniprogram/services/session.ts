import { Auth, User } from './types'
const STORAGE = 'jmzs.session.v1'
interface Stored { token: string; expires: string; user: User }
export function session(): Stored | null {
  const value = wx.getStorageSync(STORAGE) as Stored | undefined
  if (!value || typeof value.token !== 'string' || !value.user || !Number.isFinite(Date.parse(value.expires)) || Date.parse(value.expires) <= Date.now()) {
    wx.removeStorageSync(STORAGE)
    return null
  }
  return value
}
export function saveSession(auth: Auth): void {
  wx.setStorageSync(STORAGE, { token: auth.access_token, expires: auth.expires_at, user: auth.user })
}
export function updateUser(user: User): void { const value = session(); if (value) wx.setStorageSync(STORAGE, { ...value, user }) }
export function clearSession(): void { wx.removeStorageSync(STORAGE); wx.removeStorageSync('jmzs.active-conversation') }
export function requireSession(): boolean {
  if (session()) return true
  const pages = getCurrentPages()
  if (pages.length && pages[pages.length - 1].route !== 'pages/login/index') wx.reLaunch({ url: '/pages/login/index' })
  return false
}
