import { developmentApiBase } from './config.local'

// 上线前填入已配置合法域名的正式 HTTPS 地址（末尾包含 /mini-api）。
const releaseApiBase = ''
// 开发版允许走本机回环地址的 http，便于在开发者工具里直连 deploy/gateway（127.0.0.1:8080）。
// 真机、体验版和正式版仍必须是 https，避免把明文地址带进发布包。
const LOOPBACK_HTTP = /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?(\/|$)/
export function apiBase(): string {
  const version = wx.getAccountInfoSync().miniProgram.envVersion
  const base = version === 'release' ? releaseApiBase : developmentApiBase
  if (!base) throw new Error('请先配置小程序服务地址')
  if (!/^https:\/\//.test(base) && !(version === 'develop' && LOOPBACK_HTTP.test(base))) {
    throw new Error('请先配置小程序服务地址')
  }
  return base.replace(/\/$/, '')
}
