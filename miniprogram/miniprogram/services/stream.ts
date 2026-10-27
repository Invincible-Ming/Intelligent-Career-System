import { apiBase } from '../config'
import { session } from './session'
import { ApiError, checkResponse } from './api'

// WeChat's JS runtime does not consistently expose TextDecoder. Carry incomplete
// UTF-8 sequences across chunks, including four-byte emoji characters.
export class Utf8Decoder {
  private pending: number[] = []
  decode(buffer: ArrayBuffer): string {
    const bytes = this.pending.concat(Array.from(new Uint8Array(buffer)))
    this.pending = []
    let result = '', i = 0
    while (i < bytes.length) {
      const b = bytes[i], n = b < 128 ? 1 : b < 224 ? 2 : b < 240 ? 3 : 4
      if (i + n > bytes.length) { this.pending = bytes.slice(i); break }
      let code = b & (n === 1 ? 127 : n === 2 ? 31 : n === 3 ? 15 : 7)
      for (let j = 1; j < n; j++) code = (code << 6) | (bytes[i + j] & 63)
      result += String.fromCodePoint(code)
      i += n
    }
    return result
  }
}
export class SseDecoder {
  private utf8 = new Utf8Decoder()
  private buffer = ''
  feed(chunk: ArrayBuffer): string[] {
    this.buffer += this.utf8.decode(chunk)
    this.buffer = this.buffer.replace(/\r\n/g, '\n')
    const frames: string[] = []
    let end = this.buffer.indexOf('\n\n')
    while (end >= 0) {
      const frame = this.buffer.slice(0, end)
      this.buffer = this.buffer.slice(end + 2)
      const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n')
      if (data) frames.push(data)
      end = this.buffer.indexOf('\n\n')
    }
    if (this.buffer.length > 2000000) throw new ApiError('回复超过显示范围')
    return frames
  }
}
export interface StreamEvent { conversation_id?: string; user_message_id?: string; message_id?: string; choices?: { delta?: { content?: string } }[]; content?: string; error?: { message?: string } }
export function streamChat(payload: object, onEvent: (event: StreamEvent) => void): { done: Promise<void>; stop: () => void } {
  let task: WechatMiniprogram.RequestTask | undefined
  let stopped = false, ended = false, received = false
  const decoder = new SseDecoder()
  const done = new Promise<void>((resolve, reject) => {
    try {
      const token = session()?.token
      if (!token) throw new ApiError('请先登录', 401)
      const consume = (frame: string): void => {
        if (frame === '[DONE]') { ended = true; return }
        let value: StreamEvent
        try { value = JSON.parse(frame) as StreamEvent } catch (_) { return }
        if (value.error) { reject(new ApiError(value.error.message || '回复中断')); task?.abort(); return }
        onEvent(value)
      }
      task = wx.request({ url: apiBase() + '/chat/completions', method: 'POST', data: payload,
        enableChunked: true, responseType: 'arraybuffer', timeout: 120000,
        header: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        success: response => {
          try {
            if (response.statusCode >= 400) {
              const body = new Utf8Decoder().decode(response.data as ArrayBuffer)
              let detail: unknown = null
              try { detail = JSON.parse(body) } catch (_) { /* generic response below */ }
              checkResponse(response.statusCode, detail)
            }
            if (!received && response.data instanceof ArrayBuffer) decoder.feed(response.data).forEach(consume)
            if (ended || stopped) resolve(); else reject(new ApiError('连接在回复完成前中断，请查看历史后重试'))
          } catch (error) { reject(error) }
        },
        fail: () => stopped ? resolve() : reject(new ApiError('连接已中断，请查看历史后重试')),
      })
      task.onChunkReceived(response => {
        received = true
        try { decoder.feed(response.data).forEach(consume) } catch (error) { reject(error); task?.abort() }
      })
    } catch (error) { reject(error) }
  })
  return { done, stop: () => { stopped = true; task?.abort() } }
}
