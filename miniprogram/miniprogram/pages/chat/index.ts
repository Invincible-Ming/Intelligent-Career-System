import { errorMessage, notifyError, request } from '../../services/api'
import { requireSession, session } from '../../services/session'
import { markdown } from '../../services/markdown'
import { streamChat } from '../../services/stream'
import { Conversation, Message, Options } from '../../services/types'

const PAGE_SIZE = 50
let active: ReturnType<typeof streamChat> | null = null
let alive = false
let visible = false
let generation = 0
function renderMessages(messages: Message[]): Message[] {
  return messages.map(message => ({ ...message, html: message.role === 'assistant' ? markdown(message.content) : '' }))
}
Page({
  _historyRequest: 0,
  _pendingPrompt: '',
  data: { username:'', draft:'', messages:[] as Message[], conversations:[] as Conversation[], conversationId:'', sending:false, historyOpen:false, error:'', scrollTo:'', maxChars:4000, historyOffset:0, canLoadMore:false, loadingHistory:false },
  onLoad() { alive = true },
  async onShow() {
    alive = visible = true
    if (!requireSession()) return
    this.setData({ username:session()?.user.username || '' })
    try {
      const options = await request<Options>('/auth/options')
      if (!alive || !visible) return
      this.setData({ maxChars:options.chat_max_input_chars })
      await this.loadConversations()
      if (this.data.conversationId && !this.data.sending) await this.loadLatest(this.data.conversationId)
    } catch (error) { if (visible) notifyError(error) }
  },
  onHide() { visible = false; this.stop() },
  onUnload() { alive = visible = false; this.stop() },
  input(e: WechatMiniprogram.Input) { this.setData({ draft:e.detail.value }) },
  async loadConversations() {
    const conversations = await request<Conversation[]>('/chat/conversations')
    if (alive && visible) this.setData({ conversations })
  },
  async toggleHistory() {
    this.setData({ historyOpen:!this.data.historyOpen })
    if (this.data.historyOpen) { try { await this.loadConversations() } catch (error) { notifyError(error) } }
  },
  async loadLatest(id: string) {
    const sequence = ++this._historyRequest
    this.setData({ loadingHistory:true })
    try {
      // The backend counts offset from the NEWEST message and reverses each
      // returned page for display. Offset 0 is always the latest page.
      const messages = renderMessages(await request<Message[]>('/chat/conversations/' + id + '/messages?limit=' + PAGE_SIZE + '&offset=0'))
      if (!alive || !visible || sequence !== this._historyRequest || id !== this.data.conversationId) return
      this.setData({ messages, historyOffset:messages.length, canLoadMore:messages.length === PAGE_SIZE })
      this.toBottom()
    } finally {
      if (alive && sequence === this._historyRequest) this.setData({ loadingHistory:false })
    }
  },
  async openConversation(e: WechatMiniprogram.TouchEvent) {
    if (this.data.sending) return
    const id = String(e.currentTarget.dataset.id)
    this.setData({ conversationId:id, messages:[], historyOpen:false, error:'', canLoadMore:false, historyOffset:0 })
    try { await this.loadLatest(id) } catch (error) { notifyError(error) }
  },
  async loadMore() {
    if (this.data.loadingHistory || this.data.sending || !this.data.canLoadMore || !this.data.conversationId) return
    const id = this.data.conversationId, offset = this.data.historyOffset, sequence = ++this._historyRequest
    this.setData({ loadingHistory:true })
    try {
      const messages = renderMessages(await request<Message[]>('/chat/conversations/' + id + '/messages?limit=' + PAGE_SIZE + '&offset=' + offset))
      if (!alive || !visible || sequence !== this._historyRequest || id !== this.data.conversationId) return
      this.setData({ messages:messages.concat(this.data.messages), historyOffset:offset + messages.length, canLoadMore:messages.length === PAGE_SIZE })
    } catch (error) { if (visible) notifyError(error) }
    finally { if (alive && sequence === this._historyRequest) this.setData({ loadingHistory:false }) }
  },
  newChat() {
    if (this.data.sending) return
    this._historyRequest++
    this.setData({ conversationId:'', messages:[], draft:'', error:'', historyOpen:false, canLoadMore:false, historyOffset:0, loadingHistory:false })
  },
  async manageConversation(e: WechatMiniprogram.TouchEvent) {
    const id = String(e.currentTarget.dataset.id), conv = this.data.conversations.find(item => item.id === id)
    if (!conv) return
    const choice = await wx.showActionSheet({ itemList:['重命名', conv.pinned ? '取消置顶' : '置顶', '删除对话'] }).catch(() => null)
    if (!choice) return
    try {
      if (choice.tapIndex === 2) {
        const confirmation = await wx.showModal({ title:'删除这段对话？', content:'删除后无法恢复。', confirmText:'删除', confirmColor:'#b4453d' })
        if (!confirmation.confirm) return
        await request('/chat/conversations/' + id, 'DELETE')
        if (this.data.conversationId === id) this.newChat()
      } else if (choice.tapIndex === 1) {
        await request('/chat/conversations/' + id + '/update?pinned=' + !conv.pinned, 'POST')
      } else {
        const result = await wx.showModal({ title:'重命名对话', editable:true, placeholderText:conv.title })
        if (!result.confirm || !result.content?.trim()) return
        await request('/chat/conversations/' + id + '/update?title=' + encodeURIComponent(result.content.trim()), 'POST')
      }
      await this.loadConversations()
    } catch (error) { notifyError(error) }
  },
  toBottom() {
    if (!alive || !visible) return
    this.setData({ scrollTo:'' })
    wx.nextTick(() => { if (alive && visible) this.setData({ scrollTo:'chat-bottom' }) })
  },
  stop() {
    const stream = active, id = this.data.conversationId
    generation++
    this._historyRequest++
    active = null
    stream?.stop()
    if (alive) {
      // Cancelling during conversation creation restores the unsent question.
      this.setData({ sending:false, loadingHistory:false, ...(this._pendingPrompt ? { draft:this._pendingPrompt } : {}) })
    }
    this._pendingPrompt = ''
    if (stream && alive && visible && id) {
      const epoch = generation
      void stream.done.catch(() => {}).then(async () => {
        if (alive && visible && epoch === generation && id === this.data.conversationId) {
          try { await this.loadLatest(id); await this.loadConversations() }
          catch (error) { if (visible) notifyError(error) }
        }
      })
    }
  },
  async send() {
    const prompt = this.data.draft.trim()
    if (!prompt || this.data.sending || this.data.loadingHistory || !visible) return
    const epoch = ++generation
    this._pendingPrompt = prompt
    this.setData({ sending:true, error:'', draft:'', historyOpen:false })
    try {
      let id = this.data.conversationId
      if (!id) {
        const created = await request<{id:string}>('/chat/conversations', 'POST', { title:prompt.slice(0,30) })
        // A hide, unload or explicit Stop invalidates pending async work even
        // before a streaming request exists. A later onShow cannot revive it.
        if (!alive || !visible || epoch !== generation) return
        id = created.id
        this.setData({ conversationId:id })
      }
      if (!alive || !visible || epoch !== generation) return
      this._pendingPrompt = ''
      const messages = this.data.messages.concat([{ role:'user', content:prompt }, { role:'assistant', content:'' }])
      this.setData({ messages }); this.toBottom()
      const index = messages.length - 1
      const stream = streamChat({ conversation_id:id, message:prompt, stream:true }, event => {
        if (!alive || !visible || epoch !== generation) return
        const delta = event.choices?.[0]?.delta?.content || event.content || ''
        const updates: Record<string,unknown> = {}
        if (delta) {
          messages[index].content += delta
          updates['messages[' + index + '].content'] = messages[index].content
          updates['messages[' + index + '].html'] = markdown(messages[index].content)
        }
        if (event.message_id) { messages[index].id = event.message_id; updates['messages[' + index + '].id'] = event.message_id }
        if (Object.keys(updates).length) this.setData(updates)
        this.toBottom()
      })
      active = stream
      await stream.done
      if (alive && visible && epoch === generation) {
        await this.loadLatest(id)
        await this.loadConversations()
      }
    } catch (error) {
      if (alive && visible && epoch === generation) {
        this.setData({ error:errorMessage(error), ...(this._pendingPrompt ? { draft:this._pendingPrompt } : {}) })
      }
    } finally {
      if (alive && epoch === generation) {
        this._pendingPrompt = ''; active = null
        this.setData({ sending:false }); this.toBottom()
      }
    }
  },
})
