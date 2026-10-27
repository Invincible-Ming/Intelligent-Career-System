export interface User { id: string; username: string; is_admin: boolean; knowledge_provider: 'ima' | 'milvus' }
export interface Auth { access_token: string; expires_at: string; user: User }
export interface Options { registration_enabled: boolean; chat_max_input_chars: number; jd_max_input_chars: number; max_upload_size_mb: number; knowledge_provider: string }
export interface Conversation { id: string; title: string; pinned: boolean; message_count: number }
export interface Message { id?: string; role: string; content: string; html?: string; incomplete?: boolean }
export interface DocumentItem { id: string; filename: string; document_type: string; status: string; chunk_count: number; created_at?: string; error_message?: string; label?: string; statusLabel?: string }
export interface SearchResult { document_id: string; content: string; source: string; score: number }
export interface MatchReport { total_score: number; match_level: string; scores: Record<string, number>; matched_skills: string[]; missing_skills: string[]; strengths: string[]; risks: string[]; suggestions: string[]; verification?: { summary: string; status: string }; job_analysis?: { job_title: string } }
export interface Run { run_id: string; task_type: string; status: string; result: Record<string, unknown> | null; error_message?: string; created_at?: string; label?: string }
export interface Job { job_id: string; kind: string; status: string; run_id?: string; message: string; error: string; result: Record<string, unknown> | null }
export interface Wallet { balance: number; available: number; reserved: number; checked_in_today: boolean; rules: { initial_points: number; daily_checkin_points: number; points_per_1000_tokens: number } }
export interface CreditEntry { id: string; kind: string; status: string; points: number; input_tokens: number; output_tokens: number; created_at: string; label?: string; pointsText?: string; timeText?: string }
