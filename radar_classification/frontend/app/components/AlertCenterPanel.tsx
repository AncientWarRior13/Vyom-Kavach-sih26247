'use client';











import React, { useEffect, useState } from 'react';
import { Send, RefreshCw, Mail, MessageSquare, Radio } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8081';

interface RosterEntry { role: string; channel: string; contact: string; }
interface Recipient extends RosterEntry { status: string; simulated: boolean; }
interface PersonalContact {
  role: string; channel: string; contact: string; simulated: boolean; status: string;
  detail: any;
}
interface AlertRecord {
  alert_id: string; timestamp: number; track_id: string; predicted_class: string;
  threat_level: string; threat_score: number; range_m: number; closing_mps: number;
  tti_sec: number | null; rationale: string;
  recipients: Recipient[]; personal_contacts: PersonalContact[]; webhook: any;
}

export default function AlertCenterPanel({ theme }: { theme: 'dark' | 'light' }) {
  const isDark = theme === 'dark';
  const [roster, setRoster] = useState<RosterEntry[] | null>(null);
  const [history, setHistory] = useState<AlertRecord[]>([]);
  const [webhookConfigured, setWebhookConfigured] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    fetch(`${API_BASE}/alert/roster`).then(r => r.json()).then(d => setRoster(d.roster)).catch(() => {});
    fetch(`${API_BASE}/alert/history`).then(r => r.json()).then(d => {
      setHistory(d.alerts || []);
      setWebhookConfigured(!!d.webhook_configured);
    }).catch(() => setError('Could not reach the API — is uvicorn api.app:app running on :8081?'));
  };

  useEffect(() => { refresh(); const id = setInterval(refresh, 5000); return () => clearInterval(id); }, []);

  const sendTestAlert = async () => {
    setSending(true);
    try {
      const res = await fetch(`${API_BASE}/alert/dispatch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          track_id: `TEST-${Date.now() % 100000}`,
          predicted_class: 'drone', threat_level: 'CRITICAL', threat_score: 92,
          range_m: 210, closing_mps: 18.4, tti_sec: 11,
          rationale: 'DRONE 210 m · closing 18.4 m/s · TTI 11 s · manual test dispatch',
        }),
      });
      await res.json();
      refresh();
    } catch {
      setError('Test dispatch failed — API unreachable.');
    } finally {
      setSending(false);
    }
  };

  const emailConfigured = history.some(h => h.personal_contacts?.some(
    p => p.channel === 'email' && p.detail?.attempted !== false
  ));

  return (
    <div className="flex flex-col gap-6">
      <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-3 transition-colors duration-300 ${isDark ? 'bg-zinc-950/70 border-red-950/40' : 'bg-white border-zinc-200'}`}>
        <div className={`flex justify-between items-center border-b pb-2 ${isDark ? 'border-zinc-900' : 'border-zinc-200'}`}>
          <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
            <Radio className="w-4 h-4 text-red-400" /> Alert Dispatch
          </span>
          <button
            onClick={sendTestAlert}
            disabled={sending}
            className="px-3 py-1.5 font-mono text-[11px] font-bold border rounded-lg flex items-center gap-1.5 cursor-pointer transition-all bg-red-500/20 text-red-400 border-red-500/40 hover:bg-red-500/30 disabled:opacity-50"
          >
            {sending ? <><RefreshCw className="w-3 h-3 animate-spin" /> Dispatching…</> : <><Send className="w-3 h-3" /> Send Test Alert</>}
          </button>
        </div>

        <div className={`text-[10px] font-sans leading-relaxed p-2.5 rounded-lg border ${isDark ? 'bg-amber-950/10 border-amber-900/30 text-amber-300/90' : 'bg-amber-50 border-amber-200 text-amber-700'}`}>
          The roster below is <strong>simulated demo data</strong> — fictional roles, example.com
          addresses, 555-01xx numbers. No real SMS/email is ever sent to it. The only real channel
          is the personal email row (Gmail SMTP, configured via <code>RADARSENSE_SMTP_PASS</code> on
          the server) and an optional webhook — both shown separately below.
        </div>

        {error && <p className="text-[11px] font-mono text-red-400">{error}</p>}

        <div className="grid grid-cols-2 gap-3 text-[10px] font-mono">
          <div className={`border rounded-lg p-2.5 flex items-center gap-2 ${isDark ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
            <Mail className="w-3.5 h-3.5 text-zinc-500" />
            <span>Real email: <span className={emailConfigured ? 'text-emerald-400 font-bold' : 'text-zinc-500'}>
              {emailConfigured ? 'CONFIGURED' : 'not configured (RADARSENSE_SMTP_PASS unset)'}
            </span></span>
          </div>
          <div className={`border rounded-lg p-2.5 flex items-center gap-2 ${isDark ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
            <MessageSquare className="w-3.5 h-3.5 text-zinc-500" />
            <span>Webhook: <span className={webhookConfigured ? 'text-emerald-400 font-bold' : 'text-zinc-500'}>
              {webhookConfigured ? 'CONFIGURED' : 'not configured (RADARSENSE_ALERT_WEBHOOK_URL unset)'}
            </span></span>
          </div>
        </div>
      </div>

      <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-3 transition-colors duration-300 ${isDark ? 'bg-zinc-950/70 border-red-950/40' : 'bg-white border-zinc-200'}`}>
        <span className={`font-mono text-sm font-semibold tracking-wider ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>Simulated Roster</span>
        <div className="grid grid-cols-2 gap-2">
          {(roster || []).map((r, i) => (
            <div key={i} className={`border rounded-lg p-2 text-[10px] font-mono ${isDark ? 'bg-zinc-900/30 border-zinc-900 text-zinc-400' : 'bg-zinc-50 border-zinc-200 text-zinc-600'}`}>
              <div className="font-bold">{r.role}</div>
              <div className="opacity-70">{r.channel.toUpperCase()} · {r.contact}</div>
            </div>
          ))}
        </div>
      </div>

      <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-3 transition-colors duration-300 ${isDark ? 'bg-zinc-950/70 border-red-950/40' : 'bg-white border-zinc-200'}`}>
        <span className={`font-mono text-sm font-semibold tracking-wider ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>Dispatch History</span>
        {history.length === 0 && (
          <p className="text-[11px] font-mono text-zinc-500">No alerts dispatched yet — a CRITICAL threat on the main radar page, or the test button above, will populate this.</p>
        )}
        <div className="flex flex-col gap-2 max-h-[420px] overflow-y-auto">
          {history.map(h => (
            <div key={h.alert_id} className={`border rounded-lg p-2.5 text-[10px] font-mono ${isDark ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'}`}>
              <div className="flex justify-between items-center mb-1">
                <span className="font-bold text-red-400">{h.threat_level} · {h.predicted_class.toUpperCase()} · {h.track_id}</span>
                <span className="text-zinc-500">{new Date(h.timestamp * 1000).toLocaleTimeString()}</span>
              </div>
              <div className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>{h.rationale}</div>
              <div className="flex gap-3 mt-1 text-zinc-500">
                <span>{h.recipients.length} simulated recipients</span>
                <span>{h.personal_contacts.filter(p => !p.simulated && p.status === 'sent').length} real email sent</span>
                {h.webhook && <span>webhook: {h.webhook.ok ? 'ok' : 'failed'}</span>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
