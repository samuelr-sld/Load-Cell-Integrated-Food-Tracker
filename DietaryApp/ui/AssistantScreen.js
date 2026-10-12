import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  StyleSheet, Text, View, TouchableOpacity, ScrollView, SafeAreaView, TextInput, ActivityIndicator,
  KeyboardAvoidingView, Platform, Alert, AppState,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createNutritionTools } from '../domain/nutritionTools';
import { ask, HISTORY_MESSAGES } from '../ai/assistant';
import { createGatewayProvider, createMockProvider, normalizeBaseUrl, newId } from '../ai/providers';
import { ChartView } from './Charts';

// Connection settings live apart from conversation records, so a token is never stored in a message.
// shortcut: AsyncStorage is plain text on the device; move the token to expo-secure-store before non-demo use.
const SETTINGS_KEY = 'platego.assistant.settings';
const DEFAULTS = { provider: 'laptop', baseUrl: '', token: '' };
const STARTERS = ['Analyze my week', 'Chart my protein this week', 'Compare this week and last week', 'Suggest a meal'];
const STATUS_COLOR = { checking: '#94a3b8', ok: '#10b981', down: '#dc2626' };

const Chip = ({ label, onPress, on, disabled }) => (
  <TouchableOpacity style={[s.chip, on && s.chipOn, disabled && { opacity: 0.4 }]} onPress={onPress} disabled={disabled} accessibilityRole="button">
    <Text style={[s.chipText, on && s.chipTextOn]}>{label}</Text>
  </TouchableOpacity>
);

export default function AssistantScreen({ repo, today, onClose }) {
  const [settings, setSettings] = useState(null);
  const [draft, setDraft] = useState(DEFAULTS);
  const [showSettings, setShowSettings] = useState(false);
  const [status, setStatus] = useState({ state: 'checking', message: 'Checking…' });
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [storageError, setStorageError] = useState(false);
  const [openFacts, setOpenFacts] = useState({});
  const abortRef = useRef(null);
  const scrollRef = useRef(null);
  const todayRef = useRef(today);
  todayRef.current = today;

  const tools = useMemo(() => createNutritionTools({ repo, today: () => todayRef.current }), [repo]);
  const mock = useMemo(() => createMockProvider({ today: () => todayRef.current }), []);
  const provider = useMemo(() => (settings?.provider === 'laptop' ? createGatewayProvider(settings) : mock), [settings, mock]);

  useEffect(() => {
    AsyncStorage.getItem(SETTINGS_KEY)
      .then((raw) => ({ ...DEFAULTS, ...(raw ? JSON.parse(raw) : {}) }))
      .catch(() => DEFAULTS)
      .then((v) => { setSettings(v); setDraft(v); setShowSettings(v.provider === 'laptop' && !v.baseUrl); });
    repo.getConversation()
      .then((ms) => { setMessages(ms); setConversationId(ms[0]?.conversationId ?? null); })
      .catch(() => setStorageError(true));
    return () => abortRef.current?.abort(); // closing the screen cancels the request in flight
  }, [repo]);

  const check = useCallback(async () => {
    if (!settings) return;
    setStatus({ state: 'checking', message: 'Checking…' });
    const h = await provider.health().catch(() => ({ ok: false, message: 'Status unknown.' }));
    setStatus({ state: h.ok ? 'ok' : 'down', message: h.message });
  }, [provider, settings]);

  useEffect(() => { check(); }, [check]);
  useEffect(() => {
    const sub = AppState.addEventListener('change', (st) => st === 'active' && check());
    return () => sub.remove();
  }, [check]);

  const saveSettings = async () => {
    const baseUrl = normalizeBaseUrl(draft.baseUrl);
    if (draft.provider === 'laptop' && (!baseUrl || !draft.token.trim())) {
      Alert.alert('Laptop AI settings', 'Enter an address like http://192.168.1.20:8100 and the pairing token shown by the gateway.');
      return;
    }
    const next = { ...draft, baseUrl: baseUrl ?? '', token: draft.token.trim() };
    setSettings(next);
    setShowSettings(false);
    await AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(next)).catch(() => setStorageError(true));
  };

  const add = (cid, msg) => {
    setMessages((ms) => [...ms, msg]);
    repo.appendAssistantMessage(cid, msg.role, msg).catch(() => setStorageError(true));
  };

  const send = async (text) => {
    const question = String(text).trim().slice(0, 500);
    if (!question || busy) return;
    setInput('');
    const cid = conversationId ?? newId();
    setConversationId(cid);
    const history = messages.slice(-HISTORY_MESSAGES).map((m) => ({ role: m.role, content: m.text }));
    add(cid, { role: 'user', text: question });
    const ac = new AbortController();
    abortRef.current = ac;
    setBusy(true);
    try {
      const r = await ask({ provider, fallbackProvider: mock, tools, question, history, today: todayRef.current, signal: ac.signal, requestId: newId() });
      add(cid, { role: 'assistant', question, text: r.text, facts: r.facts, charts: r.charts, notes: r.notes, source: r.source, model: r.model, notice: r.notice, errorCode: r.errorCode, followUps: r.followUps });
      if (provider.id === 'laptop' && r.errorCode && r.errorCode !== 'ungrounded' && r.errorCode !== 'max_rounds') setStatus({ state: 'down', message: r.notice });
    } catch (e) {
      const cancelled = e?.code === 'cancelled';
      if (!cancelled) console.error('assistant failed:', e?.message);
      add(cid, { role: 'assistant', question, text: cancelled ? 'Cancelled.' : 'Something went wrong while reading your diary.', errorCode: cancelled ? 'cancelled' : 'internal' });
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  };

  const clear = () => Alert.alert('Clear conversation?', 'Deletes this conversation from this device. Your meals are not affected.', [
    { text: 'Cancel', style: 'cancel' },
    {
      text: 'Clear', style: 'destructive',
      onPress: () => repo.clearConversations()
        .then(() => { setMessages([]); setConversationId(null); setStorageError(false); })
        .catch(() => setStorageError(true)),
    },
  ]);

  const last = messages[messages.length - 1];

  const bubble = (m, key) => (m.role === 'user' ? (
    <View key={key} style={[s.bubble, s.user]}><Text style={s.userText}>{m.text}</Text></View>
  ) : (
    <View key={key} style={[s.bubble, s.bot]}>
      {!!m.notice && <Text style={s.notice}>⚠ {m.notice}</Text>}
      <Text style={s.botText}>{m.text}</Text>
      {!!m.facts && (
        <TouchableOpacity onPress={() => setOpenFacts((o) => ({ ...o, [key]: !o[key] }))} accessibilityRole="button">
          <Text style={s.link}>{openFacts[key] ? '▾' : '▸'} Figures from your diary</Text>
          {openFacts[key] && <Text style={s.facts}>{m.facts}</Text>}
        </TouchableOpacity>
      )}
      {(m.charts ?? []).map((c, i) => <View key={i} style={s.chart}><ChartView chart={c} /></View>)}
      {(m.notes ?? []).map((n) => <Text key={n} style={s.noteLine}>• {n}</Text>)}
      <View style={s.metaRow}>
        <Text style={s.meta}>
          {m.source === 'model' ? `Laptop AI · ${m.model ?? ''} · numbers checked against your diary` : m.errorCode === 'cancelled' || m.errorCode === 'internal' ? '' : 'Built-in answer from your diary (no AI)'}
        </Text>
        {!!m.errorCode && m.question && !busy && (
          <TouchableOpacity onPress={() => send(m.question)} accessibilityRole="button"><Text style={s.retry}>↻ Retry</Text></TouchableOpacity>
        )}
      </View>
    </View>
  ));

  return (
    <SafeAreaView style={s.container}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <View style={s.header}>
          <View style={s.banner}>
            <Text style={s.bannerTitle}>NUTRITION ASSISTANT</Text>
            <TouchableOpacity onPress={onClose} style={s.closeBtn} accessibilityLabel="Close"><Text style={s.closeText}>✕</Text></TouchableOpacity>
          </View>
          <View style={s.statusRow}>
            <View style={[s.dot, { backgroundColor: STATUS_COLOR[status.state] }]} />
            <Text style={s.statusText} numberOfLines={2}>{provider.label}: {status.message}</Text>
            <TouchableOpacity onPress={check} accessibilityLabel="Check connection"><Text style={s.link}>Test</Text></TouchableOpacity>
            <TouchableOpacity onPress={() => setShowSettings((v) => !v)}><Text style={s.link}>Settings</Text></TouchableOpacity>
            {messages.length > 0 && <TouchableOpacity onPress={clear} disabled={busy}><Text style={[s.link, { color: '#dc2626' }]}>Clear</Text></TouchableOpacity>}
          </View>
          {storageError && <Text style={s.warn}>Conversation history could not be read or saved on this device; answers still work.</Text>}
        </View>

        {showSettings && (
          <View style={s.settings}>
            <View style={s.row}>
              <Chip label="Laptop AI (Ollama)" on={draft.provider === 'laptop'} onPress={() => setDraft({ ...draft, provider: 'laptop' })} />
              <Chip label="Built-in only (no AI)" on={draft.provider === 'offline'} onPress={() => setDraft({ ...draft, provider: 'offline' })} />
            </View>
            {draft.provider === 'laptop' && (
              <>
                <TextInput style={s.field} value={draft.baseUrl} onChangeText={(v) => setDraft({ ...draft, baseUrl: v })} placeholder="http://192.168.1.20:8100"
                  autoCapitalize="none" autoCorrect={false} keyboardType="url" accessibilityLabel="Gateway address" />
                <TextInput style={s.field} value={draft.token} onChangeText={(v) => setDraft({ ...draft, token: v })} placeholder="Pairing token"
                  autoCapitalize="none" autoCorrect={false} secureTextEntry accessibilityLabel="Pairing token" />
                <Text style={s.hint}>Run "python -m ai_gateway" on the laptop: it prints both values. Phone and laptop must be on the same Wi-Fi. Only short summaries for each question are sent, never your whole diary.</Text>
              </>
            )}
            <TouchableOpacity style={s.saveBtn} onPress={saveSettings}><Text style={s.saveText}>Save & test</Text></TouchableOpacity>
          </View>
        )}

        <ScrollView ref={scrollRef} style={{ flex: 1 }} contentContainerStyle={{ padding: 14 }} onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}>
          {messages.length === 0 && (
            <View style={s.empty}>
              <Text style={s.emptyTitle}>Ask about your meals</Text>
              <Text style={s.emptyText}>Answers and charts are computed from the meals saved on this phone. The AI only explains them, and every number is checked against your diary.</Text>
              <View style={[s.row, { justifyContent: 'center', marginTop: 10 }]}>
                {STARTERS.map((q) => <Chip key={q} label={q} onPress={() => send(q)} disabled={busy} />)}
              </View>
            </View>
          )}
          {messages.map((m, i) => bubble(m, `${i}-${m.role}`))}
          {busy && (
            <View style={[s.bubble, s.bot, s.row]}>
              <ActivityIndicator size="small" color="#10b981" />
              <Text style={s.botText}>  {provider.id === 'laptop' ? 'Asking the laptop AI…' : 'Reading your diary…'}</Text>
            </View>
          )}
          {!busy && last?.role === 'assistant' && last.followUps?.length > 0 && (
            <View style={[s.row, { marginTop: 4 }]}>{last.followUps.map((q) => <Chip key={q} label={q} onPress={() => send(q)} />)}</View>
          )}
        </ScrollView>

        <View style={s.inputRow}>
          <TextInput style={s.input} value={input} onChangeText={setInput} placeholder="Ask about your nutrition…" editable={!busy}
            onSubmitEditing={() => send(input)} returnKeyType="send" maxLength={500} accessibilityLabel="Question" />
          {busy ? (
            <TouchableOpacity style={[s.sendBtn, { backgroundColor: '#dc2626' }]} onPress={() => abortRef.current?.abort()} accessibilityLabel="Stop">
              <Text style={s.sendText}>Stop</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={[s.sendBtn, !input.trim() && { opacity: 0.4 }]} onPress={() => send(input)} disabled={!input.trim()} accessibilityLabel="Send">
              <Text style={s.sendText}>Send</Text>
            </TouchableOpacity>
          )}
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#ffffff' },
  header: { paddingHorizontal: 14, paddingTop: 14 },
  banner: { backgroundColor: '#0c3875', borderRadius: 8, paddingVertical: 12, paddingHorizontal: 14, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  bannerTitle: { color: '#ffffff', fontWeight: '900', fontSize: 16, letterSpacing: 0.5 },
  closeBtn: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#e2e8f0', alignItems: 'center', justifyContent: 'center' },
  closeText: { fontSize: 13, fontWeight: 'bold', color: '#475569' },
  statusRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8 },
  dot: { width: 9, height: 9, borderRadius: 5 },
  statusText: { flex: 1, fontSize: 12, color: '#334155' },
  link: { fontSize: 12, fontWeight: '700', color: '#1e3a8a' },
  warn: { fontSize: 11, color: '#b45309', paddingBottom: 6 },
  settings: { marginHorizontal: 14, padding: 12, borderRadius: 12, borderWidth: 1, borderColor: '#cbd5e1', backgroundColor: '#f8fafc' },
  field: { borderWidth: 1, borderColor: '#cbd5e1', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 8, marginTop: 8, fontSize: 14, backgroundColor: '#ffffff' },
  hint: { fontSize: 11, color: '#64748b', marginTop: 8 },
  saveBtn: { backgroundColor: '#1e3a8a', borderRadius: 10, paddingVertical: 10, alignItems: 'center', marginTop: 10 },
  saveText: { color: '#ffffff', fontWeight: 'bold' },
  row: { flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  chip: { paddingVertical: 6, paddingHorizontal: 12, borderRadius: 16, borderWidth: 1, borderColor: '#cbd5e1', backgroundColor: '#ffffff' },
  chipOn: { backgroundColor: '#10b981', borderColor: '#10b981' },
  chipText: { fontSize: 13, color: '#334155', fontWeight: '600' },
  chipTextOn: { color: '#ffffff' },
  empty: { alignItems: 'center', paddingVertical: 30, paddingHorizontal: 10 },
  emptyTitle: { fontSize: 18, fontWeight: '800', color: '#0f172a' },
  emptyText: { fontSize: 13, color: '#64748b', textAlign: 'center', marginTop: 6 },
  bubble: { borderRadius: 14, padding: 11, marginVertical: 5, maxWidth: '92%' },
  user: { alignSelf: 'flex-end', backgroundColor: '#10b981' },
  userText: { color: '#ffffff', fontSize: 15 },
  bot: { alignSelf: 'flex-start', backgroundColor: '#f1f5f9' },
  botText: { color: '#0f172a', fontSize: 15, lineHeight: 21 },
  notice: { color: '#b45309', fontSize: 12, marginBottom: 6, fontWeight: '600' },
  chart: { backgroundColor: '#ffffff', borderRadius: 10, padding: 10, marginTop: 8, borderWidth: 1, borderColor: '#e2e8f0' },
  noteLine: { fontSize: 11, color: '#475569', marginTop: 3 },
  facts: { fontSize: 12, color: '#334155', marginTop: 4, backgroundColor: '#ffffff', borderRadius: 8, padding: 8 },
  metaRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 6, gap: 8 },
  meta: { flex: 1, fontSize: 10, color: '#94a3b8', fontStyle: 'italic' },
  retry: { fontSize: 13, fontWeight: '800', color: '#1e3a8a' },
  inputRow: { flexDirection: 'row', padding: 10, gap: 8, borderTopWidth: 1, borderColor: '#e2e8f0' },
  input: { flex: 1, borderWidth: 1, borderColor: '#cbd5e1', borderRadius: 20, paddingHorizontal: 14, paddingVertical: 9, fontSize: 15 },
  sendBtn: { backgroundColor: '#10b981', borderRadius: 20, paddingHorizontal: 16, justifyContent: 'center' },
  sendText: { color: '#ffffff', fontWeight: 'bold', fontSize: 15 },
});
