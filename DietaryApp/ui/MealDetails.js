import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, SafeAreaView, Alert, Platform } from 'react-native';
import { mealTotals, nutrientsForServing } from '../domain/meal';
import { RevisionConflictError } from '../data/mealRepository';

const fmt = (v) => (v == null ? '—' : v);
const cap = (c) => c[0].toUpperCase() + c.slice(1);
const when = (ms) => (ms == null ? 'Undated' : new Date(ms).toLocaleString());

const confirm = (title, message, onYes) => {
  // react-native-web's Alert ignores buttons, so the browser dialog is the only confirmation there.
  if (Platform.OS === 'web') {
    if (globalThis.confirm?.(`${title}\n\n${message}`)) onYes();
    return;
  }
  Alert.alert(title, message, [{ text: 'Cancel', style: 'cancel' }, { text: 'Delete', style: 'destructive', onPress: onYes }]);
};

export const syncLabel = (meal, { pending, conflict }) => {
  if (meal.source.kind === 'sample') return 'Sample demo data · on this device only, never synced';
  if (conflict) return '⚠ Sync conflict · choose which version to keep below';
  if (meal.ownerStatus === 'pending') return 'Saved on this device · syncs after sign-in';
  if (pending) return 'Saved on this device · waiting to sync';
  return 'Synced';
};

const Version = ({ title, m }) => {
  let t = {};
  try { t = mealTotals(m.items); } catch { /* cloud copy is untrusted; show unknown rather than crash */ }
  return (
    <View style={s.version}>
      <Text style={s.versionTitle}>{title}</Text>
      {m.deletedAt != null && <Text style={s.deleted}>Deleted</Text>}
      <Text style={s.vLine}>{cap(String(m.category || 'unknown'))} · {when(m.loggedAt)}</Text>
      {(Array.isArray(m.items) ? m.items : []).map((it, i) => <Text key={i} style={s.vLine}>• {String(it?.name)} {String(it?.grams)} g</Text>)}
      <Text style={s.vLine}>{fmt(t.calories)} kcal · P {fmt(t.protein)} · C {fmt(t.carbs)} · F {fmt(t.fat)}</Text>
    </View>
  );
};

// Details of one canonical (SQLite) meal: items, totals, sync state, edit, delete and conflict choice.
export default function MealDetails({ repo, meal, conflict, pending, onEdit, onClose, children }) {
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const sample = meal.source.kind === 'sample';

  const run = async (fn) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof RevisionConflictError
        ? 'This meal changed meanwhile (another edit or a cloud sync). Review it and try again.'
        : `Failed: ${e.message}`);
    }
    setBusy(false);
  };

  const remove = () => confirm(
    'Delete this meal?',
    sample ? 'This sample is removed from this device.' : 'It is removed from your diary here and, after syncing, on your other devices.',
    () => run(async () => {
      await repo.deleteMeal(meal.id, meal.revision.rev);
      onClose();
    })
  );

  return (
    <SafeAreaView style={s.container}>
      <ScrollView contentContainerStyle={{ padding: 20 }}>
        <View style={s.header}>
          <Text style={s.title}>{cap(meal.category)} {sample && <Text style={s.sampleBadge}> SAMPLE </Text>}</Text>
          <TouchableOpacity onPress={onClose} style={s.closeBtn} accessibilityLabel="Close">
            <Text style={s.closeText}>✕</Text>
          </TouchableOpacity>
        </View>
        <Text style={s.sub}>{meal.loggedAt == null ? 'Undated meal' : `Eaten at ${when(meal.loggedAt)}`}</Text>
        <Text style={s.sub}>
          {meal.source.kind === 'camera' ? 'From a camera scan (original scan kept unchanged)' : sample ? 'Sample demo data' : 'Entered manually'}
        </Text>
        <Text style={[s.sync, conflict && { color: '#b45309' }]}>{syncLabel(meal, { pending, conflict })}</Text>

        <Text style={s.heading}>Foods</Text>
        {meal.items.map((it) => {
          const n = nutrientsForServing(it.per100g, it.grams);
          return (
            <View key={it.id} style={s.itemRow}>
              <Text style={s.itemName}>{it.name} · {it.grams} g</Text>
              <Text style={s.itemVals}>{fmt(n.calories)} kcal · P {fmt(n.protein)} · C {fmt(n.carbs)} · F {fmt(n.fat)}</Text>
            </View>
          );
        })}
        <View style={s.totals}>
          <Text style={s.totalsText}>
            Total {fmt(meal.totals.calories)} kcal · Protein {fmt(meal.totals.protein)} g · Carbs {fmt(meal.totals.carbs)} g · Fat {fmt(meal.totals.fat)} g
          </Text>
          {Object.values(meal.totals).some((v) => v == null) && <Text style={s.note}>— = unknown (not counted as zero)</Text>}
        </View>

        {conflict && (
          <View style={s.conflict}>
            <Text style={s.heading}>This meal was changed on another device</Text>
            <View style={{ flexDirection: 'row', gap: 8 }}>
              <Version title="This device" m={meal} />
              {conflict.remote ? <Version title="Cloud" m={conflict.remote} /> : <View style={s.version}><Text>Cloud copy missing</Text></View>}
            </View>
            <TouchableOpacity style={[s.btn, { backgroundColor: '#1e3a8a' }]} disabled={busy} onPress={() => run(() => repo.resolveConflict(conflict.id, 'local'))}>
              <Text style={s.btnText}>Keep this device's version</Text>
            </TouchableOpacity>
            {conflict.remote && (
              <TouchableOpacity style={[s.btn, { backgroundColor: '#475569' }]} disabled={busy} onPress={() => run(() => repo.resolveConflict(conflict.id, 'cloud'))}>
                <Text style={s.btnText}>Keep cloud version</Text>
              </TouchableOpacity>
            )}
            <Text style={s.note}>Keeping this device's version is re-checked against the cloud; if it changed again you will be asked again.</Text>
          </View>
        )}

        {error && <Text style={s.error}>{error}</Text>}

        <View style={{ flexDirection: 'row', gap: 8, marginTop: 6 }}>
          <TouchableOpacity style={[s.btn, { flex: 1, backgroundColor: '#10b981' }]} disabled={busy} onPress={onEdit}>
            <Text style={s.btnText}>Edit</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[s.btn, { flex: 1, backgroundColor: '#dc2626' }]} disabled={busy} onPress={remove}>
            <Text style={s.btnText}>Delete</Text>
          </TouchableOpacity>
        </View>

        {children}

        <TouchableOpacity style={[s.btn, { backgroundColor: '#111827', marginBottom: 30 }]} onPress={onClose}>
          <Text style={s.btnText}>Done</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#ffffff' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  title: { fontSize: 20, fontWeight: '800', color: '#111827', flex: 1 },
  sampleBadge: { fontSize: 12, color: '#ffffff', backgroundColor: '#f59e0b', fontWeight: '800' },
  closeBtn: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#e2e8f0', alignItems: 'center', justifyContent: 'center' },
  closeText: { fontSize: 13, fontWeight: 'bold', color: '#475569' },
  sub: { fontSize: 13, color: '#6b7280', marginTop: 4 },
  sync: { fontSize: 13, color: '#047857', marginTop: 6, fontWeight: '600' },
  heading: { fontSize: 16, fontWeight: '700', color: '#374151', marginTop: 16, marginBottom: 8 },
  itemRow: { paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: '#f3f4f6' },
  itemName: { fontSize: 14, fontWeight: '600', color: '#111827' },
  itemVals: { fontSize: 12, color: '#4b5563', marginTop: 2 },
  totals: { backgroundColor: '#ecfdf5', borderRadius: 10, padding: 10, marginTop: 10 },
  totalsText: { fontSize: 13, color: '#065f46', fontWeight: '600' },
  note: { fontSize: 11, color: '#6b7280', fontStyle: 'italic', marginTop: 4 },
  conflict: { backgroundColor: '#fffbeb', borderColor: '#fcd34d', borderWidth: 1, borderRadius: 12, padding: 12, marginTop: 14 },
  version: { flex: 1, backgroundColor: '#ffffff', borderRadius: 8, padding: 8, borderWidth: 1, borderColor: '#e5e7eb' },
  versionTitle: { fontSize: 12, fontWeight: '800', color: '#1e293b', marginBottom: 4 },
  vLine: { fontSize: 12, color: '#334155', marginVertical: 1 },
  deleted: { fontSize: 12, fontWeight: '800', color: '#dc2626' },
  error: { color: '#b91c1c', fontSize: 13, marginTop: 10 },
  btn: { paddingVertical: 12, borderRadius: 12, alignItems: 'center', marginTop: 8 },
  btnText: { color: '#ffffff', fontSize: 14, fontWeight: 'bold' },
});
