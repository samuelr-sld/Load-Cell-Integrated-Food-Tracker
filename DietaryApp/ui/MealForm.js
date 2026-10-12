import React, { useState } from 'react';
import {
  StyleSheet, Text, View, TextInput, TouchableOpacity, ScrollView, Switch, SafeAreaView,
  KeyboardAvoidingView, Platform, ActivityIndicator,
} from 'react-native';
import { toDraft, parseDraft, emptyItem, FORM_CATEGORIES } from '../domain/mealForm';
import { RevisionConflictError, MealNotFoundError, DuplicateScanError } from '../data/mealRepository';
import { scanToDraft, scanSnapshot, applyCandidate } from '../domain/scan';

const fmt = (v) => (v == null ? '—' : v);
const NUTRIENT_FIELDS = [['calories', 'kcal'], ['protein', 'Protein g'], ['carbs', 'Carbs g'], ['fat', 'Fat g']];

// Create (meal = null) or edit a canonical meal. Mount with a `key` per meal so the draft resets.
// `scan` (a laptop load-cell scan) pre-fills a NEW meal for review; onScanDone('saved' | 'rejected') reports the user's decision.
export default function MealForm({ repo, meal, onClose, scan = null, onScanDone }) {
  const [draft, setDraft] = useState(() => (scan ? scanToDraft(scan) : toDraft(meal)));
  const [showErrors, setShowErrors] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);

  const parsed = parseDraft(draft, meal);
  const set = (patch) => setDraft((d) => ({ ...d, ...patch }));
  const setItem = (i, patch) => setDraft((d) => ({ ...d, items: d.items.map((it, j) => (j === i ? { ...it, ...patch } : it)) }));

  const save = async () => {
    setShowErrors(true);
    setSaveError(null);
    if (parsed.errors.length) return;
    setSaving(true);
    try {
      if (meal) await repo.updateMeal(meal.id, meal.revision.rev, parsed.patch);
      else if (scan) {
        // source.ref = scan_id is unique in SQLite, so confirming the same scan twice can never make a second meal.
        await repo.createMeal({ ...parsed.input, source: { kind: 'camera', ref: scan.scan_id }, originalScan: scanSnapshot(scan) })
          .catch((e) => { if (!(e instanceof DuplicateScanError)) throw e; });
        onScanDone?.('saved');
      } else await repo.createMeal({ ...parsed.input, source: { kind: 'manual' } });
      onClose();
    } catch (e) {
      setSaving(false);
      if (e instanceof RevisionConflictError) {
        setSaveError('This meal changed after you opened it (another edit or a cloud sync). Your changes were not saved. Close and reopen the meal to edit the latest version.');
      } else if (e instanceof MealNotFoundError) {
        setSaveError('This meal no longer exists on this device.');
      } else {
        setSaveError(`Could not save: ${e.message}`);
      }
    }
  };

  return (
    <SafeAreaView style={s.container}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={{ padding: 18 }} keyboardShouldPersistTaps="handled">
          <View style={s.header}>
            <Text style={s.title}>{meal ? 'Edit meal' : scan ? 'Review scanned meal' : 'New meal'}</Text>
            <TouchableOpacity onPress={onClose} style={s.closeBtn} accessibilityLabel="Cancel">
              <Text style={s.closeText}>✕</Text>
            </TouchableOpacity>
          </View>
          {scan && (
            <Text style={s.hint}>
              {scan.status === 'error'
                ? `The scan failed (${scan.error?.message ?? 'unknown error'}). You can still enter the meal by hand; the scale read ${scan.measured_total_grams} g.`
                : `Scale: ${scan.measured_total_grams} g${scan.tare_g ? ` (${scan.net_grams} g after the ${scan.tare_g} g plate)` : ''}. Check the food${scan.foods.length > 1 ? ' and enter the grams of each food' : ''}, then save.`}
            </Text>
          )}
          {meal?.source.kind === 'camera' && (
            <Text style={s.hint}>Scanned meal: the original scan result is kept unchanged; you are editing your corrected copy.</Text>
          )}

          <Text style={s.label}>Meal</Text>
          <View style={s.row}>
            {FORM_CATEGORIES.map((c) => (
              <TouchableOpacity
                key={c}
                style={[s.chip, draft.category === c && s.chipOn]}
                onPress={() => set({ category: c })}
                accessibilityRole="button"
                accessibilityState={{ selected: draft.category === c }}
              >
                <Text style={[s.chipText, draft.category === c && s.chipTextOn]}>{c[0].toUpperCase() + c.slice(1)}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <View style={[s.row, { justifyContent: 'space-between', marginTop: 14 }]}>
            <Text style={s.label}>Date and time</Text>
            <View style={s.row}>
              <Text style={s.small}>Undated </Text>
              <Switch value={draft.undated} onValueChange={(v) => set({ undated: v })} accessibilityLabel="Undated meal" />
            </View>
          </View>
          {!draft.undated && (
            <View style={s.row}>
              <TextInput style={[s.input, { flex: 1.4 }]} value={draft.date} onChangeText={(v) => set({ date: v })} placeholder="YYYY-MM-DD" accessibilityLabel="Date" />
              <TextInput style={[s.input, { flex: 1 }]} value={draft.time} onChangeText={(v) => set({ time: v })} placeholder="h:mm AM/PM" accessibilityLabel="Time" />
              <TouchableOpacity style={s.smallBtn} onPress={() => { const n = toDraft(null); set({ date: n.date, time: n.time }); }} accessibilityLabel="Set date and time to now">
                <Text style={s.smallBtnText}>Now</Text>
              </TouchableOpacity>
            </View>
          )}

          <Text style={[s.label, { marginTop: 14 }]}>Foods <Text style={s.small}>(nutrients per 100 g; leave blank if unknown)</Text></Text>
          {draft.items.map((it, i) => (
            <View key={i} style={s.itemCard}>
              <View style={s.row}>
                <TextInput style={[s.input, { flex: 2 }]} value={it.name} onChangeText={(v) => setItem(i, { name: v })} placeholder="Food name" accessibilityLabel={`Food ${i + 1} name`} />
                <TextInput style={[s.input, { flex: 1 }]} value={it.grams} onChangeText={(v) => setItem(i, { grams: v })} placeholder="grams" keyboardType="decimal-pad" accessibilityLabel={`Food ${i + 1} grams`} />
              </View>
              {scan?.foods.find((f) => f.region_id === it.regionId)?.candidates.map((c) => (
                <TouchableOpacity key={`${c.taxonomy}:${c.label}`} style={[s.chip, { marginBottom: 6 }, it.name === c.label && s.chipOn]} onPress={() => setItem(i, applyCandidate(it, c))}>
                  <Text style={[s.chipText, it.name === c.label && s.chipTextOn]}>{c.label} {Math.round(c.confidence * 100)}%</Text>
                </TouchableOpacity>
              ))}
              <View style={s.row}>
                {NUTRIENT_FIELDS.map(([k, ph]) => (
                  <TextInput key={k} style={[s.input, { flex: 1 }]} value={it[k]} onChangeText={(v) => setItem(i, { [k]: v })} placeholder={ph} keyboardType="decimal-pad" accessibilityLabel={`Food ${i + 1} ${k} per 100 grams`} />
                ))}
              </View>
              {draft.items.length > 1 && (
                <TouchableOpacity onPress={() => set({ items: draft.items.filter((_, j) => j !== i) })}>
                  <Text style={s.remove}>Remove food</Text>
                </TouchableOpacity>
              )}
            </View>
          ))}
          <TouchableOpacity style={s.addBtn} onPress={() => set({ items: [...draft.items, emptyItem()] })}>
            <Text style={s.addText}>＋ Add another food</Text>
          </TouchableOpacity>

          <View style={s.totals}>
            <Text style={s.totalsText}>
              Meal total: {parsed.totals ? `${fmt(parsed.totals.calories)} kcal · P ${fmt(parsed.totals.protein)} g · C ${fmt(parsed.totals.carbs)} g · F ${fmt(parsed.totals.fat)} g` : 'fix the food fields to see totals'}
            </Text>
          </View>

          {showErrors && parsed.errors.map((e) => <Text key={e} style={s.error}>• {e}</Text>)}
          {saveError && <Text style={s.error}>{saveError}</Text>}

          <TouchableOpacity style={[s.saveBtn, saving && { opacity: 0.6 }]} onPress={save} disabled={saving}>
            {saving ? <ActivityIndicator color="#fff" /> : <Text style={s.saveText}>Save on this device</Text>}
          </TouchableOpacity>
          {scan && (
            <TouchableOpacity style={{ alignItems: 'center', paddingBottom: 30 }} onPress={() => { onScanDone?.('rejected'); onClose(); }}>
              <Text style={s.remove}>Reject this scan</Text>
            </TouchableOpacity>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#ffffff' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  title: { fontSize: 20, fontWeight: '800', color: '#111827' },
  closeBtn: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#e2e8f0', alignItems: 'center', justifyContent: 'center' },
  closeText: { fontSize: 13, fontWeight: 'bold', color: '#475569' },
  hint: { fontSize: 12, color: '#6b7280', fontStyle: 'italic', marginBottom: 8 },
  label: { fontSize: 14, fontWeight: '700', color: '#374151', marginBottom: 6 },
  small: { fontSize: 12, fontWeight: '400', color: '#6b7280' },
  row: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  chip: { paddingVertical: 8, paddingHorizontal: 12, borderRadius: 18, borderWidth: 1, borderColor: '#cbd5e1' },
  chipOn: { backgroundColor: '#10b981', borderColor: '#10b981' },
  chipText: { fontSize: 13, color: '#334155', fontWeight: '600' },
  chipTextOn: { color: '#ffffff' },
  input: { borderWidth: 1, borderColor: '#cbd5e1', borderRadius: 8, paddingHorizontal: 8, paddingVertical: 8, fontSize: 14, marginBottom: 6, color: '#111827' },
  smallBtn: { paddingVertical: 8, paddingHorizontal: 10, borderRadius: 8, backgroundColor: '#e2e8f0', marginBottom: 6 },
  smallBtnText: { fontSize: 13, fontWeight: '600', color: '#334155' },
  itemCard: { backgroundColor: '#f9fafb', borderRadius: 10, padding: 10, marginBottom: 8 },
  remove: { color: '#dc2626', fontSize: 13, fontWeight: '600', marginTop: 2 },
  addBtn: { paddingVertical: 8 },
  addText: { color: '#10b981', fontSize: 14, fontWeight: '700' },
  totals: { backgroundColor: '#ecfdf5', borderRadius: 10, padding: 10, marginVertical: 10 },
  totalsText: { fontSize: 13, color: '#065f46', fontWeight: '600' },
  error: { color: '#b91c1c', fontSize: 13, marginVertical: 2 },
  saveBtn: { backgroundColor: '#111827', paddingVertical: 14, borderRadius: 12, alignItems: 'center', marginTop: 12, marginBottom: 30 },
  saveText: { color: '#ffffff', fontSize: 15, fontWeight: 'bold' },
});
