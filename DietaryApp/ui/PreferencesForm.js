import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, ScrollView, SafeAreaView, KeyboardAvoidingView, Platform } from 'react-native';
import { NUTRIENTS } from '../domain/meal';
import { DIETS } from '../domain/preferences';
import { RevisionConflictError } from '../data/mealRepository';

const GOAL_FIELDS = { calories: 'Calories (kcal)', protein: 'Protein (g)', carbs: 'Carbs (g)', fat: 'Fat (g)' };
const label = (x) => x.replace('_', '-').replace(/^./, (c) => c.toUpperCase());

// Edits the identity's goals and dietary preferences. Validation lives in domain/preferences.js
// (the repository rejects invalid documents); this form only maps text fields to that document.
export default function PreferencesForm({ repo, prefs, onClose }) {
  const [goals, setGoals] = useState(() => Object.fromEntries(NUTRIENTS.map((n) => [n, prefs.goals[n] == null ? '' : String(prefs.goals[n])])));
  const [diets, setDiets] = useState(prefs.diets);
  const [exclusions, setExclusions] = useState(prefs.exclusions.join(', '));
  const [averageBasis, setAverageBasis] = useState(prefs.averageBasis);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await repo.savePreferences(prefs.rev, {
        goals: Object.fromEntries(NUTRIENTS.map((n) => [n, goals[n].trim() === '' ? null : Number(goals[n].trim().replace(',', '.'))])),
        diets,
        exclusions: exclusions.split(','),
        averageBasis,
      });
      onClose();
    } catch (e) {
      setSaving(false);
      setError(e instanceof RevisionConflictError
        ? 'Your preferences changed meanwhile. Close and reopen to edit the latest version.'
        : e.message.replace(/^Invalid preferences: /, ''));
    }
  };

  const chip = (on, text, onPress) => (
    <TouchableOpacity key={text} style={[s.chip, on && s.chipOn]} onPress={onPress} accessibilityRole="button" accessibilityState={{ selected: on }}>
      <Text style={[s.chipText, on && s.chipTextOn]}>{text}</Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={s.container}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={{ padding: 18 }} keyboardShouldPersistTaps="handled">
          <View style={s.header}>
            <Text style={s.title}>Goals & preferences</Text>
            <TouchableOpacity onPress={onClose} style={s.closeBtn} accessibilityLabel="Cancel">
              <Text style={s.closeText}>✕</Text>
            </TouchableOpacity>
          </View>
          <Text style={s.hint}>Saved on this device only (not synced yet). Goals are your own daily targets, not medical advice.</Text>

          <Text style={s.label}>Daily goals <Text style={s.small}>(leave blank for no goal)</Text></Text>
          {NUTRIENTS.map((n) => (
            <View key={n} style={s.goalRow}>
              <Text style={s.goalName}>{GOAL_FIELDS[n]}</Text>
              <TextInput style={s.input} value={goals[n]} onChangeText={(v) => setGoals((g) => ({ ...g, [n]: v }))} keyboardType="decimal-pad" placeholder="none" accessibilityLabel={`${GOAL_FIELDS[n]} daily goal`} />
            </View>
          ))}

          <Text style={[s.label, { marginTop: 14 }]}>Dietary preferences</Text>
          <View style={s.row}>
            {DIETS.map((d) => chip(diets.includes(d), label(d), () => setDiets((x) => (x.includes(d) ? x.filter((y) => y !== d) : [...x, d]))))}
          </View>
          <Text style={s.small}>Stored as you state them. Meals are not checked against a diet, because ingredients are not recorded.</Text>

          <Text style={[s.label, { marginTop: 14 }]}>Foods to flag <Text style={s.small}>(comma-separated)</Text></Text>
          <TextInput style={[s.input, { flex: 0 }]} value={exclusions} onChangeText={setExclusions} placeholder="e.g. peanut, shrimp" autoCapitalize="none" accessibilityLabel="Foods to flag, comma separated" />
          <Text style={s.small}>Insights point out meals whose food names contain these words.</Text>

          <Text style={[s.label, { marginTop: 14 }]}>Daily averages divide by</Text>
          <View style={s.row}>
            {chip(averageBasis === 'logged_days', 'Days with meals logged', () => setAverageBasis('logged_days'))}
            {chip(averageBasis === 'calendar_days', 'Every calendar day', () => setAverageBasis('calendar_days'))}
          </View>

          {error && <Text style={s.error}>{error}</Text>}
          <TouchableOpacity style={[s.saveBtn, saving && { opacity: 0.6 }]} onPress={save} disabled={saving}>
            <Text style={s.saveText}>Save</Text>
          </TouchableOpacity>
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
  hint: { fontSize: 12, color: '#6b7280', fontStyle: 'italic', marginBottom: 12 },
  label: { fontSize: 14, fontWeight: '700', color: '#374151', marginBottom: 6 },
  small: { fontSize: 12, fontWeight: '400', color: '#6b7280', marginTop: 4 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  goalRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  goalName: { flex: 1, fontSize: 14, color: '#334155' },
  input: { flex: 1, borderWidth: 1, borderColor: '#cbd5e1', borderRadius: 8, paddingHorizontal: 8, paddingVertical: 8, fontSize: 14, marginBottom: 6, color: '#111827' },
  chip: { paddingVertical: 8, paddingHorizontal: 12, borderRadius: 18, borderWidth: 1, borderColor: '#cbd5e1' },
  chipOn: { backgroundColor: '#10b981', borderColor: '#10b981' },
  chipText: { fontSize: 13, color: '#334155', fontWeight: '600' },
  chipTextOn: { color: '#ffffff' },
  error: { color: '#b91c1c', fontSize: 13, marginTop: 10 },
  saveBtn: { backgroundColor: '#111827', paddingVertical: 14, borderRadius: 12, alignItems: 'center', marginTop: 16, marginBottom: 30 },
  saveText: { color: '#ffffff', fontSize: 15, fontWeight: 'bold' },
});
