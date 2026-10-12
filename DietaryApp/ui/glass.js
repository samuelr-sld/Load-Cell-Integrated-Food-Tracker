import React, { useSyncExternalStore } from 'react';
import { AccessibilityInfo, Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import Svg, { Circle, Path } from 'react-native-svg';
import { color, glass, MIN_TOUCH, radius, space, tone, type } from './theme';

// Shared glass kit. Blur is real only on iOS (expo-blur on Android needs a BlurTargetView) and is dropped for the
// opaque `solid` fill when the user turns on Reduce Transparency, so text never depends on the blur to be readable.
let solid = Platform.OS !== 'ios';
const subs = new Set();
const setSolid = (reduce) => { solid = Platform.OS !== 'ios' || reduce; subs.forEach((f) => f()); };
if (Platform.OS === 'ios') {
  AccessibilityInfo.isReduceTransparencyEnabled().then(setSolid).catch(() => {});
  AccessibilityInfo.addEventListener('reduceTransparencyChanged', setSolid);
}
const useSolid = () => useSyncExternalStore((f) => { subs.add(f); return () => subs.delete(f); }, () => solid);

export const fmt = (v) => (v == null ? '—' : v);
export const cap = (c) => c[0].toUpperCase() + c.slice(1);

// Page background: gradient plus fixed blobs placed where the Today card, list and dock sit, so their crisp edges
// pass under the glass (and move under it while the list scrolls). That edge is what makes the frost visible.
export const Backdrop = ({ children }) => (
  <LinearGradient colors={[color.bgTop, color.bgMid, color.bgBottom]} style={s.flex}>
    <View pointerEvents="none" style={[s.blob, { top: 60, right: -70, width: 260, height: 260, backgroundColor: color.blobMint }]} />
    <View pointerEvents="none" style={[s.blob, { top: 340, left: -110, width: 280, height: 280, backgroundColor: color.blobSage }]} />
    <View pointerEvents="none" style={[s.blob, { bottom: 40, right: -60, width: 240, height: 240, backgroundColor: color.blobLime }]} />
    {children}
  </LinearGradient>
);

// A frosted surface. `style` positions it (margins, flex); `padded={false}` lets children run edge to edge.
export const Glass = ({ variant = 'card', padded = true, style, children, ...rest }) => {
  const v = glass[variant];
  const opaque = useSolid();
  return (
    <View style={[{ borderRadius: v.radius }, glass.shadow, style]} {...rest}>
      <View style={[s.clip, { borderRadius: v.radius }]}>
        {opaque ? <View pointerEvents="none" style={[StyleSheet.absoluteFill, { backgroundColor: v.solid }]} /> : <>
          <BlurView intensity={v.intensity} tint={v.tint} style={StyleSheet.absoluteFill} />
          <LinearGradient pointerEvents="none" colors={v.fill} start={{ x: 0, y: 0 }} end={{ x: 0.3, y: 1 }} style={StyleSheet.absoluteFill} />
        </>}
        <View style={padded && s.pad}>{children}</View>
      </View>
    </View>
  );
};

const BTN = {
  primary: { bg: color.emerald, pressed: color.emeraldPressed, fg: color.onEmerald, border: color.emerald },
  secondary: { bg: 'rgba(255,255,255,0.75)', pressed: 'rgba(255,255,255,0.95)', fg: color.emerald, border: 'rgba(6,95,70,0.35)' },
  ghost: { bg: 'transparent', pressed: color.track, fg: color.emerald, border: 'transparent' },
  danger: { bg: 'transparent', pressed: tone.danger.bg, fg: color.danger, border: 'transparent' },
};

export const Button = ({ label, onPress, variant = 'primary', icon, disabled, style, accessibilityHint }) => {
  const v = BTN[variant];
  return (
    <Pressable
      onPress={onPress} disabled={disabled} accessibilityRole="button" accessibilityLabel={label} accessibilityHint={accessibilityHint}
      accessibilityState={{ disabled: !!disabled }}
      style={({ pressed }) => [s.btn, { backgroundColor: pressed ? v.pressed : v.bg, borderColor: v.border, opacity: disabled ? 0.5 : 1 }, style]}
    >
      {icon && <Icon name={icon} color={v.fg} />}
      <Text style={[type.headline, { color: v.fg }]} numberOfLines={1}>{label}</Text>
    </Pressable>
  );
};

// Floating bottom dock: secondary actions flank one prominent primary action. Sits above the home indicator.
export const Dock = ({ bottom, children }) => (
  <View pointerEvents="box-none" style={[s.dock, { bottom: Math.max(bottom, space.md) }]}>
    <Glass variant="bar" padded={false} style={glass.lift}><View style={s.dockRow}>{children}</View></Glass>
  </View>
);

export const DockButton = ({ label, icon, onPress, primary }) => (
  <Pressable
    onPress={onPress} accessibilityRole="button" accessibilityLabel={label}
    style={({ pressed }) => [s.dockBtn, primary && s.dockPrimary, pressed && { opacity: 0.7 }]}
  >
    <Icon name={icon} color={primary ? color.onEmerald : color.emerald} size={primary ? 22 : 24} />
    <Text style={[type.caption, { color: primary ? color.onEmerald : color.emerald, fontWeight: '600' }]} numberOfLines={1}>{label}</Text>
  </Pressable>
);

const ICON = {
  plus: 'M12 5v14M5 12h14',
  chat: 'M4 5h16v11H10l-4 4v-4H4z',
  chart: 'M5 20V11M12 20V4M19 20v-6',
  close: 'M6 6l12 12M18 6L6 18',
};
export const Icon = ({ name, color: c = color.ink, size = 22 }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" accessibilityElementsHidden importantForAccessibility="no-hide-descendants">
    <Path d={ICON[name]} stroke={c} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" fill="none" />
  </Svg>
);

export const StatusPill = ({ tone: t = 'neutral', label, onPress }) => {
  const c = tone[t];
  const body = (
    <View style={[s.pill, { backgroundColor: c.bg }]}>
      <View style={[s.dot, { backgroundColor: c.fg }]} />
      <Text style={[type.caption, { color: c.fg, fontWeight: '600' }]} numberOfLines={1}>{label}</Text>
    </View>
  );
  return onPress
    ? <Pressable onPress={onPress} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }} accessibilityRole="button" accessibilityLabel={label}>{body}</Pressable>
    : <View accessibilityLabel={label}>{body}</View>;
};

// Inline tag on a meal card: "Estimated", "Corrected", "Sample".
export const Tag = ({ label, tone: t = 'neutral' }) => (
  <View style={[s.tag, { backgroundColor: tone[t].bg }]}><Text style={[type.caption, { color: tone[t].fg, fontWeight: '700' }]}>{label}</Text></View>
);

export const Banner = ({ tone: t = 'warn', title, body, action }) => (
  <Glass padded={false} style={s.bannerWrap}>
    <View style={[s.banner, { borderLeftColor: tone[t].fg }]} accessibilityRole="alert">
      <View style={s.flex}>
        <Text style={[type.footnote, { color: tone[t].fg, fontWeight: '700' }]}>{title}</Text>
        {!!body && <Text style={[type.footnote, { color: color.inkMuted, marginTop: 2 }]}>{body}</Text>}
      </View>
      {action && <Button variant="ghost" label={action.label} onPress={action.onPress} />}
    </View>
  </Glass>
);

// Calorie ring. `fraction` 0..1 from homeView; children render in the centre.
export const Ring = ({ fraction, over, size = 148, stroke = 14, children }) => {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  return (
    <View style={{ width: size, height: size }}>
      <Svg width={size} height={size} style={{ transform: [{ rotate: '-90deg' }] }}>
        <Circle cx={size / 2} cy={size / 2} r={r} stroke={color.track} strokeWidth={stroke} fill="none" />
        {fraction > 0 && (
          <Circle cx={size / 2} cy={size / 2} r={r} stroke={over ? color.warn : color.emerald} strokeWidth={stroke} fill="none"
            strokeLinecap="round" strokeDasharray={`${c * fraction} ${c}`} />
        )}
      </Svg>
      <View style={[StyleSheet.absoluteFill, s.center]}>{children}</View>
    </View>
  );
};

export const ProgressBar = ({ fraction, tint, over }) => (
  <View style={s.barTrack} accessibilityElementsHidden importantForAccessibility="no-hide-descendants">
    <View style={[s.barFill, { width: `${Math.round(fraction * 100)}%`, backgroundColor: over ? color.warn : tint }]} />
  </View>
);

export const SheetHeader = ({ title, onClose }) => (
  <View style={s.sheetHeader}>
    <Text style={[type.title, s.flex, { color: color.ink }]} numberOfLines={1} accessibilityRole="header">{title}</Text>
    <Pressable onPress={onClose} accessibilityRole="button" accessibilityLabel="Close" style={s.closeBtn}>
      <Icon name="close" color={color.inkMuted} size={18} />
    </Pressable>
  </View>
);

export const EmptyState = ({ title, body, action, secondary }) => (
  <Glass style={s.emptyWrap}>
    <View style={s.center}>
      <Text style={[type.headline, { color: color.ink, textAlign: 'center' }]}>{title}</Text>
      <Text style={[type.body, { color: color.inkMuted, textAlign: 'center', marginTop: space.xs }]}>{body}</Text>
      {action && <Button label={action.label} onPress={action.onPress} icon="plus" style={{ marginTop: space.lg, alignSelf: 'stretch' }} />}
      {secondary && <Button variant="ghost" label={secondary.label} onPress={secondary.onPress} style={{ marginTop: space.xs }} />}
    </View>
  </Glass>
);

// Static placeholder (no animation, so nothing to reduce): the shape of the Today card while the diary loads.
export const LoadingCard = () => (
  <Glass variant="hero" accessibilityLabel="Loading your meals" accessibilityRole="progressbar">
    <View style={s.skelRow}>
      <View style={s.skelRing} />
      <View style={s.flex}>
        {[0, 1, 2].map((i) => <View key={i} style={s.skelLine} />)}
      </View>
    </View>
  </Glass>
);

const s = StyleSheet.create({
  flex: { flex: 1 },
  center: { alignItems: 'center', justifyContent: 'center' },
  blob: { position: 'absolute', borderRadius: 999 },
  clip: { overflow: 'hidden', borderWidth: 1, borderColor: glass.border },
  pad: { padding: space.lg },
  btn: { minHeight: MIN_TOUCH + 4, paddingHorizontal: space.xl, borderRadius: radius.pill, borderWidth: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: space.sm },
  dock: { position: 'absolute', left: space.lg, right: space.lg },
  dockRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-around', paddingHorizontal: space.sm, paddingVertical: space.sm },
  dockBtn: { minHeight: 56, minWidth: 84, alignItems: 'center', justifyContent: 'center', gap: 2, borderRadius: radius.lg, paddingHorizontal: space.md },
  dockPrimary: { backgroundColor: color.emerald, minWidth: 112 },
  pill: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 10, paddingVertical: 6, borderRadius: radius.pill, alignSelf: 'flex-start' },
  dot: { width: 7, height: 7, borderRadius: 4 },
  tag: { paddingHorizontal: 8, paddingVertical: 2, borderRadius: radius.pill },
  bannerWrap: { marginBottom: space.md },
  banner: { flexDirection: 'row', alignItems: 'center', gap: space.sm, padding: space.md, borderLeftWidth: 4 },
  barTrack: { height: 8, borderRadius: 4, backgroundColor: color.track, overflow: 'hidden' },
  barFill: { height: 8, borderRadius: 4 },
  sheetHeader: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.sm },
  closeBtn: { width: MIN_TOUCH, height: MIN_TOUCH, borderRadius: MIN_TOUCH / 2, backgroundColor: color.track, alignItems: 'center', justifyContent: 'center' },
  emptyWrap: { marginTop: space.lg },
  skelRow: { flexDirection: 'row', alignItems: 'center', gap: space.xl },
  skelRing: { width: 120, height: 120, borderRadius: 60, borderWidth: 14, borderColor: color.track },
  skelLine: { height: 14, borderRadius: 7, backgroundColor: color.track, marginVertical: 8 },
});
