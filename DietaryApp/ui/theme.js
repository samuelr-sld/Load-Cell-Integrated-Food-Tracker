// PlateGo "soft glass" tokens. Every glass surface, button and text style reads from here.
// Contrast (ink on the lightest/darkest backdrop stop): ink 15:1, inkMuted 7:1; white on emerald 7.7:1.
export const color = {
  bgTop: '#F8F6EF', bgMid: '#EAF3EC', bgBottom: '#D8E9DE', // warm off-white → mint → sage
  // Saturated, hard-edged shapes behind the glass: blur only shows when there is colour and edges to frost.
  blobMint: 'rgba(72, 187, 140, 0.50)', blobSage: 'rgba(120, 170, 132, 0.45)', blobLime: 'rgba(176, 214, 150, 0.55)',
  ink: '#0F1F17', inkMuted: '#3F5248', inkFaint: '#4A5C52',
  emerald: '#065F46', emeraldPressed: '#044B38', onEmerald: '#FFFFFF',
  warn: '#92400E', danger: '#B91C1C', info: '#1D4ED8', ok: '#047857',
  track: 'rgba(15, 31, 23, 0.10)',
};

export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 };
export const radius = { sm: 12, md: 16, lg: 24, pill: 999 };
export const MIN_TOUCH = 44;

// System font (SF Pro on iPhone). Sizes never drop below 12; numbers use tabular figures so totals don't jitter.
export const type = {
  display: { fontSize: 40, lineHeight: 44, fontWeight: '700', letterSpacing: -0.8, fontVariant: ['tabular-nums'] },
  title: { fontSize: 28, lineHeight: 34, fontWeight: '700', letterSpacing: -0.4 },
  headline: { fontSize: 17, lineHeight: 22, fontWeight: '600' },
  body: { fontSize: 16, lineHeight: 22, fontWeight: '400' },
  number: { fontSize: 17, lineHeight: 22, fontWeight: '600', fontVariant: ['tabular-nums'] },
  footnote: { fontSize: 13, lineHeight: 18, fontWeight: '500' },
  caption: { fontSize: 12, lineHeight: 16, fontWeight: '500' },
};

// Glass surfaces. iOS `intensity` is the fraction of the system material applied (expo-blur animates the effect to
// intensity/100), so low values mean barely any blur. `fill` is a top-lit sheen over the blur; its lower stop is the
// floor the contrast numbers above assume (ink ≥ 11:1, inkMuted ≥ 5.6:1, ignoring the material's own white).
// `solid` replaces blur + fill where blur is unavailable or Reduce Transparency is on.
export const glass = {
  card: { intensity: 85, tint: 'systemUltraThinMaterialLight', fill: ['rgba(255,255,255,0.50)', 'rgba(255,255,255,0.26)'], solid: 'rgba(255,255,255,0.94)', radius: radius.lg },
  hero: { intensity: 95, tint: 'systemThinMaterialLight', fill: ['rgba(255,255,255,0.55)', 'rgba(255,255,255,0.30)'], solid: 'rgba(255,255,255,0.96)', radius: 28 },
  bar: { intensity: 100, tint: 'systemThinMaterialLight', fill: ['rgba(255,255,255,0.55)', 'rgba(255,255,255,0.35)'], solid: 'rgba(255,255,255,0.97)', radius: 32 },
  border: 'rgba(255,255,255,0.75)',
  // Kept faint: on iOS a translucent card's own shadow shows through it and greys the glass.
  shadow: { shadowColor: '#0F3D2B', shadowOpacity: 0.08, shadowRadius: 20, shadowOffset: { width: 0, height: 8 }, elevation: 3 },
  lift: { shadowOpacity: 0.18, shadowRadius: 24, shadowOffset: { width: 0, height: 10 }, elevation: 8 },
};

export const tone = {
  neutral: { fg: color.inkMuted, bg: 'rgba(15,31,23,0.07)' },
  ok: { fg: color.ok, bg: 'rgba(4,120,87,0.12)' },
  info: { fg: color.info, bg: 'rgba(29,78,216,0.10)' },
  warn: { fg: color.warn, bg: 'rgba(146,64,14,0.12)' },
  danger: { fg: color.danger, bg: 'rgba(185,28,28,0.10)' },
};
