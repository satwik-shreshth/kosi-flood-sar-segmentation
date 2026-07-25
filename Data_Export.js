// ── DATE RANGES ──
var PRE_START  = '2024-08-01';
var PRE_END    = '2024-08-20';
var POST_START = '2024-09-01';
var POST_END   = '2024-09-20';

var ROI = ee.Geometry.Rectangle([86.0, 25.5, 87.5, 26.8]);

// ── FILTER SENTINEL-1 ──
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(ROI)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV', 'VH']);

// ── PRE & POST COMPOSITES ──
var pre_flood  = s1.filterDate(PRE_START, PRE_END).mean().clip(ROI);
var post_flood = s1.filterDate(POST_START, POST_END).mean().clip(ROI);

// ── CHECK IMAGE COUNT ──
print('Pre-flood images  :', s1.filterDate(PRE_START, PRE_END).size());
print('Post-flood images :', s1.filterDate(POST_START, POST_END).size());

// ── VISUALIZE ──
var sarVis = {bands: ['VV'], min: -25, max: 0, palette: ['black','white']};

Map.centerObject(ROI, 9);
Map.addLayer(pre_flood,  sarVis, 'Pre-flood SAR (Aug)');
Map.addLayer(post_flood, sarVis, 'Post-flood SAR (Sep)');
// ── LOG RATIO CHANGE DETECTION ──
// In SAR: flooded areas show DECREASE in backscatter post-flood
// Log ratio = post - pre (in dB) → negative values = new water

var log_ratio = post_flood.select('VV')
  .subtract(pre_flood.select('VV'))
  .rename('log_ratio');

// ── OTSU THRESHOLDING (automated) ──
// Compute histogram to find optimal threshold
var histogram = log_ratio.reduceRegion({
  reducer: ee.Reducer.histogram(255, 0.1),
  geometry: ROI,
  scale: 10,
  maxPixels: 1e10
});

print('Log Ratio Stats:', log_ratio.reduceRegion({
  reducer: ee.Reducer.mean()
    .combine(ee.Reducer.min(), '', true)
    .combine(ee.Reducer.max(), '', true)
    .combine(ee.Reducer.stdDev(), '', true),
  geometry: ROI,
  scale: 10,
  maxPixels: 1e10
}));

// ── SIMPLE THRESHOLD FLOOD MASK ──
// Pixels with log_ratio < -2 dB = likely flooded
var flood_mask = log_ratio.lt(-2).rename('flood');

// ── JRC PERMANENT WATER (to exclude rivers always there) ──
var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('seasonality');

// Permanent water = present >10 months/year
var permanent_water = jrc.gte(10);

// Flood only = new water - permanent water
var flood_only = flood_mask
  .where(permanent_water, 0)
  .selfMask();

// ── VISUALIZE ──
Map.addLayer(log_ratio, 
  {min: -5, max: 5, palette: ['blue','white','red']}, 
  'Log Ratio (blue=decrease=flood)');

Map.addLayer(flood_mask.selfMask(), 
  {palette: ['cyan']}, 
  'Raw Flood Mask');

Map.addLayer(flood_only, 
  {palette: ['orange']}, 
  'Flood Only (excl. permanent water)');

Map.addLayer(permanent_water.selfMask(),
  {palette: ['blue']},
  'Permanent Water (JRC)');

print('Flood pixel count:', flood_only.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: ROI,
  scale: 10,
  maxPixels: 1e10
}));

// ── REFINED FLOOD LABEL ──
// Using stricter threshold based on your stdDev (~2.96)
// mean - 1*stdDev = -0.08 - 2.96 = ~-3 dB → good flood threshold

var FLOOD_THRESHOLD = -3;

var flood_refined = log_ratio.lt(FLOOD_THRESHOLD)
  .rename('flood')
  .clip(ROI);

// Remove permanent water
var flood_final = flood_refined
  .where(permanent_water, 0)
  .rename('flood_label');

// ── BUILD SAR FEATURE STACK ──
// Channels: VV_post, VH_post, VV_pre, VH_pre, log_ratio, elevation
var dem = ee.Image('USGS/SRTMGL1_003').clip(ROI).rename('elevation');
var flat_mask = dem.lt(200);   // remove hilly areas

var sar_stack = post_flood
  .rename(['VV_post', 'VH_post'])
  .addBands(pre_flood.rename(['VV_pre', 'VH_pre']))
  .addBands(log_ratio)
  .addBands(dem)
  .updateMask(flat_mask)
  .clip(ROI);

// ── PRINT BAND NAMES TO CONFIRM ──
print('SAR Stack bands:', sar_stack.bandNames());
print('Label bands    :', flood_final.bandNames());

// ── VISUALIZE FINAL LABEL ──
Map.addLayer(flood_refined.selfMask(),
  {palette: ['red']}, 'Flood Refined (threshold -3dB)');

Map.addLayer(flood_final.selfMask(),
  {palette: ['magenta']}, 'Final Flood Label (no permanent water)');

Map.addLayer(sar_stack,
  {bands: ['VV_post'], min: -25, max: 0}, 'SAR Stack VV_post');

// ── EXPORT SAR STACK ──
Export.image.toDrive({
  image: sar_stack.toFloat(),
  description: 'Bihar_Kosi_SAR_Stack_2024',
  folder: 'flood_mapping',
  fileNamePrefix: 'Bihar_Kosi_SAR_Stack',
  region: ROI,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// ── EXPORT FLOOD LABEL ──
Export.image.toDrive({
  image: flood_final.toFloat(),
  description: 'Bihar_Kosi_Flood_Label_2024',
  folder: 'flood_mapping',
  fileNamePrefix: 'Bihar_Kosi_Flood_Label',
  region: ROI,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

print('Export tasks submitted! Check Tasks tab →');