"""
Test the exact example from PDF
"""
import sys
sys.path.insert(0, '/media/liuyu/DataDrive/WWW2026_demo/stindex')

from stindex import STIndexExtractor
from stindex.models.schemas import ExtractionConfig

# Exact text from PDF
text = """On March 15, 2022, a strong cyclone hit the coastal areas near Broome, Western Australia and later moved inland towards Fitzroy Crossing by March 17."""

print("=" * 80)
print("PDF Example Validation Test")
print("=" * 80)
print(f"\nInput text:\n{text}\n")

# Create extractor with local model
config = ExtractionConfig(
    llm_provider="local",
    model_name="Qwen/Qwen3-8B",
    enable_temporal=True,
    enable_spatial=True,
)

extractor = STIndexExtractor(config=config)
result = extractor.extract(text)

print("=" * 80)
print("TEMPORAL OUTPUT")
print("=" * 80)
print("\nPDF Expected:")
print("  • March 15, 2022 → 2022-03-15")
print("  • March 17 → 2022-03-17")

print("\nActual Output:")
for entity in result.temporal_entities:
    print(f"  • '{entity.text}' → {entity.normalized}")
    
print("\n" + "=" * 80)
print("SPATIAL OUTPUT")
print("=" * 80)
print("\nPDF Expected:")
print("  • Broome, Western Australia → (17.9614° S, 122.2359° E)")
print("  • Fitzroy Crossing, Western Australia → (18.1976° S, 125.5669° E)")

print("\nActual Output:")
for entity in result.spatial_entities:
    lat_str = f"{abs(entity.latitude):.4f}° {'S' if entity.latitude < 0 else 'N'}"
    lon_str = f"{abs(entity.longitude):.4f}° {'E' if entity.longitude > 0 else 'W'}"
    print(f"  • '{entity.text}' → ({lat_str}, {lon_str})")

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

# Validate temporal
temporal_correct = 0
temporal_total = 2
for entity in result.temporal_entities:
    if entity.text == "March 15, 2022" and entity.normalized == "2022-03-15":
        temporal_correct += 1
    elif entity.text == "March 17" and entity.normalized == "2022-03-17":
        temporal_correct += 1

# Validate spatial
spatial_correct = 0
spatial_total = 2
for entity in result.spatial_entities:
    if "Broome" in entity.text and -18.5 < entity.latitude < -17.5 and 121.5 < entity.longitude < 123:
        spatial_correct += 1
        print(f"  ✓ Broome location correct (within expected range)")
    if "Fitzroy" in entity.text and -18.5 < entity.latitude < -17.5 and 125 < entity.longitude < 126:
        spatial_correct += 1
        print(f"  ✓ Fitzroy Crossing location correct (within expected range)")

print(f"\nTemporal: {temporal_correct}/{temporal_total} correct")
print(f"Spatial: {spatial_correct}/{spatial_total} correct")
print(f"\nOverall: {temporal_correct + spatial_correct}/{temporal_total + spatial_total} correct")

if temporal_correct == temporal_total and spatial_correct == spatial_total:
    print("\n✅ FULLY COMPLIANT WITH PDF REQUIREMENTS")
else:
    print(f"\n⚠️ Partial compliance: {((temporal_correct + spatial_correct)/(temporal_total + spatial_total))*100:.1f}%")

print("=" * 80)
