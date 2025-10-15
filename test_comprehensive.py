"""
Comprehensive Test Suite for STIndex
Directly evaluates system capabilities without predefined answers
"""

import sys
from pathlib import Path

# Add project root to path (generic approach)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stindex import STIndexExtractor
from stindex.models.schemas import ExtractionConfig

# Test configuration
config = ExtractionConfig(
    llm_provider="local",
    model_name="Qwen/Qwen3-8B",
    enable_temporal=True,
    enable_spatial=True,
)

print("=" * 100)
print("STIndex Comprehensive Capability Test")
print("=" * 100)
print("\nLoading model...")

extractor = STIndexExtractor(config=config)

print("✓ Model loaded\n")

# Test results tracking
test_results = []

def run_test(category: str, name: str, text: str):
    """Run a test and display results."""
    print(f"\n{'─' * 100}")
    print(f"[{category}] {name}")
    print(f"{'─' * 100}")
    print(f"Input: {text}")
    print()

    try:
        result = extractor.extract(text)

        # Display temporal results
        if result.temporal_entities:
            print(f"时间实体 ({len(result.temporal_entities)} 个):")
            for entity in result.temporal_entities:
                print(f"  • '{entity.text}' → {entity.normalized} [{entity.temporal_type.value}]")
        else:
            print("时间实体: 无")

        print()

        # Display spatial results
        if result.spatial_entities:
            print(f"空间实体 ({len(result.spatial_entities)} 个):")
            for entity in result.spatial_entities:
                lat_str = f"{abs(entity.latitude):.4f}° {'S' if entity.latitude < 0 else 'N'}"
                lon_str = f"{abs(entity.longitude):.4f}° {'E' if entity.longitude > 0 else 'W'}"
                print(f"  • '{entity.text}' → ({lat_str}, {lon_str})")
        else:
            print("空间实体: 无")

        test_results.append({
            "category": category,
            "name": name,
            "temporal_count": len(result.temporal_entities),
            "spatial_count": len(result.spatial_entities),
            "status": "success"
        })

    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        test_results.append({
            "category": category,
            "name": name,
            "status": "error",
            "error": str(e)
        })

# =============================================================================
# TEMPORAL TESTS
# =============================================================================

print("\n" + "=" * 100)
print("第1部分: 时间提取能力测试")
print("=" * 100)

run_test("时间", "1.1 显式年份的绝对日期",
    "The project started on January 15, 2020, was paused on March 20, 2021, and resumed on September 5, 2022.")

run_test("时间", "1.2 无年份日期 - 年份推断",
    "In 2023, the conference began on March 10. The workshop was on March 11, and the closing ceremony happened on March 12.")

run_test("时间", "1.3 日期区间",
    "The exhibition will run from May 1, 2024 to May 31, 2024.")

run_test("时间", "1.4 混合日期格式",
    "The event on 2024-06-15 follows the announcement from June 1, 2024, and precedes the deadline of July 15, 2024.")

run_test("时间", "1.5 相对时间表达",
    "The meeting was yesterday, the report is due tomorrow, and the review happens next week.")

run_test("时间", "1.6 带时刻的日期",
    "The webinar starts at 2:00 PM on March 15, 2024.")

run_test("时间", "1.7 持续时间",
    "The training program lasts 3 weeks.")

run_test("时间", "1.8 复杂时间上下文",
    "The study began in January 2020, was interrupted in March 2020 due to COVID-19, and resumed in September 2021.")

run_test("时间", "1.9 跨年区间",
    "The study ran from December 2022 to February 2023.")

run_test("时间", "1.10 历史日期",
    "World War II ended on September 2, 1945. The Berlin Wall fell on November 9, 1989.")

# =============================================================================
# SPATIAL TESTS
# =============================================================================

print("\n" + "=" * 100)
print("第2部分: 空间提取能力测试")
print("=" * 100)

run_test("空间", "2.1 主要世界城市",
    "The tour includes stops in Paris, Tokyo, New York, and Sydney.")

run_test("空间", "2.2 城市+国家上下文",
    "The conference has venues in Berlin, Germany; Toronto, Canada; and Melbourne, Australia.")

run_test("空间", "2.3 歧义地名消歧 - Springfield",
    "Springfield, Illinois is the state capital. Springfield, Massachusetts has a different history.")

run_test("空间", "2.4 州和地区",
    "California, Texas, and Florida are the most populous US states.")

run_test("空间", "2.5 地标",
    "The Eiffel Tower in Paris and the Statue of Liberty in New York are iconic landmarks.")

run_test("空间", "2.6 同国多城市",
    "The Australian tour covers Sydney, Melbourne, Brisbane, Perth, and Adelaide.")

run_test("空间", "2.7 小城镇+州上下文",
    "The study was conducted in Boulder, Colorado and Ann Arbor, Michigan.")

run_test("空间", "2.8 非洲城市",
    "The research team visited Lagos, Nigeria; Nairobi, Kenya; and Cairo, Egypt.")

run_test("空间", "2.9 亚洲城市",
    "The company has offices in Singapore, Seoul, Bangkok, and Mumbai.")

run_test("空间", "2.10 欧洲首都",
    "The summit rotates between Brussels, Geneva, Vienna, and Copenhagen.")

# =============================================================================
# COMBINED TESTS
# =============================================================================

print("\n" + "=" * 100)
print("第3部分: 时空联合提取测试")
print("=" * 100)

run_test("联合", "3.1 新闻报道 - 飓风",
    "On August 29, 2005, Hurricane Katrina made landfall near New Orleans, Louisiana. By August 31, the storm had moved through Mississippi.")

run_test("联合", "3.2 旅行行程",
    "We'll arrive in Rome on June 5, 2024, stay three days, then travel to Florence on June 8.")

run_test("联合", "3.3 会议公告",
    "The International AI Conference will be held in Singapore from September 15-20, 2024.")

run_test("联合", "3.4 历史事件 - 登月",
    "On July 20, 1969, Apollo 11 landed on the Moon.")

run_test("联合", "3.5 商业扩张时间线",
    "The company opened its Tokyo office in March 2020, followed by Shanghai in July 2020.")

run_test("联合", "3.6 科研野外考察",
    "The expedition began in Nairobi, Kenya on February 1, 2023. Researchers spent two weeks in the Serengeti.")

run_test("联合", "3.7 体育赛事",
    "The 2026 FIFA World Cup will be jointly hosted by the United States, Canada, and Mexico from June 11 to July 19, 2026.")

run_test("联合", "3.8 气候事件 - PDF示例",
    "On March 15, 2022, a strong cyclone hit the coastal areas near Broome, Western Australia and later moved inland by March 17.")

run_test("联合", "3.9 政治事件 - 峰会",
    "The G20 Summit took place in Bali, Indonesia on November 15-16, 2022.")

run_test("联合", "3.10 自然灾害时间线",
    "The earthquake struck off the coast of Sumatra on December 26, 2004. The tsunami affected Thailand and Sri Lanka.")

# =============================================================================
# EDGE CASES
# =============================================================================

print("\n" + "=" * 100)
print("第4部分: 边界情况测试")
print("=" * 100)

run_test("边界", "4.1 无时空信息",
    "The algorithm uses machine learning to optimize performance.")

run_test("边界", "4.2 密集信息",
    "Between January 5 and January 10, 2024, the team visited Paris, London, Berlin, and Amsterdam.")

run_test("边界", "4.3 嵌套地点",
    "The office is located in Austin, Texas, United States, near the University of Texas campus.")

run_test("边界", "4.4 中文地名",
    "The meeting will be held in Beijing, China on December 1, 2024.")

run_test("边界", "4.5 多重歧义",
    "Cambridge researchers met with Cambridge colleagues to discuss the Cambridge study.")

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("测试结果汇总")
print("=" * 100)

# Count by category
categories = {}
for result in test_results:
    cat = result["category"]
    if cat not in categories:
        categories[cat] = {"total": 0, "success": 0, "error": 0}
    categories[cat]["total"] += 1
    if result["status"] == "success":
        categories[cat]["success"] += 1
    else:
        categories[cat]["error"] += 1

# Overall stats
total = len(test_results)
success = sum(1 for r in test_results if r["status"] == "success")
errors = total - success

print(f"\n总体统计:")
print(f"  总测试数: {total}")
print(f"  成功: {success} ({100*success/total:.1f}%)")
print(f"  错误: {errors} ({100*errors/total:.1f}%)")

print(f"\n分类统计:")
for cat, stats in categories.items():
    success_rate = 100 * stats["success"] / stats["total"]
    print(f"  [{cat}] {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

# Temporal/Spatial extraction stats
temporal_counts = [r.get("temporal_count", 0) for r in test_results if r["status"] == "success"]
spatial_counts = [r.get("spatial_count", 0) for r in test_results if r["status"] == "success"]

if temporal_counts:
    print(f"\n时间实体提取:")
    print(f"  总计: {sum(temporal_counts)} 个")
    print(f"  平均: {sum(temporal_counts)/len(temporal_counts):.1f} 个/测试")

if spatial_counts:
    print(f"\n空间实体提取:")
    print(f"  总计: {sum(spatial_counts)} 个")
    print(f"  平均: {sum(spatial_counts)/len(spatial_counts):.1f} 个/测试")

print("\n" + "=" * 100)
print("测试完成!")
print("=" * 100)
