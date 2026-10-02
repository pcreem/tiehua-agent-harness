#!/usr/bin/env python3
"""POI Content Auditor - Batch Audit for Tourism Data Quality.

Reads attraction data and uses LLM to check for:
- Hallucination (fabricated information)
- Format consistency (required fields, types, enums)
- Freshness (outdated information)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import Counter

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import create_llm_client
from src.llm.types import LLMMessage


@dataclass
class AuditFinding:
    """Single audit finding."""
    poi_id: str
    field: str
    issue_type: str  # hallucination, format, freshness
    severity: str    # high, medium, low
    message: str
    current_value: Any
    suggested_fix: Optional[str] = None


@dataclass
class AuditReport:
    """Complete POI Audit Report."""
    generated_at: str
    data_source: str
    summary: Dict[str, Any]
    hallucination_findings: List[Dict[str, Any]]
    format_issues: List[Dict[str, Any]]
    freshness_issues: List[Dict[str, Any]]
    by_category: Dict[str, Dict[str, int]]


class POIContentAuditor:
    """Audits POI (Point of Interest) data for quality issues."""

    def __init__(
        self,
        data_path: Path,
        judge_model: str = "poolside/laguna-s-2.1:free",
        api_key: Optional[str] = None,
        freshness_threshold_days: int = 730,  # 2 years
    ):
        self.data_path = Path(data_path)
        self.judge_model = judge_model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.freshness_threshold_days = freshness_threshold_days

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.llm_client = create_llm_client(
            model=judge_model,
            api_key=self.api_key,
            timeout=60.0,
        )

        # Required fields and their types
        self.required_fields = {
            "id": str,
            "name": str,
            "city": str,
            "category": str,
        }

        # Valid enum values
        self.valid_categories = {"sightseeing", "food", "accommodation", "activity", "nature", "service"}
        self.valid_indoor_outdoor = {"indoor", "outdoor", "both", ""}
        self.valid_family_suitability = {"high", "medium", "low", "unknown", ""}

    def audit(self, output_path: Optional[Path] = None, limit: int = 0, skip_hallucination: bool = False) -> AuditReport:
        """Run full audit on POI data."""
        # 1. Load data
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        attractions = data.get("attractions", [])
        metadata = data.get("metadata", {})

        # Apply limit
        if limit > 0:
            attractions = attractions[:limit]

        print(f"Loaded {len(attractions)} POIs from {self.data_path}")

        # 2. Run audits
        hallucination_findings = []
        format_issues = []
        freshness_issues = []

        for i, poi in enumerate(attractions):
            poi_id = poi.get("id", f"index_{i}")
            print(f"Auditing {poi_id} ({i+1}/{len(attractions)}): {poi.get('name', 'unnamed')}")

            # Format audit
            format_issues.extend(self._audit_format(poi, poi_id))

            # Freshness audit
            freshness_issues.extend(self._audit_freshness(poi, poi_id))

            # Hallucination audit (LLM-based)
            if not skip_hallucination:
                hallucination_findings.extend(self._audit_hallucination(poi, poi_id))

        # 3. Aggregate by category
        by_category = self._aggregate_by_category(attractions, hallucination_findings, format_issues, freshness_issues)

        # 4. Build report
        report = AuditReport(
            generated_at=datetime.now().isoformat(),
            data_source=metadata.get("source", "unknown"),
            summary={
                "total_pois": len(attractions),
                "passed": len(attractions) - len(set(f.poi_id for f in hallucination_findings + format_issues + freshness_issues if f.severity == "high")),
                "warnings": len([f for f in hallucination_findings + format_issues + freshness_issues if f.severity == "medium"]),
                "failures": len([f for f in hallucination_findings + format_issues + freshness_issues if f.severity == "high"]),
                "hallucination_count": len(hallucination_findings),
                "format_issues_count": len(format_issues),
                "freshness_issues_count": len(freshness_issues),
            },
            hallucination_findings=[asdict(f) for f in hallucination_findings],
            format_issues=[asdict(f) for f in format_issues],
            freshness_issues=[asdict(f) for f in freshness_issues],
            by_category=by_category,
        )

        # 5. Save report
        if output_path:
            self._save_report(report, output_path)

        return report

    def _audit_format(self, poi: Dict, poi_id: str) -> List[AuditFinding]:
        """Check format consistency."""
        findings = []

        # Check required fields
        for field, expected_type in self.required_fields.items():
            if field not in poi:
                findings.append(AuditFinding(
                    poi_id=poi_id,
                    field=field,
                    issue_type="format",
                    severity="high",
                    message=f"Missing required field: {field}",
                    current_value=None,
                    suggested_fix=f"Add {field} field",
                ))
            elif not isinstance(poi[field], expected_type):
                findings.append(AuditFinding(
                    poi_id=poi_id,
                    field=field,
                    issue_type="format",
                    severity="medium",
                    message=f"Field {field} has wrong type: expected {expected_type.__name__}, got {type(poi[field]).__name__}",
                    current_value=poi[field],
                    suggested_fix=f"Convert to {expected_type.__name__}",
                ))

        # Check category enum
        category = poi.get("category", "")
        if category and category not in self.valid_categories:
            findings.append(AuditFinding(
                poi_id=poi_id,
                field="category",
                issue_type="format",
                severity="medium",
                message=f"Invalid category: '{category}' not in {self.valid_categories}",
                current_value=category,
                suggested_fix=f"Use one of: {', '.join(sorted(self.valid_categories))}",
            ))

        # Check indoor_outdoor
        indoor_outdoor = poi.get("indoor_outdoor", "")
        if indoor_outdoor and indoor_outdoor not in self.valid_indoor_outdoor:
            findings.append(AuditFinding(
                poi_id=poi_id,
                field="indoor_outdoor",
                issue_type="format",
                severity="low",
                message=f"Non-standard indoor_outdoor value: '{indoor_outdoor}'",
                current_value=indoor_outdoor,
                suggested_fix=f"Use one of: {', '.join(sorted(self.valid_indoor_outdoor))}",
            ))

        # Check family_suitability
        family = poi.get("family_suitability", "")
        if family and family not in self.valid_family_suitability:
            findings.append(AuditFinding(
                poi_id=poi_id,
                field="family_suitability",
                issue_type="format",
                severity="low",
                message=f"Non-standard family_suitability value: '{family}'",
                current_value=family,
                suggested_fix=f"Use one of: {', '.join(sorted(self.valid_family_suitability))}",
            ))

        # Check coordinates
        lat = poi.get("latitude")
        lon = poi.get("longitude")
        if lat is not None:
            try:
                lat_f = float(lat)
                if not (-90 <= lat_f <= 90):
                    findings.append(AuditFinding(
                        poi_id=poi_id,
                        field="latitude",
                        issue_type="format",
                        severity="high",
                        message=f"Latitude out of range: {lat_f}",
                        current_value=lat,
                        suggested_fix="Correct latitude to [-90, 90]",
                    ))
            except (ValueError, TypeError):
                findings.append(AuditFinding(
                    poi_id=poi_id,
                    field="latitude",
                    issue_type="format",
                    severity="high",
                    message=f"Invalid latitude format: {lat}",
                    current_value=lat,
                    suggested_fix="Use numeric latitude",
                ))

        if lon is not None:
            try:
                lon_f = float(lon)
                if not (-180 <= lon_f <= 180):
                    findings.append(AuditFinding(
                        poi_id=poi_id,
                        field="longitude",
                        issue_type="format",
                        severity="high",
                        message=f"Longitude out of range: {lon_f}",
                        current_value=lon,
                        suggested_fix="Correct longitude to [-180, 180]",
                    ))
            except (ValueError, TypeError):
                findings.append(AuditFinding(
                    poi_id=poi_id,
                    field="longitude",
                    issue_type="format",
                    severity="high",
                    message=f"Invalid longitude format: {lon}",
                    current_value=lon,
                    suggested_fix="Use numeric longitude",
                ))

        # Check last_updated format
        last_updated = poi.get("last_updated")
        if last_updated:
            try:
                datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            except ValueError:
                findings.append(AuditFinding(
                    poi_id=poi_id,
                    field="last_updated",
                    issue_type="format",
                    severity="medium",
                    message=f"Invalid date format (expected ISO 8601): {last_updated}",
                    current_value=last_updated,
                    suggested_fix="Use ISO 8601 format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
                ))

        return findings

    def _audit_freshness(self, poi: Dict, poi_id: str) -> List[AuditFinding]:
        """Check data freshness."""
        findings = []
        last_updated = poi.get("last_updated")

        if not last_updated:
            findings.append(AuditFinding(
                poi_id=poi_id,
                field="last_updated",
                issue_type="freshness",
                severity="medium",
                message="Missing last_updated field - cannot verify freshness",
                current_value=None,
                suggested_fix="Add last_updated field with ISO 8601 date",
            ))
            return findings

        try:
            update_date = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            age_days = (datetime.now(update_date.tzinfo) - update_date).days

            if age_days > self.freshness_threshold_days:
                findings.append(AuditFinding(
                    poi_id=poi_id,
                    field="last_updated",
                    issue_type="freshness",
                    severity="medium" if age_days < self.freshness_threshold_days * 2 else "high",
                    message=f"Data is {age_days} days old (threshold: {self.freshness_threshold_days} days)",
                    current_value=last_updated,
                    suggested_fix="Update data from source or verify current accuracy",
                ))

            # Check source_url accessibility (placeholder - would need HTTP check)
            source_url = poi.get("source_url")
            if source_url and not source_url.startswith(("http://", "https://")):
                findings.append(AuditFinding(
                    poi_id=poi_id,
                    field="source_url",
                    issue_type="freshness",
                    severity="low",
                    message=f"Source URL may be invalid: {source_url}",
                    current_value=source_url,
                    suggested_fix="Verify URL starts with http:// or https://",
                ))

        except ValueError:
            # Invalid date format already caught in format audit
            pass

        return findings

    def _audit_hallucination(self, poi: Dict, poi_id: str) -> List[AuditFinding]:
        """Use LLM to detect potential hallucinations."""
        findings = []

        # Build prompt for LLM
        prompt = self._build_hallucination_prompt(poi)

        try:
            response = self.llm_client.complete(
                messages=[
                    LLMMessage.system("You are a fact-checker for Taiwan tourism data. Identify potentially fabricated or incorrect information."),
                    LLMMessage.user(prompt)
                ],
                temperature=0.1,
            )

            result_text = response.content or ""
            findings.extend(self._parse_hallucination_result(poi_id, result_text))

        except Exception as e:
            print(f"  Warning: LLM hallucination check failed for {poi_id}: {e}")

        return findings

    def _build_hallucination_prompt(self, poi: Dict) -> str:
        """Build prompt for hallucination detection."""
        poi_json = json.dumps(poi, ensure_ascii=False, indent=2)

        return f"""請檢查以下台東觀光景點資料是否可能包含虛構/錯誤資訊：

{poi_json}

檢查重點：
1. **名稱是否真實存在** - 是否為已知景點/住宿/餐廳名稱
2. **地址格式** - 是否符合台灣地址格式（縣/市、鄉/鎮/市/區、路/街、號）
3. **經緯度合理性** - 是否落在台東縣合理範圍內 (北緯 22.0-23.5, 東經 120.5-121.8)
4. **類別與描述一致性** - 描述內容是否與 category 標籤相符
5. **電話/網址格式** - 電話是否為台灣格式，網址是否可疑
6. **誇大/錯誤宣稱** - 如聲稱「世界遺產」、「米其林」、「唯一」等需可驗證

輸出格式 (JSON Only)：
{{
  "issues": [
    {{
      "field": "欄位名稱",
      "issue": "具體問題描述",
      "severity": "high/medium/low",
      "confidence": 0.0-1.0
    }}
  ],
  "overall_assessment": "可信/存疑/明顯錯誤"
}}"""

    def _parse_hallucination_result(self, poi_id: str, result_text: str) -> List[AuditFinding]:
        """Parse LLM hallucination check result."""
        findings = []

        try:
            # Try to extract JSON from response
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(result_text[start:end])

                for issue in result.get("issues", []):
                    findings.append(AuditFinding(
                        poi_id=poi_id,
                        field=issue.get("field", "unknown"),
                        issue_type="hallucination",
                        severity=issue.get("severity", "medium"),
                        message=issue.get("issue", "Potential hallucination detected"),
                        current_value=poi.get(issue.get("field", "")),
                        suggested_fix="Verify with official source",
                    ))
        except json.JSONDecodeError:
            pass

        return findings

    def _aggregate_by_category(
        self,
        attractions: List[Dict],
        hallucination_findings: List[AuditFinding],
        format_issues: List[AuditFinding],
        freshness_issues: List[AuditFinding],
    ) -> Dict[str, Dict[str, int]]:
        """Aggregate findings by category."""
        by_category = {}

        # Initialize counters
        for poi in attractions:
            cat = poi.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = {"pass": 0, "warn": 0, "fail": 0, "total": 0}
            by_category[cat]["total"] += 1

        # Count findings by severity
        all_findings = hallucination_findings + format_issues + freshness_issues
        poi_severity = {}

        for finding in all_findings:
            pid = finding.poi_id
            if pid not in poi_severity:
                poi_severity[pid] = finding.severity
            else:
                # Keep highest severity
                severity_order = {"high": 3, "medium": 2, "low": 1}
                if severity_order.get(finding.severity, 0) > severity_order.get(poi_severity[pid], 0):
                    poi_severity[pid] = finding.severity

        # Map back to categories
        poi_to_cat = {poi.get("id", f"index_{i}"): poi.get("category", "unknown") for i, poi in enumerate(attractions)}

        for pid, severity in poi_severity.items():
            cat = poi_to_cat.get(pid, "unknown")
            if cat in by_category:
                if severity == "high":
                    by_category[cat]["fail"] += 1
                elif severity == "medium":
                    by_category[cat]["warn"] += 1
                else:
                    by_category[cat]["pass"] += 1

        # Fill in pass for those with no findings
        for cat, stats in by_category.items():
            accounted = stats["pass"] + stats["warn"] + stats["fail"]
            if accounted < stats["total"]:
                stats["pass"] += stats["total"] - accounted

        return by_category

    def _save_report(self, report: AuditReport, output_path: Path):
        """Save audit report as JSON and Markdown."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

        # Save Markdown
        md_path = output_path.with_suffix(".md")
        self._write_markdown_report(report, md_path)

        print(f"Report saved to {json_path} and {md_path}")

    def _write_markdown_report(self, report: AuditReport, md_path: Path):
        """Write human-readable Markdown audit report."""
        lines = [
            "# POI Content Audit Report",
            f"\nGenerated: {report.generated_at}",
            f"Data Source: {report.data_source}",
            "\n## Summary",
            f"- Total POIs: {report.summary['total_pois']}",
            f"- Passed: {report.summary['passed']}",
            f"- Warnings: {report.summary['warnings']}",
            f"- Failures: {report.summary['failures']}",
            f"- Hallucination Findings: {report.summary['hallucination_count']}",
            f"- Format Issues: {report.summary['format_issues_count']}",
            f"- Freshness Issues: {report.summary['freshness_issues_count']}",
            "\n## By Category",
        ]

        for cat, stats in sorted(report.by_category.items()):
            lines.append(f"\n### {cat}")
            lines.append(f"- Total: {stats['total']}")
            lines.append(f"- Pass: {stats['pass']}")
            lines.append(f"- Warnings: {stats['warn']}")
            lines.append(f"- Failures: {stats['fail']}")

        if report.hallucination_findings:
            lines.append("\n## Hallucination Findings (Top 10)")
            for f in report.hallucination_findings[:10]:
                lines.append(f"- **{f['poi_id']}** ({f['field']}) [{f['severity']}]: {f['message']}")

        if report.format_issues:
            lines.append("\n## Format Issues (Top 10)")
            for f in report.format_issues[:10]:
                lines.append(f"- **{f['poi_id']}** ({f['field']}) [{f['severity']}]: {f['message']}")

        if report.freshness_issues:
            lines.append("\n## Freshness Issues (Top 10)")
            for f in report.freshness_issues[:10]:
                lines.append(f"- **{f['poi_id']}** ({f['field']}) [{f['severity']}]: {f['message']}")

        md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Audit POI content for quality issues")
    parser.add_argument("--data", default="data/processed/attractions.json", help="Path to attractions JSON")
    parser.add_argument("--output", default="evaluation/reports/poi_audit", help="Output path (without extension)")
    parser.add_argument("--judge-model", default="poolside/laguna-s-2.1:free", help="LLM Judge model")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--freshness-days", type=int, default=730, help="Freshness threshold in days")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of POIs to audit (0 = all)")
    parser.add_argument("--skip-hallucination", action="store_true", help="Skip LLM hallucination check for faster testing")

    args = parser.parse_args()

    auditor = POIContentAuditor(
        data_path=Path(args.data),
        judge_model=args.judge_model,
        api_key=args.api_key,
        freshness_threshold_days=args.freshness_days,
    )

    report = auditor.audit(Path(args.output), limit=args.limit, skip_hallucination=args.skip_hallucination)

    # Print summary
    print("\n=== POI Audit Summary ===")
    print(f"Total: {report.summary['total_pois']}")
    print(f"Passed: {report.summary['passed']}")
    print(f"Warnings: {report.summary['warnings']}")
    print(f"Failures: {report.summary['failures']}")
    print(f"Categories: {list(report.by_category.keys())}")


if __name__ == "__main__":
    main()