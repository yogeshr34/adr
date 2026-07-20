import type { AnalyzePrescriptionResponse } from "@/lib/api";

const SEVERITY_STYLES: Record<string, { badge: string; ring: string; bar: string }> = {
  Low: { badge: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30", ring: "text-emerald-400", bar: "bg-emerald-500" },
  Moderate: { badge: "bg-amber-500/15 text-amber-400 border-amber-500/30", ring: "text-amber-400", bar: "bg-amber-500" },
  High: { badge: "bg-red-500/15 text-red-400 border-red-500/30", ring: "text-red-400", bar: "bg-red-500" },
};

export default function RiskSummary({ result }: { result: AnalyzePrescriptionResponse }) {
  const style = SEVERITY_STYLES[result.severity] ?? SEVERITY_STYLES.Moderate;
  const pct = Math.round(result.adr_risk_score * 100);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-wide text-slate-500">ADR Risk Assessment</p>
          <div className="mt-1 flex items-center gap-3">
            <span className="text-4xl font-bold text-slate-50">{pct}%</span>
            <span className={`rounded-full border px-3 py-1 text-sm font-medium ${style.badge}`}>
              {result.severity} risk
            </span>
          </div>
        </div>
        <div className="text-right text-sm text-slate-400">
          <p>Confidence: <span className="text-slate-200">{Math.round(result.confidence * 100)}%</span></p>
          <p>
            PRR proxy: <span className="text-slate-200">{result.prr.toFixed(2)}</span>
            {result.prr_signal && <span className="ml-1 text-amber-400">(signal ≥ 2.0)</span>}
          </p>
          <p>Patient age: <span className="text-slate-200">{result.patient_age}{result.age_estimated ? " (est.)" : ""}</span></p>
        </div>
      </div>

      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full ${style.bar}`} style={{ width: `${pct}%` }} />
      </div>

      <div className="mt-5 rounded-lg bg-slate-950/60 p-4">
        <p className="text-sm font-medium text-slate-300">Recommendation</p>
        <p className="mt-1 text-slate-100">{result.recommendation}</p>
      </div>

      {result.conditions.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {result.conditions.map((c) => (
            <span key={c} className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300">
              {c.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}

      {result.warnings.length > 0 && (
        <ul className="mt-4 space-y-1 text-sm text-amber-400">
          {result.warnings.map((w) => (
            <li key={w}>⚠ {w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
