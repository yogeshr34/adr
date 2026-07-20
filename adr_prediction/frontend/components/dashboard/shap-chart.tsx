import type { AnalyzePrescriptionResponse } from "@/lib/api";

const FEATURE_LABELS: Record<string, string> = {
  age: "Patient age",
  drug_count: "Number of drugs",
  condition_count: "Number of conditions",
  high_risk_drug_count: "High-risk drugs",
  max_dosage_ratio: "Dosage vs. typical max",
  interaction_score: "Interaction severity",
  polypharmacy: "Polypharmacy (5+ drugs)",
};

export default function ShapChart({ result }: { result: AnalyzePrescriptionResponse }) {
  const entries = Object.entries(result.shap_values);

  if (entries.length === 0) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
        <p className="text-sm font-medium text-slate-300">Why this score? (SHAP)</p>
        <p className="mt-2 text-sm text-slate-500">Explanation unavailable for this prediction.</p>
      </div>
    );
  }

  const sorted = [...entries].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.0001);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
      <p className="text-sm font-medium text-slate-300">Why this score? (SHAP feature contributions)</p>
      <p className="mt-1 text-xs text-slate-500">Red pushes risk up, green pushes risk down.</p>
      <div className="mt-4 space-y-3">
        {sorted.map(([feature, value]) => {
          const widthPct = (Math.abs(value) / maxAbs) * 100;
          const positive = value >= 0;
          return (
            <div key={feature} className="grid grid-cols-[9rem_1fr] items-center gap-3 text-sm">
              <span className="text-slate-400">{FEATURE_LABELS[feature] ?? feature}</span>
              <div className="flex h-4 items-center">
                <div className="relative h-2 w-full rounded bg-slate-800">
                  <div
                    className={`absolute top-0 h-2 rounded ${positive ? "left-1/2 bg-red-500" : "right-1/2 bg-emerald-500"}`}
                    style={{ width: `${widthPct / 2}%` }}
                  />
                  <div className="absolute left-1/2 top-0 h-2 w-px bg-slate-600" />
                </div>
                <span className={`ml-2 w-16 text-xs ${positive ? "text-red-400" : "text-emerald-400"}`}>
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(3)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
